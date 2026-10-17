"""The GPU's worker and its handover, through the fake backend."""

import asyncio
import threading

import pytest
from fastapi.testclient import TestClient

from fake_backend import FakeBackend, write_model
from mlx_decision import DecisionModel
from mlx_decision.gpu import RETRY_AFTER, Busy, Gpu
from mlx_decision.pool import ModelPool, discover
from mlx_decision.server import create_app

BODY = {"state": "Stripe is down", "questions": {"urgent": {"type": "noul"}}}


class Recorder:
    """A pool's load function that notes each load and whether the GPU was held."""

    def __init__(self):
        self.gpu: Gpu | None = None
        self.loads: list[tuple[str, str | None]] = []

    def __call__(self, spec):
        self.loads.append((spec.name, self.gpu.holder))
        return DecisionModel(FakeBackend())


@pytest.fixture
def gpu(tmp_path):
    for name in ("alpha", "beta", "gamma"):
        write_model(tmp_path / "models" / name)
    recorder = Recorder()
    gpu = Gpu(ModelPool(discover([tmp_path / "models"]), budget=10**9, load=recorder))
    recorder.gpu = gpu
    gpu.recorder = recorder
    yield gpu
    gpu.close()


def test_requests_queued_across_acquire_never_load_a_model_while_held(gpu):
    pool, gate = gpu.pool, threading.Event()

    def first():  # the request on the worker when the job asks for the GPU
        model = pool.get("alpha")
        assert gate.wait(5)
        return model.decide_request(BODY).model

    async def scenario():
        running = asyncio.ensure_future(gpu.run(first))
        queued = [asyncio.ensure_future(gpu.run(pool.get, name)) for name in ("beta", "gamma")]
        await asyncio.sleep(0.05)  # all three are on the worker's queue
        handover = asyncio.ensure_future(gpu.acquire("training fit"))
        while gpu.holder is None:
            await asyncio.sleep(0)
        with pytest.raises(Busy):  # at the door
            gpu.check()
        late = asyncio.ensure_future(gpu.run(pool.get, "beta"))
        gate.set()
        results = await asyncio.gather(running, *queued, late, return_exceptions=True)
        await handover
        return results

    answered, *refused = asyncio.run(scenario())
    assert answered == "alpha"  # the request it was on finishes
    assert all(isinstance(error, Busy) for error in refused)
    assert refused[0].holder == "training fit"
    assert "busy with training fit" in str(refused[0])
    assert gpu.recorder.loads == [("alpha", None)]
    assert pool.loaded() == []  # unloaded after that request

    gpu.release("training fit")
    asyncio.run(gpu.run(pool.get, "beta"))
    assert gpu.recorder.loads[-1] == ("beta", None)


def test_one_holder_at_a_time_and_only_it_releases(gpu):
    asyncio.run(gpu.acquire("convert"))
    with pytest.raises(Busy, match="busy with convert"):
        asyncio.run(gpu.acquire("training fit"))
    gpu.release("training fit")  # not the holder: nothing changes
    assert gpu.holder == "convert"
    # Bookkeeping that loads nothing still runs while held.
    assert asyncio.run(gpu.run(lambda: gpu.pool.names, check=False)) == ["alpha", "beta", "gamma"]
    gpu.release("convert")
    assert gpu.holder is None
    asyncio.run(gpu.run(gpu.pool.get, "alpha"))
    assert gpu.pool.loaded() == ["alpha"]


def test_a_failed_unload_gives_the_gpu_back(gpu, monkeypatch):
    def broken():
        raise RuntimeError("cannot unload")

    monkeypatch.setattr(gpu.pool, "unload_all", broken)
    with pytest.raises(RuntimeError, match="cannot unload"):
        asyncio.run(gpu.acquire("convert"))
    assert gpu.holder is None


def test_the_api_answers_503_while_held_without_reading_the_body(tmp_path):
    loads = []

    def load(spec):
        loads.append(spec.name)
        return DecisionModel(FakeBackend())

    pool = ModelPool(discover([write_model(tmp_path / "fake")]), budget=10**9, load=load)
    gpu = Gpu(pool)
    with TestClient(create_app(gpu)) as client:
        assert loads == ["fake"]  # the default model, at startup
        asyncio.run(gpu.acquire("training fit"))
        assert pool.loaded() == []
        response = client.post("/v1/systemone", content=b"not even JSON")
        assert response.status_code == 503
        assert response.headers["retry-after"] == str(RETRY_AFTER)
        error = response.json()["error"]
        assert error["code"] == "busy" and error["type"] == "api_error"
        assert "training fit" in error["message"]
        assert client.get("/health").json()["loaded"] == []
        gpu.release("training fit")
        assert client.post("/v1/systemone", json=BODY).status_code == 200
        assert loads == ["fake", "fake"]
