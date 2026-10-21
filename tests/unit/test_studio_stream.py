"""The Studio's stream: the hub's ids, buffer, snapshot and ping, and the SSE route."""

import asyncio
import json
import socket
import threading
import time
from contextlib import suppress

import httpx
import pytest

from fake_backend import write_model, write_sized_model
from mlx_decision.memory import working_set
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio, server
from mlx_decision.studio.stream import Hub, Subscriber, encode

BODY = {"state": "Stripe is down", "questions": {"urgent": {"type": "noul"}}, "model": "fake"}
SET = {"state": "x", "questions": {}}


def numbers(items) -> list[int]:
    return [int(message_id.rsplit("-", 1)[1]) for message_id, _ in items]


# The hub


def test_ids_count_up_and_name_the_process():
    hub = Hub()
    ids = [hub.publish("workspace", {"n": n}) for n in range(3)]
    assert ids == [f"{hub.process}-1", f"{hub.process}-2", f"{hub.process}-3"]
    assert hub.last_id == ids[-1]
    assert Hub().process != hub.process


def test_since_replays_what_is_kept():
    hub = Hub(keep=5)
    for n in range(8):
        hub.publish("workspace", {"n": n})
    assert numbers(hub.since(hub.id_of(5))) == [6, 7, 8]
    assert hub.since(hub.id_of(8)) == []
    assert numbers(hub.since(hub.id_of(3))) == [4, 5, 6, 7, 8]  # 4 is the oldest kept
    assert hub.since(hub.id_of(2)) is None  # 3 is no longer kept: a gap
    assert hub.since(hub.id_of(9)) is None  # never sent
    assert hub.since("deadbeef-5") is None  # an earlier process
    for bad in (None, "", "5", f"{hub.process}-", f"{hub.process}-x", f"{hub.process}--1"):
        assert hub.since(bad) is None


def test_since_before_anything_was_published():
    hub = Hub()
    assert hub.since(hub.id_of(0)) == []
    assert hub.since(hub.id_of(1)) is None


def test_a_fresh_subscriber_gets_a_snapshot_with_the_latest_id():
    hub = Hub(snapshot=lambda: {"status": {"busy": None}})
    hub.publish("workspace", {"n": 1})
    subscriber, first = hub.subscribe()
    assert first == [(hub.id_of(1), {"type": "snapshot", "status": {"busy": None}})]
    assert subscriber in hub.subscribers
    _, first = hub.subscribe(hub.id_of(0))
    assert first == [(hub.id_of(1), {"type": "workspace", "n": 1})]


def test_two_subscribers_get_the_same_ids():
    hub = Hub()
    a, _ = hub.subscribe()
    b, _ = hub.subscribe()
    for n in range(3):
        hub.publish("workspace", {"n": n})
    drained = [[q.queue.get_nowait() for _ in range(3)] for q in (a, b)]
    assert drained[0] == drained[1]
    assert numbers(drained[0]) == [1, 2, 3]


def test_publish_changed_skips_repeats():
    hub = Hub()
    assert hub.publish_changed("status", {"busy": None}) is not None
    assert hub.publish_changed("status", {"busy": None}) is None
    assert hub.publish_changed("status", {"busy": "fit"}) is not None
    assert hub.publish_changed("other", {"busy": "fit"}) is not None
    assert hub.count == 3


def test_a_subscriber_that_falls_behind_is_dropped(monkeypatch):
    import mlx_decision.studio.stream as stream

    monkeypatch.setattr(stream, "QUEUE", 2)
    hub = Hub()
    subscriber, _ = hub.subscribe()
    for n in range(3):
        hub.publish("workspace", {"n": n})
    assert subscriber.dropped and subscriber not in hub.subscribers


def test_encode():
    assert encode("p-3", {"type": "x", "text": "ü"}) == 'id: p-3\ndata: {"type":"x","text":"ü"}\n\n'


async def take(events, count: int) -> list[str]:
    return [await anext(events) for _ in range(count)]


def test_events_retry_first_then_messages_then_ping():
    async def main():
        hub = Hub(snapshot=lambda: {"a": 1}, ping_seconds=0.05)
        subscriber, first = hub.subscribe()
        events = hub.events(subscriber, first)
        assert await take(events, 2) == [
            "retry: 1000\n\n",
            encode(hub.id_of(0), {"type": "snapshot", "a": 1}),
        ]
        hub.publish("workspace", {"n": 1})
        assert await anext(events) == encode(hub.id_of(1), {"type": "workspace", "n": 1})
        assert await anext(events) == ": ping\n\n"
        hub.close()
        with pytest.raises(StopAsyncIteration):
            await anext(events)
        assert not hub.subscribers

    asyncio.run(main())


def test_a_dropped_subscriber_ends_after_what_it_has(monkeypatch):
    import mlx_decision.studio.stream as stream

    monkeypatch.setattr(stream, "QUEUE", 1)

    async def main():
        hub = Hub()
        subscriber, first = hub.subscribe()
        events = hub.events(subscriber, first)
        await take(events, 2)
        hub.publish("workspace", {"n": 1})
        hub.publish("workspace", {"n": 2})  # the queue is full: dropped
        assert await anext(events) == encode(hub.id_of(1), {"type": "workspace", "n": 1})
        with pytest.raises(StopAsyncIteration):
            await anext(events)
        # The browser reconnects from its last id: nothing is missing.
        assert numbers(hub.since(hub.id_of(1))) == [2]

    asyncio.run(main())


def test_soon_runs_on_the_loop_from_another_thread():
    async def main():
        hub = Hub()
        hub.attach(asyncio.get_running_loop())
        seen = []
        thread = threading.Thread(
            target=hub.soon, args=(lambda: seen.append(threading.get_ident()),)
        )
        thread.start()
        thread.join()
        await asyncio.sleep(0.01)
        assert seen == [threading.get_ident()]

    asyncio.run(main())
    Hub().soon(lambda: None)  # without a loop: dropped


def test_subscriber_put_reports_a_full_queue():
    async def main():
        subscriber = Subscriber()
        subscriber.queue = asyncio.Queue(maxsize=1)
        assert subscriber.put(("p-1", {}))
        assert not subscriber.put(("p-2", {}))
        assert subscriber.dropped

    asyncio.run(main())


# The route, on a real server


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Studio:
    """A Studio served by uvicorn in a thread, with a logged-in client."""

    def __init__(self, tmp_path):
        # Two models, so no default loads at start; sized, so memory counts.
        write_sized_model(tmp_path / "models" / "fake", 1000)
        write_model(tmp_path / "models" / "other")
        self.port = free_port()
        pool = ModelPool(discover([tmp_path / "models"]), budget=10**9)
        self.app = create_studio(pool, port=self.port, workspace=tmp_path / "ws")
        self.server = server(self.app, port=self.port, log_level="warning", lifespan="on")
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            assert time.monotonic() < deadline, "the server did not start"
            time.sleep(0.01)
        self.base = f"http://127.0.0.1:{self.port}"
        token = self.app.state.session.token
        self.client = httpx.Client(base_url=self.base, timeout=10)
        response = self.client.get(f"/?token={token}")
        assert response.status_code == 303 and self.client.cookies, response.text

    def stream(self, last_event_id: str | None = None):
        headers = {"Last-Event-ID": last_event_id} if last_event_id else {}
        return self.client.stream("GET", "/studio/api/stream", headers=headers)

    def stop(self) -> float:
        started = time.monotonic()
        self.server.should_exit = True
        self.thread.join(15)
        assert not self.thread.is_alive(), "the server did not stop"
        return time.monotonic() - started


@pytest.fixture
def studio(tmp_path):
    studio = Studio(tmp_path)
    yield studio
    if studio.thread.is_alive():
        studio.stop()
    studio.client.close()


def read_messages(lines, count: int) -> list[tuple[str, dict]]:
    """``count`` messages (id, data) from an SSE response's lines; skips the rest."""
    messages, message_id = [], None
    for line in lines:
        if line.startswith("id: "):
            message_id = line[4:]
        elif line.startswith("data: "):
            messages.append((message_id, json.loads(line[6:])))
            if len(messages) == count:
                return messages
    raise AssertionError(f"the stream ended after {messages}")


def test_the_stream_starts_with_a_snapshot_and_sends_changes(studio):
    with studio.stream() as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-store"
        lines = response.iter_lines()
        assert next(lines) == "retry: 1000"
        [(snapshot_id, snapshot)] = read_messages(lines, 1)
        assert snapshot == {
            "type": "snapshot",
            "status": {
                "busy": None,
                "loaded": [],
                "groups": [],
                "used": {},
                "memory": {"used": 0, "budget": 10**9, "working_set": working_set()},
            },
            "jobs": [],
            "batch": None,  # the running batch
        }
        assert snapshot_id == studio.app.state.hub.last_id
        # A request through the Jev API loads the model: a status message.
        assert studio.client.post("/v1/systemone", json=BODY).status_code == 200
        [(_, loaded)] = read_messages(lines, 1)
        assert loaded["type"] == "status" and loaded["loaded"] == ["fake"]
        assert loaded["memory"]["used"] > 0
        # The same request again changes only when the model was last used (a
        # message only when that moved to the next second). A set saved sends one.
        assert studio.client.post("/v1/systemone", json=BODY).status_code == 200
        studio.app.state.workspace.save_set("tickets", SET)
        [(_, saved)] = read_messages(lines, 1)
        if saved["type"] == "status":
            assert {**saved, "used": None} == {**loaded, "used": None}
            [(_, saved)] = read_messages(lines, 1)
        assert saved == {
            "type": "workspace",
            "folder": "sets",
            "name": "tickets",
            "action": "saved",
        }


def test_status_follows_the_gpu_holder(studio):
    gpu, hub = studio.app.state.gpu, studio.app.state.hub
    with studio.stream() as response:
        lines = response.iter_lines()
        read_messages(lines, 1)  # the snapshot
        studio.client.post("/v1/systemone", json=BODY)
        read_messages(lines, 1)  # loaded
        future = asyncio.run_coroutine_threadsafe(gpu.acquire("fit-1"), hub.loop)
        future.result(5)
        messages = [data for _, data in read_messages(lines, 2)]
        assert messages[0]["busy"] == "fit-1" and messages[0]["loaded"] == ["fake"]
        assert messages[1]["busy"] == "fit-1" and messages[1]["loaded"] == []
        assert messages[1]["memory"]["used"] == 0
        assert studio.client.post("/v1/systemone", json=BODY).status_code == 503
        hub.loop.call_soon_threadsafe(gpu.release, "fit-1")
        [(_, released)] = read_messages(lines, 1)
        assert released["busy"] is None


def test_replay_after_a_dropped_connection_has_no_gap(studio):
    files, hub = studio.app.state.workspace, studio.app.state.hub
    with studio.stream() as response:
        lines = response.iter_lines()
        read_messages(lines, 1)
        files.save_set("one", SET)
        [(last_id, _)] = read_messages(lines, 1)
    # Disconnected: these happen while the tab is gone.
    for name in ("two", "three", "four"):
        files.save_set(name, SET)
    deadline = time.monotonic() + 5
    while hub.count < int(last_id.rsplit("-", 1)[1]) + 3:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    with studio.stream(last_id) as response:
        lines = response.iter_lines()
        replayed = read_messages(lines, 3)
        files.save_set("five", SET)
        replayed += read_messages(lines, 1)
    assert [data["name"] for _, data in replayed] == ["two", "three", "four", "five"]
    first = int(last_id.rsplit("-", 1)[1])
    assert numbers(replayed) == [first + 1, first + 2, first + 3, first + 4]


def test_an_unknown_id_gets_a_snapshot(studio):
    with studio.stream("0123abcd-7") as response:
        [(_, message)] = read_messages(response.iter_lines(), 1)
    assert message["type"] == "snapshot"


def test_two_tabs_see_the_same_ids(studio):
    files = studio.app.state.workspace
    with studio.stream() as one, studio.stream() as two:
        lines_one, lines_two = one.iter_lines(), two.iter_lines()
        snapshots = read_messages(lines_one, 1) + read_messages(lines_two, 1)
        assert snapshots[0][0] == snapshots[1][0]
        for name in ("a", "b", "c"):
            files.save_set(name, SET)
        assert read_messages(lines_one, 3) == read_messages(lines_two, 3)


def test_the_stream_needs_the_session(studio):
    response = httpx.get(f"{studio.base}/studio/api/stream")
    assert response.status_code == 401


def test_the_studio_stops_with_a_stream_open(studio):
    with studio.stream() as response:
        lines = response.iter_lines()
        read_messages(lines, 1)
        assert studio.stop() < 5
        with suppress(httpx.HTTPError):  # the stream ended, cleanly or cut
            assert [line for line in lines if line.startswith("data: ")] == []
