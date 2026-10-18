"""The config file: values, warnings and errors, and the commands' flag > file > default."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fake_backend import write_model, write_sized_model
from mlx_decision.cli import app
from mlx_decision.config import Config, ConfigError, home, load


def write_config(folder: Path, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "config.toml"
    path.write_text(text)
    return path


# the file


def test_home_is_the_variable_or_a_folder_in_the_user_home(monkeypatch, tmp_path):
    monkeypatch.setenv("MLX_DECISION_HOME", str(tmp_path / "elsewhere"))
    assert home() == tmp_path / "elsewhere"
    monkeypatch.setenv("MLX_DECISION_HOME", "~/x")
    assert home() == Path.home() / "x"
    monkeypatch.delenv("MLX_DECISION_HOME")
    assert home() == Path.home() / ".mlx-decision"


def test_without_a_file_nothing_is_set(mlx_decision_home):
    config = load()
    assert config == Config(mlx_decision_home / "config.toml")
    assert config.warnings == () and config.models == () and config.port is None


def test_the_six_keys(tmp_path):
    path = write_config(
        tmp_path / "home",
        """
        models = ["~/Models/laya", "./mine", "Cloudflare/clef-flash"]
        default_model = "laya"
        memory_budget_gb = 24
        host = "0.0.0.0"
        port = 8100
        workspace = "studio"
        """,
    )
    config = load(path)
    assert config.models == (
        str(Path.home() / "Models/laya"),
        str(tmp_path / "home/./mine"),
        "Cloudflare/clef-flash",
    )
    assert config.default_model == "laya"
    assert config.memory_budget_gb == 24.0 and isinstance(config.memory_budget_gb, float)
    assert (config.host, config.port) == ("0.0.0.0", 8100)
    assert config.workspace == tmp_path / "home/studio"
    assert config.warnings == ()


def test_an_unknown_key_is_a_warning_naming_file_and_key(tmp_path):
    path = write_config(tmp_path, 'prot = 8100\napi_key = "secret"\nport = 8200\n')
    config = load(path)
    assert config.port == 8200
    assert len(config.warnings) == 2
    assert config.warnings[0] == f"{path}: unknown key 'prot' (it is not read)"
    assert "--api-key" in config.warnings[1] and "MLX_DECISION_API_KEY" in config.warnings[1]
    assert "secret" not in config.warnings[1]


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('models = "~/Models/laya"', "models must be a list of models"),
        ("models = [1]", "models entries must be a string, not a whole number"),
        ('models = [""]', "models entries must not be empty"),
        ("default_model = true", "default_model must be a string, not true or false"),
        ('memory_budget_gb = "24"', "memory_budget_gb must be a number of GB, not a string"),
        ("memory_budget_gb = -1", "memory_budget_gb must not be negative"),
        ("host = 127", "host must be a string, not a whole number"),
        ('port = "8000"', "port must be a whole number, not a string"),
        ("port = 80.5", "port must be a whole number, not a number"),
        ("port = 70000", "port must be between 1 and 65535"),
        ("workspace = []", "workspace must be a string, not a list"),
        ("[workspace]", "workspace must be a string, not a table"),
    ],
)
def test_a_wrong_type_is_an_error_naming_file_and_key(tmp_path, line, message):
    path = write_config(tmp_path, line + "\n")
    with pytest.raises(ConfigError) as error:
        load(path)
    assert str(error.value).startswith(f"{path}: {message}")


def test_a_file_that_is_not_toml_is_an_error_naming_it(tmp_path):
    path = write_config(tmp_path, "port = \n")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load(path)


def test_the_flag_wins_over_the_file_and_the_file_over_the_default(tmp_path):
    config = Config(tmp_path / "config.toml", port=8100, models=("a",))
    assert config.resolve(9000, "port", 8000) == 9000
    assert config.resolve(None, "port", 8000) == 8100
    assert config.resolve(None, "host", "127.0.0.1") == "127.0.0.1"
    assert config.resolve(None, "memory_budget_gb") is None
    assert Config(tmp_path).resolve(None, "models", ["x"]) == ["x"]


# the default model


@pytest.fixture
def folders(tmp_path):
    """A model folder ``laya`` and a folder of models with ``one`` and ``two``."""
    write_model(tmp_path / "laya")
    write_model(tmp_path / "many/one")
    write_model(tmp_path / "many/two")
    return tmp_path


def test_the_default_model_is_found_among_the_models(folders):
    models = (str(folders / "laya"), str(folders / "many"), "Cloudflare/clef-flash")
    config = Config(folders / "config.toml", models=models)

    def reference(name):
        return Config(config.path, models=models, default_model=name).default_reference()

    assert reference("laya") == str(folders / "laya")
    assert reference("two") == str(folders / "many/two")
    assert reference("clef-flash") == "Cloudflare/clef-flash"
    assert reference("Cloudflare/clef-flash") == "Cloudflare/clef-flash"
    assert reference("org/other") == "org/other"  # a reference of its own
    assert config.default_reference() is None
    with pytest.raises(ConfigError, match="default_model names none of the models in models: x"):
        reference("x")


def test_without_models_the_default_model_is_a_reference(folders):
    config = Config(folders / "config.toml", default_model=str(folders / "laya"))
    assert config.default_reference() == str(folders / "laya")
    assert config.model_references() == [str(folders / "laya")]
    assert config.default_name(["laya", "other"]) == "laya"
    assert Config(folders, default_model="laya").default_reference() == "laya"


def test_a_default_model_of_its_own_joins_the_models(folders):
    models = (str(folders / "many"),)
    assert Config(folders, models=models, default_model="one").model_references() == [
        str(folders / "many")
    ]
    other = Config(folders, models=models, default_model=str(folders / "laya"))
    assert other.model_references() == [str(folders / "many"), str(folders / "laya")]
    assert Config(folders).model_references() == []


# the commands


def run_fake(*args):
    return CliRunner().invoke(app, ["run", "-s", "x", "--noul", "q=Is it?", "--json", *args])


def test_run_without_m_takes_the_default_model(mlx_decision_home, fake_model_path):
    missing = run_fake()
    assert missing.exit_code == 1
    assert f"pass -m MODEL (or set default_model in {mlx_decision_home / 'config.toml'})" in (
        missing.stderr
    )
    write_config(mlx_decision_home, f'default_model = "{fake_model_path}"\n')
    result = run_fake()
    assert result.exit_code == 0, result.output
    assert "q" in json.loads(result.stdout)["answers"]


def test_run_names_a_wrong_value_and_warns_about_an_unknown_key(mlx_decision_home, fake_model_path):
    path = write_config(mlx_decision_home, "colour = 1\n")
    result = run_fake("-m", str(fake_model_path))
    assert result.exit_code == 0, result.output
    assert f"warning: {path}: unknown key 'colour'" in result.stderr
    write_config(mlx_decision_home, "port = true\n")
    result = run_fake("-m", str(fake_model_path))
    assert result.exit_code == 1
    assert f"error: {path}: port must be a whole number" in result.stderr


def test_training_eval_without_m_takes_the_default_model(mlx_decision_home, fake_model_path):
    data = mlx_decision_home / "labelled.jsonl"
    line = {"state": "x", "questions": {"q": {"type": "noul"}}, "labels": {"q": True}}
    data.write_text(json.dumps(line) + "\n")
    write_config(mlx_decision_home, f'default_model = "{fake_model_path}"\n')
    result = CliRunner().invoke(app, ["training", "eval", "--data", str(data), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["models"][0]["model"] == str(fake_model_path)


@pytest.fixture
def serve(monkeypatch):
    """The server command with the app and uvicorn replaced: (result, pool, uvicorn's
    keywords)."""
    uvicorn = pytest.importorskip("uvicorn")
    import mlx_decision.server as server

    def run(*args):
        seen = {}
        monkeypatch.setattr(server, "create_app", lambda pool, **_: pool)
        monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(pool=app, **kw))
        result = CliRunner().invoke(app, ["server", *map(str, args)])
        return result, seen.get("pool"), seen

    return run


@pytest.fixture
def models_folder(tmp_path):
    root = tmp_path / "models"
    for name in ("alpha", "beta"):
        write_sized_model(root / name, 1000)
    return root


def test_server_takes_models_host_port_budget_and_default_from_the_file(
    serve, mlx_decision_home, models_folder
):
    missing, _, _ = serve()
    assert (
        missing.exit_code == 1
        and "pass -m MODEL (or set models_dir or models in " in missing.stderr
    )
    write_config(
        mlx_decision_home,
        f"""
        models = ["{models_folder}"]
        default_model = "beta"
        memory_budget_gb = 2
        host = "0.0.0.0"
        port = 8100
        """,
    )
    result, pool, seen = serve()
    assert result.exit_code == 0, result.output
    assert pool.names == ["alpha", "beta"] and pool.default == "beta"
    assert pool.budget == 2 * 10**9
    assert (seen["host"], seen["port"]) == ("0.0.0.0", 8100)

    # Flags win; the file's default applies only to a model in use.
    other = write_sized_model(models_folder.parent / "other", 1000)
    args = ["--host", "127.0.0.1", "--port", 9000, "--memory-budget", 1]
    result, pool, seen = serve("-m", other, "-m", models_folder, *args)
    assert result.exit_code == 0, result.output
    assert pool.names == ["other", "alpha", "beta"] and pool.default == "beta"
    assert pool.budget == 10**9 and (seen["host"], seen["port"]) == ("127.0.0.1", 9000)
    result, pool, _ = serve("-m", other, "--default-model", "other")
    assert pool.default == "other"
    result, pool, _ = serve("-m", models_folder / "alpha")
    assert result.exit_code == 0, result.output
    assert pool.names == ["alpha"] and pool.default == "alpha"


def test_server_adds_a_default_model_of_its_own(serve, mlx_decision_home, models_folder):
    other = write_sized_model(models_folder.parent / "other", 1000)
    write_config(mlx_decision_home, f'models = ["{models_folder}"]\ndefault_model = "{other}"\n')
    result, pool, seen = serve()
    assert result.exit_code == 0, result.output
    assert pool.names == ["alpha", "beta", "other"] and pool.default == "other"
    assert (seen["host"], seen["port"]) == ("127.0.0.1", 8000)


def test_benchmark_without_m_runs_the_models_of_the_file(mlx_decision_home, tmp_path, monkeypatch):
    import mlx_decision.benchmark as benchmark

    calls = []

    def run(path, arguments):
        calls.append(Path(path).name)
        done = CliRunner().invoke(app, ["benchmark", "-m", path, "--json", *arguments])
        return done.exit_code, done.stdout, done.stderr

    monkeypatch.setattr(benchmark, "run_in_process", run)
    first, second = write_model(tmp_path / "first"), write_model(tmp_path / "second")
    write_config(mlx_decision_home, f'models = ["{first}", "{second}"]\n')
    args = ["benchmark", "--lengths", "20", "--questions", "1", "--repeats", "1", "--json"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert calls == ["first", "second"]
    assert [run["model"] for run in json.loads(result.stdout)] == ["first", "second"]


def test_mcp_without_m_or_a_default_model_says_how(mlx_decision_home):
    pytest.importorskip("mcp")
    result = CliRunner().invoke(app, ["mcp"])
    assert result.exit_code == 1
    assert "pass -m MODEL (or set default_model in " in result.stderr


# models_dir


@pytest.fixture
def models_dir(tmp_path):
    """A models_dir with ``alpha`` and ``beta`` and a subfolder that is no model."""
    root = tmp_path / "Decision"
    for name in ("alpha", "beta"):
        write_sized_model(root / name, 1000)
    (root / "notes").mkdir()
    return root


def test_models_dir_is_a_folder_or_the_cache(tmp_path, models_dir):
    home = tmp_path / "home"
    assert load(write_config(home, f'models_dir = "{models_dir}"\n')).models_dir == models_dir
    assert load(write_config(home, 'models_dir = "./mine"\n')).models_dir == home / "mine"
    assert load(write_config(home, 'models_dir = "~/m"\n')).models_dir == Path.home() / "m"
    config = load(write_config(home, 'models_dir = "hf-cache"\n'))
    assert config.models_dir == "hf-cache" and config.models_folder is None
    assert config.model_references() == [] and config.warnings == ()


def test_models_dir_errors_and_warnings(tmp_path):
    home = tmp_path / "home"
    path = write_config(home, "models_dir = 3\n")
    with pytest.raises(ConfigError, match=f"{path}: models_dir must be a string"):
        load(path)
    (home / "file").write_text("x")
    with pytest.raises(ConfigError, match="models_dir must be a folder"):
        load(write_config(home, 'models_dir = "./file"\n'))
    config = load(write_config(home, 'models_dir = "./missing"\n'))
    assert config.warnings == (f"{path}: models_dir {home / 'missing'} does not exist "
                               "(no models listed from it)",)  # fmt: skip
    assert config.models_folder is None and config.model_references() == []


def test_names_are_looked_up_in_models_dir_then_models(tmp_path, models_dir, folders):
    other = write_model(tmp_path / "elsewhere/alpha")
    models = (str(other), str(folders / "many"), "Cloudflare/clef-flash")
    config = Config(tmp_path / "config.toml", models=models, models_dir=models_dir)
    assert config.lookup("alpha") == str(models_dir / "alpha")  # models_dir first
    assert config.lookup("two") == str(folders / "many/two")
    assert config.lookup("clef-flash") == "Cloudflare/clef-flash"
    for reference in ("x", "./alpha", "~/alpha", "org/alpha"):
        assert config.lookup(reference) is None
    assert config.model_names() == ["alpha", "beta", "one", "two", "clef-flash"]
    assert Config(tmp_path, models_dir="hf-cache").lookup("alpha") is None


def test_the_default_model_is_found_in_models_dir(tmp_path, models_dir):
    config = Config(tmp_path / "config.toml", models_dir=models_dir, default_model="beta")
    assert config.default_reference() == str(models_dir / "beta")
    assert config.model_references() == [str(models_dir)]
    wrong = Config(config.path, models_dir=models_dir, default_model="gamma")
    with pytest.raises(ConfigError, match="none of the models in models_dir: gamma"):
        wrong.default_reference()


def test_models_dir_is_served_first_without_listing_a_model_twice(tmp_path, models_dir, folders):
    path = tmp_path / "config.toml"
    models = (str(folders / "laya"), str(models_dir / "alpha"), str(models_dir))
    config = Config(
        path, models=models, models_dir=models_dir, default_model=str(models_dir / "beta")
    )
    assert config.model_references() == [str(models_dir), str(folders / "laya")]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert Config(path, models=models[:1], models_dir=empty).model_references() == list(models[:1])


def test_run_takes_a_name_in_models_dir(mlx_decision_home, models_dir, tmp_path, monkeypatch):
    write_config(mlx_decision_home, f'models_dir = "{models_dir}"\n')
    result = run_fake("-m", "beta")
    assert result.exit_code == 0, result.output
    unknown = run_fake("-m", "gamma")
    assert unknown.exit_code == 1
    assert "no model named gamma in models_dir; there are: alpha, beta" in unknown.stderr
    # A folder of that name in the working directory wins over the name.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "beta").mkdir()
    shadowed = run_fake("-m", "beta")
    assert shadowed.exit_code == 1 and "beta" in shadowed.stderr


def test_without_models_dir_a_name_goes_to_the_loader_as_before(tmp_path):
    from mlx_decision.cli import model_reference

    # A repo id without an organisation, such as gpt2, stays one.
    assert model_reference("beta", Config(tmp_path / "config.toml")) == "beta"


def test_server_serves_models_dir_and_takes_names(serve, mlx_decision_home, models_dir):
    write_config(mlx_decision_home, f'models_dir = "{models_dir}"\ndefault_model = "beta"\n')
    result, pool, _ = serve()
    assert result.exit_code == 0, result.output
    assert pool.names == ["alpha", "beta"] and pool.default == "beta"
    result, pool, _ = serve("-m", "alpha")
    assert result.exit_code == 0, result.output
    assert pool.names == ["alpha"]


def test_training_eval_takes_names_for_both_models(mlx_decision_home, models_dir):
    data = mlx_decision_home / "labelled.jsonl"
    line = {"state": "x", "questions": {"q": {"type": "noul"}}, "labels": {"q": True}}
    data.write_text(json.dumps(line) + "\n")
    write_config(mlx_decision_home, f'models_dir = "{models_dir}"\n')
    args = ["training", "eval", "-m", "alpha", "--compare", "beta", "--data", str(data), "--json"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    models = [entry["model"] for entry in json.loads(result.stdout)["models"]]
    assert models == [str(models_dir / "alpha"), str(models_dir / "beta")]


# the one write: models_dir


def test_set_models_dir_creates_the_file(tmp_path):
    from mlx_decision.config import set_models_dir

    path = tmp_path / "new home" / "config.toml"
    assert set_models_dir(tmp_path / "Decision", path) == f'models_dir = "{tmp_path / "Decision"}"'
    assert load(path).models_dir == tmp_path / "Decision"
    other = tmp_path / "other.toml"
    assert set_models_dir(None, other) == 'models_dir = "hf-cache"'
    assert load(other).models_dir == "hf-cache"
    home_path = tmp_path / "home.toml"
    assert set_models_dir(Path.home() / "Data/Models", home_path) == 'models_dir = "~/Data/Models"'


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("# mine\nport = 8100  # note\n", '# mine\nport = 8100  # note\nmodels_dir = "{v}"\n'),
        ("port = 8100", 'port = 8100\nmodels_dir = "{v}"\n'),
        (
            'host = "x"\n\n[later]\nkey = 1\n',
            'host = "x"\n\nmodels_dir = "{v}"\n\n[later]\nkey = 1\n',
        ),
    ],
)
def test_set_models_dir_keeps_the_rest_of_the_file(tmp_path, before, after):
    from mlx_decision.config import set_models_dir

    path = tmp_path / "config.toml"
    path.write_text(before)
    folder = tmp_path / "M"
    set_models_dir(folder, path)
    assert path.read_text() == after.format(v=folder)


def test_set_models_dir_quotes_the_value(tmp_path):
    from mlx_decision.config import set_models_dir

    folder = tmp_path / 'with "quotes" and \\ back'
    path = tmp_path / "config.toml"
    set_models_dir(folder, path)
    assert load(path).models_dir == folder


def test_set_models_dir_refuses_what_it_cannot_do_cleanly(tmp_path):
    from mlx_decision.config import set_models_dir

    path = tmp_path / "config.toml"
    path.write_text('models_dir = "hf-cache"\n')
    with pytest.raises(ConfigError, match="already sets models_dir"):
        set_models_dir(None, path)
    path.write_text("port = \n")
    with pytest.raises(ConfigError, match="not valid TOML"):
        set_models_dir(None, path)
    nested = 'models = [\n["a"]\n]\n'  # a line that looks like a table
    path.write_text(nested)
    with pytest.raises(ConfigError, match="add models_dir"):
        set_models_dir(None, path)
    assert path.read_text() == nested
    assert not list(tmp_path.glob(".*.tmp"))
