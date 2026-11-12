"""Tests for config.py: defaults, config files and command-line overrides."""

import argparse
import dataclasses
import json

import pytest

from config import (
    add_setting_options,
    given_settings,
    load_config,
    make_config,
    resolve,
    save_config,
    setting,
)


@dataclasses.dataclass(frozen=True)
class Example:
    name: str = setting("", "a name")
    count: int = setting(3, "a count")
    rate: float = setting(0.5, "a rate")
    seed: int | None = setting(None, "a seed")
    mode: str = setting("a", "a mode", choices=("a", "b"))
    flag: bool = setting(False, "a flag")


def parse(argv):
    parser = argparse.ArgumentParser()
    add_setting_options(parser, Example)
    return parser.parse_args(argv)


def write(tmp_path, values):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values))
    return path


def test_defaults_when_nothing_given():
    assert resolve(Example, None, parse([])) == Example()


def test_file_overrides_defaults(tmp_path):
    path = write(tmp_path, {"count": 7, "mode": "b"})
    assert resolve(Example, path, parse([])) == Example(count=7, mode="b")


def test_command_line_overrides_file(tmp_path):
    path = write(tmp_path, {"count": 7, "rate": 0.1})
    config = resolve(Example, path, parse(["--count", "9"]))
    assert config == Example(count=9, rate=0.1)


def test_bool_option_both_ways(tmp_path):
    path = write(tmp_path, {"flag": True})
    assert resolve(Example, None, parse(["--flag"])).flag is True
    assert resolve(Example, path, parse(["--no-flag"])).flag is False


def test_given_settings_lists_only_command_line_options():
    assert given_settings(parse(["--count", "2", "--no-flag"]), Example) == [
        "--count",
        "--flag",
    ]


def test_unknown_key_rejected(tmp_path):
    path = write(tmp_path, {"cuont": 7})
    with pytest.raises(ValueError, match="unknown settings: cuont"):
        resolve(Example, path, parse([]))


def test_file_must_hold_an_object(tmp_path):
    path = write(tmp_path, [1, 2])
    with pytest.raises(ValueError, match="JSON object"):
        load_config(path, Example)


def test_int_accepted_for_float():
    config = make_config(Example, {"rate": 2})
    assert config.rate == 2.0
    assert type(config.rate) is float


@pytest.mark.parametrize(
    "values",
    [
        {"count": True},  # a bool isn't an int, though Python says it is
        {"count": 2.5},
        {"count": None},  # only seed allows null
        {"rate": "fast"},
        {"flag": 1},
        {"name": 3},
    ],
)
def test_wrong_type_rejected(values):
    with pytest.raises(ValueError, match="should be"):
        make_config(Example, values)


def test_null_allowed_where_declared():
    assert make_config(Example, {"seed": None}).seed is None
    assert make_config(Example, {"seed": 4}).seed == 4


def test_invalid_choice_rejected():
    with pytest.raises(ValueError, match="one of"):
        make_config(Example, {"mode": "c"})


def test_invalid_choice_rejected_on_command_line():
    with pytest.raises(SystemExit):
        parse(["--mode", "c"])


def test_saved_config_loads_back_the_same(tmp_path):
    config = Example(name="x", count=1, rate=0.25, seed=8, mode="b", flag=True)
    path = tmp_path / "saved.json"
    save_config(path, config)
    assert load_config(path, Example) == config
