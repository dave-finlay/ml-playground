"""Settings for an experiment script, gathered from three places.

Each setting has a built-in default, which a JSON config file can
override, which the command line can override in turn. The settings are
declared once, as fields of a dataclass made with setting(), and the
command-line options are generated from those fields: a field
learning_rate becomes --learning-rate.

    @dataclasses.dataclass(frozen=True)
    class Config:
        batches: int = setting(200, "number of weight updates")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_file")
    add_setting_options(parser, Config)
    args = parser.parse_args()
    config = resolve(Config, args.config_file, args)

A config file is a JSON object whose keys are field names. Unknown keys
and values of the wrong type are errors rather than being ignored, so a
typo can't silently leave a setting at its default. save_config writes
the resolved settings in the same format, so a run's saved settings can
be used as a config file to repeat it.
"""

import argparse
import dataclasses
import json
import types
import typing


def setting(default, help, choices=None):
    """Declare a dataclass field as a setting, with help text for its
    command-line option and, optionally, the values it may take."""
    return dataclasses.field(
        default=default, metadata={"help": help, "choices": choices}
    )


def option_name(name):
    """Command-line option for a setting: learning_rate -> --learning-rate."""
    return "--" + name.replace("_", "-")


def allowed_types(annotation):
    """Types a setting's annotation allows: int | None gives (int, NoneType)."""
    if isinstance(annotation, types.UnionType):
        return typing.get_args(annotation)
    return (annotation,)


def check_value(name, value, annotation, choices):
    """Return value if it suits the setting, converting an int to a float
    where a float is wanted. Raise ValueError if it doesn't suit."""
    allowed = allowed_types(annotation)
    if type(value) is int and float in allowed and int not in allowed:
        value = float(value)
    # Exact types, so that True isn't taken for the int 1.
    if type(value) not in allowed:
        names = " or ".join("null" if t is type(None) else t.__name__ for t in allowed)
        raise ValueError(f"{name} should be {names}, not {value!r}")
    if choices is not None and value not in choices:
        raise ValueError(f"{name} should be one of {list(choices)}, not {value!r}")
    return value


def make_config(config_class, values):
    """Build config_class from a dict of setting values, checking each one.
    Settings missing from values keep their defaults."""
    hints = typing.get_type_hints(config_class)
    checked = {}
    for field in dataclasses.fields(config_class):
        if field.name in values:
            checked[field.name] = check_value(
                field.name,
                values[field.name],
                hints[field.name],
                field.metadata["choices"],
            )
    return config_class(**checked)


def load_config_file(path, config_class):
    """Read a config file and return its settings as a dict, without
    defaults filled in. Raise ValueError for keys that aren't settings."""
    with open(path) as f:
        values = json.load(f)
    if not isinstance(values, dict):
        raise ValueError(f"{path} should hold a JSON object")
    known = {field.name for field in dataclasses.fields(config_class)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"{path} has unknown settings: {', '.join(unknown)}")
    return values


def load_config(path, config_class):
    """Read a config file into config_class, with defaults for any setting
    it leaves out."""
    return make_config(config_class, load_config_file(path, config_class))


def save_config(path, config):
    """Write every setting of config to path as a JSON config file."""
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)
        f.write("\n")


def describe_default(value):
    if value is None:
        return "none"
    if value == "":
        return "empty"
    return str(value)


def add_setting_options(parser, config_class):
    """Add a command-line option for each setting of config_class.

    The options default to None, meaning "not given", so that resolve can
    tell them apart from values given on the command line.
    """
    group = parser.add_argument_group(
        "settings", "each overrides the config file, which overrides the default"
    )
    hints = typing.get_type_hints(config_class)
    for field in dataclasses.fields(config_class):
        kind = next(t for t in allowed_types(hints[field.name]) if t is not type(None))
        help_text = field.metadata["help"].replace("%", "%%")
        kwargs = {
            "dest": field.name,
            "default": None,
            "help": f"{help_text} (default: {describe_default(field.default)})",
        }
        if kind is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = kind
            kwargs["choices"] = field.metadata["choices"]
        group.add_argument(option_name(field.name), **kwargs)


def given_settings(args, config_class):
    """Return the options for the settings given on the command line."""
    return [
        option_name(field.name)
        for field in dataclasses.fields(config_class)
        if getattr(args, field.name) is not None
    ]


def resolve(config_class, config_file, args):
    """Combine the defaults, the config file (if config_file isn't None)
    and the settings given on the command line, in that order of
    precedence from lowest to highest, into config_class."""
    values = {}
    if config_file is not None:
        values.update(load_config_file(config_file, config_class))
    for field in dataclasses.fields(config_class):
        value = getattr(args, field.name)
        if value is not None:
            values[field.name] = value
    return make_config(config_class, values)
