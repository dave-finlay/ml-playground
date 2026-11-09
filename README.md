# ml-playground

Small machine-learning experiments, built up one at a time. The first
area is reinforcement learning, starting with CartPole; other areas
(CNNs, RNNs and LSTMs, speech recognition, ...) may follow.

## Layout

Each experiment gets its own top-level directory with its own README
saying what it does, how to run it and what was learned. Code shared
between experiments (plotting, logging and so on) lives at the top
level once there's a second user for it.

## Experiments

None yet.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for Python and
dependencies. With uv installed:

```sh
uv sync                       # create .venv and install the locked dependencies
uv run python <experiment>/<script>.py
```

`uv.lock` is committed, so installs use exact versions. `pyproject.toml`
sets `exclude-newer = "7 days"`, so uv won't pick up package releases
less than a week old.

## License

[MIT](LICENSE)
