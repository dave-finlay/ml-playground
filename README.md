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

| Directory | What's in it |
|---|---|
| [cartpole](cartpole/README.md) | CartPole: so far, watching the pole fall with no control |

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for Python and
dependencies. With uv installed:

```sh
uv sync                       # create .venv and install the locked dependencies
uv run python <experiment>/<script>.py
```

To lint and format with [ruff](https://docs.astral.sh/ruff/) (a dev
dependency, so `uv sync` installs it; rules are in `pyproject.toml`):

```sh
uv run ruff check     # look for likely bugs and bad habits
uv run ruff format    # lay the code out consistently
```

To run the tests with [pytest](https://docs.pytest.org/) (also a dev
dependency; each experiment keeps its tests in its own `tests/`
directory, listed in `pyproject.toml`):

```sh
uv run pytest
```

`uv.lock` is committed, so installs use exact versions. `pyproject.toml`
sets `exclude-newer = "7 days"`, so uv won't pick up package releases
less than a week old.

## License

[MIT](LICENSE), except files whose header says otherwise. Code copied
from elsewhere keeps its original license, and its header says where it
came from.
