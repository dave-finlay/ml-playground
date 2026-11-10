# cartpole

Experiments with CartPole: a pole balanced on a cart that moves along a
frictionless track. The physics and renderer come from Gymnasium's
`CartPole-v1`, which follows Barto, Sutton and Anderson (1983).

## The system

The state is four numbers: cart position, cart velocity, pole angle and
pole angular velocity. Every 0.02 s (`tau`) the environment works out
the accelerations from the equations of motion and takes one Euler step.
The cart weighs 1 kg, the pole 0.1 kg and is 1 m long, and gravity is
9.8 m/s². Each episode starts with all four numbers random in ±0.05.

The only actions are 0 (push left) and 1 (push right), always with 10 N.
A normal episode ends when the pole tilts more than 12° or the cart goes
more than 2.4 m from the centre, or after 500 steps (10 s).

## cartpole.py: simulation and controllers

`cartpole.py` wraps Gymnasium's environment in a `CartPole` class, and
all access to the environment goes through it. Its `step(force)` takes
any push in newtons (positive right, negative left, 0 for none), which
standard CartPole doesn't allow; it does this by setting the
environment's internal `force_mag` before each step. Episodes run until
the pole passes 90° (standard CartPole stops at 12°).

A controller is a function from state to force. `run_simulation` runs
one episode with a given controller. So far there's one controller,
`no_control`, which always returns 0 N, so the pole just falls.

```sh
uv run python cartpole/cartpole.py              # open a window and watch
uv run python cartpole/cartpole.py --slow 4     # quarter speed
uv run python cartpole/cartpole.py --seed 7     # repeatable starting state
uv run python cartpole/cartpole.py --no-window  # just print the trace
```

With no control, the pole passes 12° after about 0.75 s and is flat after
about 1.3 s. The cart drifts a few centimetres the opposite way, pushed
by the falling pole. The window shows the starting position for a
second before the pole is released and holds the end position for two
seconds.

The equations don't model the track. Left running past 90°, the pole
would swing down through it like a pendulum, which is why the script
stops there.
