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

A controller is a function (or callable object) from state to force.
`run_simulation` runs one episode with a given controller. Pick one with
`--controller`:

- `none` (`no_control`): always 0 N, so the pole just falls.
- `p` (`PController`): proportional control on the pole angle. It pushes
  toward the side the pole is leaning with `kp × angle` newtons (`--kp`,
  in N/rad, default 20), capped at `--max-force` newtons either way
  (default 10).
- `pd` (`PDController`): adds a derivative term, `kd × (rate of change
  of the angle)` (`--kd`, in N·s/rad, default 3), which damps the
  swings. By default the rate is estimated from successive angle
  readings, `(angle now − previous angle) / 0.02 s`, as you would with
  only an angle sensor; `--rate-from velocity` uses the true angular
  velocity instead, like a perfect gyroscope. `--d-smoothing` (0 to
  just under 1, default 0) smooths the rate with an exponential moving
  average: `smoothed = a × previous smoothed + (1 − a) × rate`.
  `--kx` (N/m, default 0) adds a term proportional to the cart's
  position, to keep the cart near the centre. Positive is the useful
  direction, and it's counter-intuitive: with the cart right of centre
  it pushes further right, which tips the pole left so the angle terms
  then drive the cart back.

```sh
uv run python cartpole/cartpole.py              # open a window and watch
uv run python cartpole/cartpole.py --slow 4     # quarter speed
uv run python cartpole/cartpole.py --seed 7     # repeatable starting state
uv run python cartpole/cartpole.py --no-window  # just print the trace
uv run python cartpole/cartpole.py --controller p --kp 20
uv run python cartpole/cartpole.py --controller pd --kp 20 --kd 3
uv run python cartpole/cartpole.py --controller p --track-limit 2.4 --max-steps 500  # standard limits
```

The controller doesn't see the true state but a reading from an
`AngleSensor`, which can get the pole angle wrong in three ways (all off
by default):

- `--angle-noise`: random noise, a fresh normal random number each
  step with this standard deviation in degrees
- `--angle-scale`: a scale error, reading = scale × true angle
- `--angle-offset`: an offset error in degrees, added to every reading

The printed trace shows both the true angle and the reading.

`--track-limit` (default 5 m from the centre) and `--max-steps`
(default 1000 steps, 20 s) change when an episode ends. Both defaults
are more generous than standard CartPole's 2.4 m and 500 steps, to give
controllers room to show how they behave. The window widens with the
track, so the cart and pole stay the same size (1250 pixels wide for
5 m).

With no control, the pole passes 12° after about 0.75 s and is flat after
about 1.3 s. The cart drifts a few centimetres the opposite way, pushed
by the falling pole. The window shows the starting position for a
second before the pole is released and holds the end position for two
seconds.

The equations don't model the track. Left running past 90°, the pole
would swing down through it like a pendulum, which is why the script
stops there.

With P control (seed 1), a gain below about 10.8 N/rad can't stop the
fall, which is where the push starts to outweigh gravity for small
angles. Above it the pole swings back and forth around upright, because
nothing damps the motion, and the cart drifts, because the controller
doesn't look at it. Kp = 11 drifts off the track after 4.2 s, Kp = 20
lasts the full 10 s, and Kp = 50 swings harder and drifts off after
about 6 s.

On a track twice as wide (4.8 m) with up to 3000 steps, Kp = 20 keeps
the pole up for 23 s before the slowly growing swings make it fall.

Sensor errors with P control (Kp = 20, seed 1, 5 m track, 20 s):

- Noise up to 2° still lasts the full 20 s; 5° drifts off the track
  at 8.8 s.
- A scale error acts like a different gain: 0.8 (like Kp = 16) lasts
  20 s, while 1.2 (like Kp = 24) swings harder and falls at 16.5 s.
- An offset is the damaging one: 1° drives the cart off the track at
  5.5 s, and 3° at 3.1 s. The controller holds the pole leaning by the
  offset, and keeping a leaning pole from falling needs the cart to
  accelerate continuously that way (about g × tan(lean)), so it runs
  away.

PD control (Kp = 20, seed 1, 5 m track, 20 s), compared with P:

- With a perfect sensor, P's swings have grown to 30° by the end of the
  20 s. PD with Kd = 3 settles the pole to upright in about 2 s and
  holds it there. For small angles, Kd ≈ 5 is critical damping at
  Kp = 20 (fastest settling without overshoot).
- The cart still drifts slowly, by an amount that depends on how the
  pole settled: with Kd = 3 it creeps about 6 mm/s.
- With 1° of noise, estimating the rate from differences amplifies the
  noise (a 1° jump in 0.02 s looks like 50°/s). Kd = 3 copes, with
  swings of about 1°; Kd = 5 gets pushed around enough to drift off the
  track at 17.7 s.
- PD doesn't help with a 1° offset: the cart still runs off at 5.5 s,
  because the controller sees no error. That needs cart-position
  feedback.

Smoothing the D term (`--d-smoothing`, Kp = 20, seed 1, 1° noise):

- Kd = 5, which drifted off the track at 17.7 s unsmoothed, lasts the
  full 20 s with smoothing 0.5, and the pole's RMS angle in the second
  10 s drops from 1.5° to 0.5°.
- Too much smoothing brings the swings back, because the damping
  arrives late. At 0.95 (time constant about 0.4 s) and above, the
  overshoot after release grows even with a perfect sensor, and at 0.98
  (about 1 s) the noisy RMS angle rises to 3.7° (Kd = 5) or 5.7°
  (Kd = 3). Around 0.5 to 0.8 works well here.
- The RMS angle doesn't fall much below 0.5° however much the D term is
  smoothed, because the P term acts on the noisy angle too.

Cart-position feedback (`--kx`, PD with Kp = 20, Kd = 3, seed 1, up to
60 s):

- The intuitive sign is the wrong one: Kx = −0.5, pushing the cart
  back toward the centre, runs it off the track at 9.7 s.
- Kx = +0.5 does pull the cart back, but it then swings back and forth
  through the centre (period about 10 s) with slowly growing amplitude,
  and goes off the track at 56 s. Larger Kx swings harder and fails
  sooner (Kx = 1 at 28 s, Kx = 2 at 13 s). With a 1° offset, Kx = 0.5
  lasts 21 s.
- That's the P-on-angle story again, one level up: in the linearized
  model, P on position leaves the cart's motion undamped (the closed
  loop's s¹ coefficient is exactly zero). Damping it needs a term on
  the cart's velocity, which can be estimated from successive position
  readings, as the D term on the angle is.
