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
  then drive the cart back. `--kv` (N·s/m, default 0) adds a term on
  the cart's velocity, which damps the cart's motion. `--rate-from` and
  `--d-smoothing` apply to both rates, the angle's and the cart's.
- `learned` (`LinearController`): weights learned by `train.py` (see
  below), from the weights file given with `--weights`. It pushes
  `--max-force` newtons right when weights · (cart position, its rate,
  pole angle, its rate) ≥ 0 and left otherwise. The rates are estimated
  the way the weights were trained (differences or true velocities), as
  recorded in the weights file.

The rate estimation (differencing and smoothing) lives in
`RateEstimator`, shared by the PD and learned controllers and by
training.

Options a controller doesn't use are rejected rather than silently
ignored: `--kp` needs `--controller p` or `pd`; `--max-force` needs
`p`, `pd` or `learned`; `--kd`, `--kx`, `--kv`, `--rate-from` and
`--d-smoothing` need `--controller pd`; and `--weights` goes with
`--controller learned`.

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
- That's the P-on-angle story again, one level up. In the linearized
  model, adding Kx makes the closed loop's s¹ coefficient exactly zero
  for any Kx, and a characteristic polynomial with a missing term can't
  be stable: a slow cart oscillation slowly grows. Damping it needs a
  term on the cart's velocity, which can be estimated from successive
  position readings, as the D term on the angle is.

Cart velocity feedback (`--kv`, same settings, Kx = 0.5, 60 s):

- Kv = 0.5 to 1 works well. With a perfect sensor the cart settles
  back to the centre (never more than 3 cm away); with 1° noise it
  stays within about ±0.1 m.
- With a 1° offset the cart no longer runs away: it parks at −0.70 m
  and stays there, with the pole truly upright. That's where the Kx
  term cancels the angle term's response to the false reading:
  x = −Kp × offset / Kx = −20 × 0.0175 / 0.5 ≈ −0.70 m.
- The sign is counter-intuitive again: Kv = −1 fails within 15 s.
- Too much Kv fails too: with Kp = 20 and Kd = 3, the linearized model
  is stable only for Kv between about 0.2 and 2.0 at Kx = 0.5 (0.4 to
  1.9 at Kx = 1). Above that, the cart terms undo the damping of the
  pole's swing. Kv = 2 lasts 37 s and Kv = 4 only 3 s. These ranges
  depend on all four gains; see below.

## Choosing the four gains

For small angles, the closed loop with all four gains has the
characteristic polynomial

    s⁴ + (1.46·Kd − 0.98·Kv)·s³ + (1.46·Kp − 15.8 − 0.98·Kx)·s²
       + 14.35·Kv·s + 14.35·Kx

(the numbers come from CartPole's masses, pole length and gravity). It's
stable when all its roots have negative real parts. Some necessary
conditions read straight off it:

- Kx > 0 and Kv > 0: the counter-intuitive signs. With Kv = 0 the s
  term vanishes, which is why position feedback alone can't be stable.
- Kv < 1.5·Kd: too much cart damping cancels the pole's damping.
- Kx < (1.46·Kp − 15.8) / 0.98: Kx eats into the stiffness the angle
  loop has left over after beating gravity.

The Routh–Hurwitz test adds one more condition that ties all four gains
together, and it's the one that usually bites. The rule of thumb behind
it: **the inner (pole) loop must be stiffer than the outer (cart)
loop.** The cart terms work by asking the pole to lean, and the angle
loop has to carry that out faster than the cart loop asks. With
Kp = 14, Kd = 5, Kv = 1, for example, Kp is only just above the 10.8
needed to beat gravity, and Kx = 1 is unstable (a growing swing with a
period of about 4 s); Kx = 0.3 or less works.

Two sets that work well (each kept the pole up for 60 s over five seeds,
with a perfect sensor, with 1° noise and `--d-smoothing 0.5`, and with a
1° offset):

| | Kp | Kd | Kx | Kv | Slowest mode decays at |
|---|---|---|---|---|---|
| gentle | 20 | 3 | 0.5 | 1 | 0.68/s |
| firmer | 25 | 5 | 1 | 2 | 1.05/s |

```sh
uv run python cartpole/cartpole.py --controller pd --kp 20 --kd 3 --kx 0.5 --kv 1
uv run python cartpole/cartpole.py --controller pd --kp 25 --kd 5 --kx 1 --kv 2
```

The firmer set brings the cart back to the centre about 1.5 times
faster, and with an offset it parks closer to the centre (the parking
spot is −Kp × offset / Kx: −0.44 m instead of −0.70 m for 1°). To
explore from either, change one gain at a time and keep Kx small
compared with Kp.

## train.py: a learned controller

`train.py` learns a controller with REINFORCE, the simplest
policy-gradient method, written from scratch in numpy. The policy is
linear: given s = (cart position, its rate, pole angle, its rate), it
pushes right with probability sigmoid(w · s) and left otherwise, always
at the full 10 N. The rates are estimated by differencing successive
positions and angles by default, like the hand-tuned controllers
(`rate_from`).

Each batch plays a few episodes with the current weights, sampling each
push, then makes one update:

1. Work out each step's discounted return, G_t = r_t + γ·G_{t+1} (the
   reward is 1 per step).
2. Standardize the returns over the whole batch to get advantages A_t:
   positive for steps followed by better-than-average returns.
3. Step the weights uphill: w ← w + α · mean over all steps of
   A_t · (a_t − p_t) · s_t, where a_t is 1 for a push right and p_t is
   the probability the policy gave it.

Training starts from zero weights (a coin toss each step) on standard
CartPole (12°, 2.4 m, 500 steps). Afterwards the policy is tested
without sampling, always taking the likelier push: that's
`LinearController`, so the weights play the roles of the PD gains Kx,
Kv, Kp and Kd. Only their ratios matter, since only the sign of w · s
is used, so the printout scales them to Kp = 20 to compare with the
gentle PD gains.

### Settings, configs and runs

Every setting has a built-in default, which a JSON config file
(`--config`) overrides, which a command-line option overrides in turn.
Unknown keys and values of the wrong type are errors. The settings are
declared once, in `TrainConfig`, and `config.py` generates the
command-line options from it (`learning_rate` becomes
`--learning-rate`); `--help` lists them.

| Setting | Default | |
|---|---|---|
| `description` | empty | what the configuration is for |
| `seed` | none | seeds training and the test episodes; none differs each run |
| `rate_from` | `difference` | `difference` or `velocity` (the true velocities) |
| `batches` | 200 | number of updates |
| `batch_size` | 10 | episodes per update |
| `learning_rate` | 10 | step size α |
| `gamma` | 0.99 | discount factor γ |
| `eval_episodes` | 20 | test episodes without sampling |

Configs worth keeping go in `configs/`. A config file only needs the
settings it changes:

- `standard.json`: the defaults, written out in full
- `true-velocities.json`: the policy sees the true velocities

Each training run is saved in its own directory under `runs/` (ignored
by git), named by `--run-name` or after the config file and the time:

- `config.json`: every setting the run used. It's a config file, so
  `--config runs/NAME/config.json` repeats the run exactly when the
  seed is set.
- `weights.json`: the learned weights and their `rate_from`, for
  `cartpole.py --controller learned --weights`
- `curve.csv`: the learning curve, one row per batch: episodes so far,
  the batch's mean, shortest and longest episode, and the weights
- `results.json`: when it ran, how long training took, the git commit
  (and whether there were uncommitted changes), the test results, and
  the weights scaled to Kp = 20

`--load runs/NAME` skips training and tests and shows a saved run with
its saved settings.

```sh
uv run python cartpole/train.py                                  # defaults
uv run python cartpole/train.py --config cartpole/configs/true-velocities.json
uv run python cartpole/train.py --seed 3 --learning-rate 1 --run-name slow-lr
uv run python cartpole/train.py --config cartpole/runs/slow-lr/config.json --run-name slow-lr-again
uv run python cartpole/train.py --load cartpole/runs/slow-lr --slow 4
uv run python cartpole/cartpole.py --controller learned --weights cartpole/runs/slow-lr/weights.json
```

### Results so far

With the defaults, training takes about 2.5 s, the mean episode rises
from about 25 steps to about 400, and the policy without sampling
usually lasts all 500 steps in every test episode. Scaled to Kp = 20,
seeds land near Kx ≈ −1.5 to 0, Kv ≈ 5, Kd ≈ 28 to 32, against the
gentle gains' 0.5, 1 and 3. Standard CartPole is easy enough that many
weightings score perfectly (even bang-bang angle-only PD), so nothing
pins the weights down; the rate weights get a head start because the
rates are bigger numbers than the positions; and in 10 s the cart never
drifts far enough for Kx to matter. With longer training the ratios
keep moving slowly (Kd falls to about 19 by 1000 batches) and the
weights keep growing, so the policy becomes more certain.

On `cartpole.py`'s 5 m track and 20 s, many learned controllers drift
off the track (no Kx), and with 1° of angle noise they all fail within
a few seconds: their large Kd multiplies the noise that differencing
amplifies. Training never saw either condition.

## Tests

`tests/` checks the logic that would fail silently rather than crash:
the config rules (precedence, unknown keys, types, choices), the rate
estimation, the controllers (including that scaling the learned weights
doesn't change the pushes), the REINFORCE maths (returns by hand, the
gradient of log π against a numerical derivative, a policy gradient
worked by hand), that seeded training repeats exactly, and that
training learns at all. Run them from the repo root with
`uv run pytest`; they take about a second.
