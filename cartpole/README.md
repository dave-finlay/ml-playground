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

Settings can also come from a JSON config file (`--config`), with
command-line options overriding it, the same way as for `train.py`
(see below). Its keys are the option names with underscores
(`d_smoothing` for `--d-smoothing`), and the rule about unused options
applies to the file too. A `weights` path in a config file is relative
to the repo root (for example `cartpole/runs/NAME/weights.json`), so
the config works from any directory; `--weights` on the command line is
relative to the current directory, as usual. Configs worth keeping go
in `configs/cartpole/`:

- `pd-noise-offset.json`: the gentle PD gains with cart feedback and a
  little D smoothing (Kp 20, Kd 3, Kx 0.5, Kv 1, smoothing 0.1), staying
  well controlled with 1° of angle noise and a 1° offset
- `learned-noise-offset.json`: the same conditions with a learned
  controller, the trained model in `models/normalized-seed3/`
- `lqr-precise.json`: the LQR gains for angle cost 0.3, position cost
  0.2 and force cost 0.3, made by `lqr.py` (see below)
- `pd-noisy-sensors.json`, `learned-noisy-sensors.json`,
  `lqr-noisy-sensors.json`: PD, learned and LQR controllers with noisy
  sensors: 1° of angle noise, a 1° offset and 1 cm of position noise, a
  realistic level for a cart's position sensor. The learned one is the
  continuous controller in `models/continuous-precise-seed1` (a PD
  controller with learned gains), not the two-action model.

Trained models worth keeping, for configs to point at, go in
`models/NAME/`, which is tracked by git (`runs/` isn't, so a config
pointing into `runs/` only works where that run exists). A model is a
kept copy of a training run's directory, with the same files:
`weights.json`, the learned weights, and `config.json`, the training
settings that made them (a `train.py` config, not a `cartpole.py`
one), plus `results.json` and `curve.csv` when the run had them. Since
`config.json` includes the seed, it regenerates the weights exactly:
`train.py --config cartpole/models/NAME/config.json`. And since a
model is laid out like a run, `train.py --load cartpole/models/NAME`
tests and shows it. `runs.py promote RUN [NAME]` makes a model from a
run.
So the layout is: settings you write in `configs/` (`train/` for
`train.py`, `cartpole/` for `cartpole.py`), results worth keeping in
`models/`, and every training run in `runs/`.

```sh
uv run python cartpole/cartpole.py --config cartpole/configs/cartpole/pd-noise-offset.json
uv run python cartpole/cartpole.py --config cartpole/configs/cartpole/pd-noise-offset.json --angle-noise 2
uv run python cartpole/cartpole.py --config cartpole/configs/cartpole/learned-noise-offset.json
uv run python cartpole/cartpole.py              # open a window and watch
uv run python cartpole/cartpole.py --slow 4     # quarter speed
uv run python cartpole/cartpole.py --seed 7     # repeatable starting state
uv run python cartpole/cartpole.py --no-window  # just print the trace
uv run python cartpole/cartpole.py --controller p --kp 20
uv run python cartpole/cartpole.py --controller pd --kp 20 --kd 3
uv run python cartpole/cartpole.py --controller p --track-limit 2.4 --max-steps 500  # standard limits
```

The controller doesn't see the true state but readings from a
`Sensor`, which can get the pole angle wrong in three ways and the cart
position in one (all off by default):

- `--angle-noise`: random noise, a fresh normal random number each
  step with this standard deviation in degrees
- `--angle-scale`: a scale error, reading = scale × true angle
- `--angle-offset`: an offset error in degrees, added to every reading
- `--position-noise`: random noise on the cart position, standard
  deviation in metres. It has its own random stream, so a seed gives
  the same angle noise with or without it.

The two velocities pass through perfectly, but controllers estimating
rates from successive readings (the default) never use them. The
printed trace shows the true position and angle and their readings.

Position noise reaches the force mostly through Kv: differencing turns
σ of position noise into about σ·√2 / 0.02 s ≈ 71σ m/s of velocity
noise. With 1° angle noise and a 1° offset as well (10 seeds, RMS angle
and mean force):

| position noise | 0 | 1 cm | 5 cm | 10 cm |
|---|---|---|---|---|
| `pd-noise-offset.json` (Kv 1) | 0.93°, 2.7 N | 0.94°, 2.8 N | 1.21°, 3.6 N | 5.32°, 5.2 N |
| learned continuous (Kv 0.59) | 0.98°, 2.7 N | 0.98°, 2.7 N | 1.07°, 3.2 N | 1.57°, 4.2 N |
| `lqr-precise.json` (Kv 3.5) | 2.68°, 7.4 N | 2.77°, 7.5 N | 4.50°, 8.0 N | fails 10/10 |

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
| `normalize_inputs` | false | divide each input by its typical size (see below) |
| `actions` | 2 | 2: push left or right; 3: also don't push; `continuous`: any force (see below) |
| `force_std` | 5 | continuous: the starting spread of the sampled force, in newtons (learned) |
| `push_cost` | 0 | taken off the reward of each pushing step (needs `actions` 3) |
| `force_cost` | 0 | taken off the reward times (force / 10 N)² (needs `actions` continuous) |
| `angle_cost` | 0 | taken off the reward times (pole angle / 12°)² |
| `position_cost` | 0 | taken off the reward times (cart position / 2.4 m)² |
| `angle_noise`, `angle_scale`, `angle_offset`, `position_noise` | perfect | sensor errors in training, checkpoints and the final test, as in `cartpole.py` |
| `track_limit` | 2.4 | metres from the centre before an episode ends; `position_cost` is relative to it |
| `max_steps` | 500 | longest episode, in steps of 0.02 s |
| `checkpoint_every` | 25 | test without sampling every this many batches, keep the best weights and say whether sampling does better; 0 turns it off |
| `checkpoint_episodes` | 30 | test episodes at each checkpoint |
| `batches` | 200 | number of updates |
| `batch_size` | 10 | episodes per update |
| `learning_rate` | 10 | step size α |
| `gamma` | 0.99 | discount factor γ |
| `eval_episodes` | 20 | test episodes without sampling |

Training configs worth keeping go in `configs/train/`. A config file
only needs the settings it changes:

- `standard.json`: the defaults, written out in full
- `true-velocities.json`: the policy sees the true velocities
- `normalized.json`: normalized inputs, with learning rate 3
- `three-actions.json`: normalized, three actions, push cost 0.3
- `three-actions-long.json`: the same for 1000 batches
- `continuous.json`: continuous force, survival only, learning rate 1
- `continuous-precise.json`: continuous force with angle and position
  costs
- `continuous-noisy.json`: the same plus a force cost, trained with 1°
  of angle noise and 1 cm of position noise
- `continuous-noisy-long.json`: the same on a 5 m track for up to 20 s

A new setting should default to the old behaviour, because configs
(including saved runs' `config.json`) that leave it out get the
default. The one deliberate exception so far is `checkpoint_every`,
which became 25 once checkpoints proved their worth: configs that leave
it out now keep the best checkpoint rather than the final weights.
Kept models whose runs predate it say `"checkpoint_every": 0`, so they
still regenerate exactly.

Each training run is saved in its own directory under `runs/` (ignored
by git), named by `--run-name` or after the config file and the time:

- `config.json`: every setting the run used. It's a config file, so
  `--config runs/NAME/config.json` repeats the run exactly when the
  seed is set.
- `weights.json`: the learned weights, in the order cart position,
  its rate, pole angle, its rate (listed in the file's `inputs`) and in
  those inputs' own units even when training normalized them, with
  their `rate_from`, for `cartpole.py --controller learned --weights`
- with checkpoints (the default): `checkpoints.csv`, one row per
  checkpoint test, and `final-weights.json`, the last weights
  (`weights.json` is then the best checkpoint's)
- `curve.csv`: the learning curve, one row per batch: episodes so far,
  the batch's mean, shortest and longest episode, and the weights
- `results.json`: when it ran, how long training took, the git commit
  (and whether there were uncommitted changes), the test results, and
  the weights scaled to Kp = 20

`--no-save` trains, tests and shows the result as usual but saves
nothing, for quick experiments. `--load runs/NAME` skips training and
tests and shows a saved run with its saved settings.

`runs.py` lists, promotes and deletes runs. `promote` copies a run
into `models/` to keep it (see above), under the run's name or one you
give, never overwriting a model. `delete` takes run names or paths (so
the shell can expand patterns), shows what it will delete and asks
first unless given `--yes`. It only deletes run directories directly
inside `runs/`, so it can't touch `models/` or anything else.

```sh
uv run python cartpole/runs.py list                       # name, start time, test result, description
uv run python cartpole/runs.py promote normalized-b1000            # keep it as models/normalized-b1000
uv run python cartpole/runs.py delete normalized-seed3-again
uv run python cartpole/runs.py delete cartpole/runs/default-2026*
```

### Normalized inputs

Each weight's step is proportional to its input, and the inputs come in
very different sizes: in coin-toss episodes the angle's rate is
typically about 8 times the angle. A term's influence on the decision,
w × input, then grows with the square of its input's size, so with one
learning rate the rate terms run far ahead of the others. With
`normalize_inputs`, training first plays 20 coin-toss episodes,
measures each input's standard deviation, and has the policy see each
input divided by it, so all four weights learn at comparable rates. It
doesn't change which controllers the policy can express: weights w′ on
inputs ÷ scales are weights w′ ÷ scales on the raw inputs, which is how
they're converted back for printing, the learning curve and
`weights.json`. The scales go in `results.json`. They're measured with
the run's own rate estimation, so they'll follow changes such as sensor
noise.

```sh
uv run python cartpole/train.py                                  # defaults
uv run python cartpole/train.py --config cartpole/configs/train/true-velocities.json
uv run python cartpole/train.py --seed 3 --learning-rate 1 --run-name slow-lr
uv run python cartpole/train.py --config cartpole/runs/slow-lr/config.json --run-name slow-lr-again
uv run python cartpole/train.py --load cartpole/runs/slow-lr --slow 4
uv run python cartpole/cartpole.py --controller learned --weights cartpole/runs/slow-lr/weights.json
```

### Three actions: not pushing

Pushing full force left or right every 0.02 s, even when the pole is
nearly balanced, would wear a real motor out. With `actions` 3 the
policy can also not push. Each action k (push left, don't push, push
right) gets a score z_k = w_k · s + b_k, and the policy picks it with
probability softmax(z)_k. The update is REINFORCE as before, with
(onehot(a_t) − softmax(z_t)) in place of (a_t − p_t) and (s_t, 1) in
place of s_t, so 15 weights in all.

The biases b_k matter. Without them, scaling the state up scales every
score alike and never changes which action wins, so the choice can
depend only on which way the state points, not how far from balance it
is, and "don't push near balance" can't be expressed. With them, "don't
push" can win while the push scores are small, so the trained policy
(`ThreeActionController`) acts as a dead band: push full force when
roughly Kx·x + Kv·ẋ + Kp·θ + Kd·θ̇ passes ± a threshold, otherwise coast.
The printout reads the weights that way, assuming push left and push
right mirror each other, and gives the threshold as how far the pole
alone may lean before a push.

The reward also has to make pushing cost something, or not pushing is
never better than pushing, only sometimes no worse: `push_cost` takes
that much off each pushing step's reward. Wear is measured in testing
as the share of steps that push and how often the push changes
direction, for two-action runs too (they push on every step).

Results (normalized, learning rate 3, 200 batches, seeds 1 to 4; hard =
5 m, 20 s, 1° noise and 1° offset, 3 seeds each):

| push cost | test survival | pushing | direction changes | dead band | hard |
|---|---|---|---|---|---|
| 0 | 500 every seed | 100% | 11 to 14 per s | none | 9/12 |
| 0.1 | 500 every seed | 90 to 100% | 12 to 15 per s | none or 0.4° | 6/12 |
| 0.3 | 130 to 500 | 10 to 57% | 0.2 to 10 per s | 0.7° to 5.1° | 7/12 |
| 0.5 | 130 to 500 | 9 to 27% | 0.2 to 5 per s | 4.5° to 6° | 6/12 |

- With no cost it learns faster than two actions (every episode lasts
  500 steps by about batch 40) but never coasts once trained.
- Around 0.3 there's a real trade-off. Seed 3 lasts every test episode
  while pushing on 15% of steps and changing direction 2.5 times a
  second, against two actions' 100% and about 14; seed 1 learned a 5°
  dead band and falls at 130 steps.
- Wide dead bands survive training but fail testing: the sampled
  policy sometimes pushes inside the band, which saves it; the
  deterministic one doesn't.
- Noise defeats much of the dead band: differenced rate noise crosses
  the threshold, so in the hard conditions the same policies push on
  40 to 75% of steps.

### Checkpoints: keeping the best weights

Training optimizes the policy as it samples its actions, but the
controller you keep takes the likeliest action every time, and the two
can drift apart. A 1000-batch three-action run (push cost 0.3, seed 3)
ended with a sampled policy lasting about 498 steps while pushing on
15% of steps, but a deterministic controller lasting only 266: the push
cost had widened the dead band to about 5.8°, and the sampled policy
was relying on occasional random pushes inside it, which average out to
a gentle force that grows the further the pole is off balance.

By default (`checkpoint_every` 25; 0 turns it off), training tests the
deterministic controller every that many batches (and after the last), scores it by the reward
training uses (steps survived, less `push_cost` on pushing steps), and
keeps the best weights so far (ties go to the later checkpoint).
Checkpoint tests always start from the same positions, so checkpoints
compare fairly, but not the final test's positions: choosing the best
checkpoint on the same episodes it's then tested on would flatter it.
Each checkpoint also compares the deterministic result with the sampled
training batches since the last one and says "sampling does better" if
the sampled policy lasted more than a tenth of the episode limit
longer; the run ends with a note if that's still so at the end. It
doesn't change training itself, only which weights are kept.

On the run above, with checkpoints every 25 batches, the kept weights
came from batch 400: 499.6 steps on average in the final test (shortest
493), pushing on 9.3% of steps and changing direction 0.6 times a
second, against 266 steps for the final weights. With only 10
checkpoint episodes it kept batch 900 instead, which looked perfect on
those 10 but averaged 448 in the final test (shortest 151): a few
episodes can't tell a robust controller from a lucky one, hence the
default of 30.

### Continuous force

With `actions` continuous, the policy pushes with a force drawn from a
normal distribution: mean w · s newtons, spread σ (learned, starting at
`force_std`, never below 0.1 N), capped at ±10 N when applied. Once
trained it pushes with the mean, capped: exactly `PDController` with
gains Kx, Kv, Kp, Kd = w, which is what `cartpole.py` runs for
continuous weights. So the learned weights need no scaling to compare
with hand-tuned gains. The update is REINFORCE again: with z = (F − μ)/σ,
z/σ · s for the gains (move the mean toward forces that did better than
average) and z² − 1 for log σ.

Survival alone gave sloppy discrete controllers, so the reward can
include costs: r = 1 − `angle_cost`·(θ/12°)² − `position_cost`·(x/2.4 m)²
− `force_cost`·(F/10 N)² (each fraction capped at 1). The costs must add
up to less than 1 so every step is still worth surviving; otherwise
ending the episode early can pay. Squared costs make small corrections
cheap and large ones expensive, the idea behind the classic LQR
controller. `angle_cost` and `position_cost` work for discrete policies
too; checkpoints score by the full reward.

Both scripts now report control quality: RMS pole angle and cart
position, mean force, share of steps pushing and direction changes per
second (not meaningful for continuous force, which flips sign around 0
with every bit of sensor noise).

Results (normalized, learning rate 1, 200 batches, seeds 1 to 4; tested
on `cartpole.py`'s 5 m track for 20 s, 5 seeds each; noisy means 1°
noise and a 1° offset):

| controller | gains Kx, Kv, Kp, Kd | perfect: RMS angle | noisy: RMS angle, cart, mean force |
|---|---|---|---|
| hand-tuned PD (`pd-noise-offset.json`) | 0.5, 1, 20, 3 | 0.22° | 0.93°, 0.65 m, 2.8 N |
| survival only (3 of 4 seeds) | 0.3 to 1.4, 0.5 to 1.0, 20 to 26, 1.8 to 3.2 | 0.19° to 0.25° | 0.88° to 1.21°, 0.35 to 0.99 m, 2.0 to 3.4 N |
| angle 0.3 + position 0.2 (4 of 4) | 0.6 to 1.3, 0.5 to 1.2, 23 to 28, 2.3 to 3.2 | 0.20° to 0.22° | 0.92° to 1.03°, 0.37 to 0.85 m, 2.5 to 3.4 N |
| + force 0.3 (4 of 4) | 0.4 to 1.1, 0.7 to 0.9, 23 to 27, 2.2 to 3.1 | 0.21° to 0.24° | 0.94° to 1.00°, 0.43 to 0.88 m, 2.4 to 3.3 N |

- With a force it can choose, REINFORCE learns gains much like the
  hand-tuned ones, and controls as well: the gap with PD was the
  actions, not the learning.
- Survival alone failed once (seed 4 learned Kp = 12, barely above the
  10.8 N/rad needed to beat gravity, and no Kv, and fell on the long
  track). The angle and position costs made all seeds reliable and their
  gains more consistent.
- The force cost hardly changed the effort under noise. Training uses a
  perfect sensor, where the forces needed are tiny (about 0.03 N), so
  the cost barely bites; under noise, the effort is mostly the gains
  multiplying noise. Reducing it would mean training with noise.
- Learning rate: 1 is steady, 3 learns faster but drifts to stiffer
  gains (Kp 29 to 50), 10 is unstable.

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

Normalizing the inputs (seeds 1 to 6, 200 batches) changes all of this:

| | unnormalized, rate 10 | normalized, rate 3 |
|---|---|---|
| batches until a 10-batch average of 400 steps | never | 28 to 62 |
| mean episode, last 20 batches | 375 | 492 |
| scaled Kx, Kv, Kd, mean over seeds (gentle PD: 0.5, 1, 3) | −0.8, 5.4, 29 | 2.4, 0.3, 1.7 |
| 5 m, 20 s, perfect sensor (6 runs × 3 seeds) | 9/18 | 18/18 |
| same with 1° noise | 0/18 | 18/18 |
| same with 1° offset | 7/18 | 18/18 |

With the head start gone, training finds a controller much like the
hand-tuned ones: Kd small, and Kx positive (the counter-intuitive sign)
even though position control barely matters in 10 s. The small Kd is
what makes it robust to noise it never trained on. Kv comes out small,
smaller than the hand-tuned 1. Learning rates: 0.3 is slow, 1 works,
3 is best, and 10 is unstable (one seed's policy fell to 127 steps).

### Training under noise

With sensor settings, the policy trains on noisy readings, with rates
differenced from them, exactly as `cartpole.py` controllers see them
(the costs still use the true state). Input scales are measured from
the noisy readings, and checkpoints and the final test use the same
noise, seeded. With a perfect sensor (the default) training is as it
was before.

Continuous force, normalized, learning rate 1, angle cost 0.3 and
position cost 0.2, 4 seeds, trained with 1° of angle noise and 1 cm of
position noise; tested on the 5 m track for 20 s, 10 seeds (RMS angle
and mean force; "noisy sensors" is `*-noisy-sensors.json`: 1° angle
noise, a 1° offset, 1 cm position noise):

| controller | Kx, Kv, Kp, Kd | perfect | noisy sensors | 2° + 2 cm noise |
|---|---|---|---|---|
| hand-tuned PD (smoothing 0.1) | 0.5, 1, 20, 3 | 0.20°, 0.0 N | 0.94°, 2.8 N | 4.2°, 5.1 N |
| learned without noise (`continuous-precise-seed1`) | 0.94, 0.59, 23, 2.5 | 0.20°, 0.0 N | 0.98°, 2.7 N | 3.1°, 5.0 N |
| trained with noise (3 of 4 seeds; the 4th fails everything) | 0.6 to 1.6, −0.3 to −0.1, 26 to 34, 0.5 to 0.7 | 0.5° to 0.9° | 2.4° to 5.5°, 1.1 to 1.3 N | 2.7° to 3.4°, 1.8 to 2.4 N |
| + force cost 0.3 (2 of 4 good) | 0.8 to 4.8, −0.25 to −0.1, 28 to 33, 0.4 to 0.6 | 0.6° to 6.6° | 3.0° to 15°, 1.1 to 2.5 N | 2.9° to 11°, 1.9 to 2.8 N |
| + 1° offset in training | 1.9 to 3.6, −0.24 to 0.04, 27 to 35, 0.4 to 1.0 | 0.8° to 3.1° | 3.0° to 12°, 1.3 to 2.0 N | 3.3° to 10°, 2.1 to 2.9 N |

- Training with noise learned to cut the noisy inputs: Kd fell from
  about 2.5 to about 0.5, and Kv to about 0 (often slightly negative).
  Those are the rates, and differencing makes them by far the noisiest
  inputs. It's the classic errors-in-variables effect: a weight on an
  input that's mostly noise shrinks toward zero.
- So the controllers push gently (about 1 to 2.5 N, under half of PD's
  force under heavy noise) and, in the heaviest noise, hold the pole as
  well as or better than PD (2.7° against 4.2°). But with so little
  damping they're sloppy with good sensors (0.5° or worse against 0.2°)
  and with the moderate noisy-sensors scenario (2.4° or worse against
  0.94°).
- They're unreliable: Kv at or below 0 breaks the cart loop's stability
  condition (Kv > 0), but the cart's slow swing takes longer than a 10 s
  training episode on a 2.4 m track to show, so training can't see it
  and some seeds fail the 20 s test.
- What would help: longer training episodes on the wider track, so
  slow cart instability costs something; smoothing the rates before the
  policy sees them; or estimating them properly with a Kalman filter,
  which would let a controller keep its damping without amplifying the
  noise.

Longer episodes (`track_limit` 5, `max_steps` 1000, as in the tests)
helped only partly. With the same noise and costs, 3 of 4 seeds passed
every test (the 4th never learned to balance in any design: its
training episodes stayed short), with noisy-sensor precision of 2.4° to
3.4° at about 1.1 to 1.4 N, much as before. Kv stayed slightly
negative. On the linear model (`lqr.py`'s), the passing seeds' cart
mode still grows, at 0.05 to 0.07 per second: by about e in 15 to 20 s,
so a 20 s episode sees it grow only about threefold, which costs
little. The failing seeds' cart modes grow at 0.15 to 0.3 per second.
Their pole modes decay at only 0.3 to 0.46 per second against the
hand-tuned PD's 0.96 (all of whose modes decay): under-damped. The root
cause is the noisy rate estimates, which longer episodes can't fix.

## lqr.py: the optimal linear controller

The learned continuous controller has the same form as the classic
LQR (linear-quadratic regulator) solution: force = gains · state. And
while the pole stays up, maximizing train.py's reward (1 per step less
squared angle, position and force costs) is the same as minimizing
discounted quadratic costs, which is LQR's problem. `lqr.py` solves it
directly, for costs given in train.py's terms (or read from a train.py
config), and prints the optimal gains next to the gentle PD gains, the
closed loop's modes (how fast each decays, and whether it oscillates),
and the `cartpole.py` command that runs them; `--save` writes a
`cartpole.py` config.

The model is Gymnasium's equations linearized about upright,
angle_accel = 15.78·θ − 1.463·F and x_accel = −0.717·θ + 0.976·F (the
same numbers as the characteristic polynomial in "Choosing the four
gains"), stepped by explicit Euler like Gymnasium, so it's exact for
small angles. The gains come from iterating the discrete Riccati
equation. A force cost is required: with free force, the optimal gains
are infinite, which is also why training with no force cost keeps
making the gains stiffer.

```sh
uv run python cartpole/lqr.py --angle-cost 0.3 --position-cost 0.2 --force-cost 0.3
uv run python cartpole/lqr.py --force-cost 0.3 --config cartpole/configs/train/continuous-precise.json
uv run python cartpole/cartpole.py --config cartpole/configs/cartpole/lqr-precise.json
```

LQR against the hand-tuned and learned gains (5 m track, 20 s, 10
seeds; survival, RMS angle and cart position, mean force):

| controller | Kx, Kv, Kp, Kd | perfect sensor | 1° noise + 1° offset | 2° noise |
|---|---|---|---|---|
| gentle PD, no smoothing | 0.5, 1, 20, 3 | 10/10, 0.20° | 0.98°, 0.65 m, 3.1 N | 6/10, 10.8° |
| learned (continuous, precise + force, mean of 4) | 0.83, 0.80, 24, 2.45 | 10/10, 0.20° | 0.92°, 0.51 m, 2.7 N | 10/10, 3.0° |
| LQR, force cost 1 | 0.97, 2.0, 40, 8.6 | 10/10, 0.14° | 2.3°, 0.66 m, 6.6 N | 0/10 |
| LQR, force cost 0.3 (`lqr-precise.json`) | 1.8, 3.5, 60, 11.3 | 10/10, 0.12° | 2.7°, 0.56 m, 7.4 N | 0/10 |
| LQR, force cost 0.03 | 5.3, 9.8, 145, 21 | 10/10, 0.10° | 3.5°, 0.50 m, 8.7 N | 0/10 |

- With a perfect sensor LQR is best, as it should be: it's optimal for
  that problem. For the same costs it's far stiffer than what training
  learned (Kp 60 against about 24): training hadn't reached the optimum,
  and a 1000-batch run kept stiffening toward it.
- With noise LQR does badly: it assumes it knows the true state, and its
  large Kd multiplies the noise that differencing amplifies. The learned
  gains, short of the optimum, happen to be robust.
- The principled fix is LQG: a Kalman filter estimates the state
  (including both velocities) from the noisy readings, using the model,
  and LQR acts on the estimate. Alternatively, training with noise
  would let RL find the best gains for the noisy problem directly,
  without modelling the noise.

## kalman.py: estimating the state

Controllers estimate the two rates by differencing successive readings,
which divides each reading's noise by 0.02 s: 1° of angle noise becomes
about 70°/s of angle-rate noise. `kalman.py` is a Kalman filter that
estimates the state from the same two readings, cart position and pole
angle, using the linear model in `model.py` (the one `lqr.py` uses). Its
state is (x, x rate, angle, angle rate, angle sensor offset). Each step
it predicts the state from its last estimate and the force applied, then
corrects toward the readings by a gain that weighs how uncertain the
prediction is against how noisy the readings are. The velocities are
never measured, but the model ties them to how the positions change.
The offset isn't measured either, but it shows: if the pole really
leaned by the offset, the cart would have to keep accelerating to hold
it, and the position readings show whether it does (the observability
matrix has full rank).

The filter isn't told the actual noise. Its noise settings are rough
guesses (angle 2°, position 2 cm; unmodelled accelerations 0.5 m/s² and
0.5 rad/s²; a slight drift for the offset), and its gain depends only on
their ratios, so what's really chosen is how far to trust the readings
against the model.

`cartpole.py --compare-estimates` runs the filter alongside any run,
without it controlling anything, and compares its estimates with the
truth:

```sh
uv run python cartpole/cartpole.py --config cartpole/configs/cartpole/pd-noisy-sensors.json --compare-estimates
```

With the PD controller and noisy sensors (1° angle noise, a 1° offset,
1 cm position noise; 10 seeds; RMS error after the first second):

| | x | x rate | angle | angle rate |
|---|---|---|---|---|
| readings, differenced | 1.00 cm | 72 cm/s | 1.41° | 68°/s |
| Kalman filter | 0.32 cm | 0.9 cm/s | 0.41° | 1.6°/s |
| Kalman filter, no offset state | 0.32 cm | 1.0 cm/s | 1.71° | 6.3°/s |

The offset estimate after 20 s averaged 0.99° (0.94° to 1.05°) against
the true 1°, within about 0.1° after 2 s. Scaling the guessed sensor
noise or the guessed model error from a tenth to ten times the defaults
(a hundredfold range in their ratio) kept the angle-rate error between
1.6 and 3.7°/s and the cart-velocity error between 0.2 and 5.2 cm/s,
still at least 18 times better than differencing; scaling both alike
changed nothing. Trusting the model more did better here because the
simulation is the model near upright; a real cart's model would be
less exact.

## Tests

`tests/` checks the logic that would fail silently rather than crash
(including, for `lqr.py`, the linear model against the hand-worked
characteristic polynomial and the solution against the Riccati
equation):
the config rules (precedence, unknown keys, types, choices), the rate
estimation, the controllers (including that scaling the learned weights
doesn't change the pushes), the REINFORCE maths (returns by hand, the
gradient of log π against a numerical derivative, a policy gradient
worked by hand), that seeded training repeats exactly, and that
training learns at all. Run them from the repo root with
`uv run pytest`; they take about a second.
