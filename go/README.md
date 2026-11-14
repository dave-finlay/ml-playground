# go

Learning to play Go on a small board (5 × 5 or 7 × 7), step by step:
random players first, then a learned player against random (done: a
small network trained with REINFORCE beats the random players about 99%
of the time), then learned against learned. The rules are written from
scratch.

## game.py: the rules

Close to the Tromp-Taylor rules, the simplest complete rules of Go:

- Black and White alternate placing a stone on an empty point, or
  passing.
- An opponent group (stones joined horizontally or vertically) left
  with no liberties (adjacent empty points) after a move is captured.
- Suicide is illegal: a move may not leave the mover's own group with
  no liberties after captures. (Tromp-Taylor allows it; Chinese and
  Japanese rules don't, and forbidding it removes pointless moves.)
- Positional superko: no move may recreate an earlier board position.
  That includes the basic ko rule and means a game can't cycle.
- The game ends after two passes in a row, or at a move limit (default
  4 moves per point).
- Area scoring: stones on the board plus empty regions bordered only by
  your stones. White adds komi.

Points are numbered row by row from the top left and named like C3
(columns A, B, ... skipping I, rows numbered from the bottom).
`Game.from_rows` sets up a position from rows of text (`X`, `O`, `.`),
which the tests use.

## players.py and play.py

Three players so far:

- `random`: uniform over all legal moves, passing included.
- `eye-aware`: random, but never fills one of its own eyes, and passes
  only when nothing else is left. An eye is an empty point surrounded
  by its own stones, where the opponent holds at most one diagonal (none
  on the edge); otherwise it's a false eye. This is the standard random
  player of Go programs, and the playout policy Monte Carlo tree search
  uses.
- `human`: you, in the window (needs `--window`): click a point to play
  there, or press P to pass. An illegal click says why (occupied,
  suicide, or repeating a position) and waits for another. The board
  scores as it stands, with no removal of dead stones, so capture them
  before passing.

`play.py` plays many games between two players and summarizes them, or
shows one game move by move:

```sh
uv run python go/play.py                                    # 1000 random games, 5x5
uv run python go/play.py --size 7 --black eye-aware --white eye-aware
uv run python go/play.py --watch --seed 3                   # one game, move by move
uv run python go/play.py --window --black eye-aware --white eye-aware
uv run python go/play.py --window --black human --white eye-aware   # you play Black
uv run python go/play.py --window --black random --white human      # you play White
```

`--window` plays one game in a pygame window (`view.py`): the board with
its coordinates, a red dot on the latest stone, and the move and
captures so far below. `--delay` sets the seconds after each computer
move (default 0.5); space pauses and resumes, and Escape or closing the
window stops the game. At the end each side's territory is marked with
small squares of its colour, and the score stays up until a key is
pressed.

## Random against random

1000 games each, komi 0:

| players | board | Black wins | White wins | draws | length | ended by |
|---|---|---|---|---|---|---|
| random vs random | 5 × 5 | 50.6% | 43.8% | 5.6% | 39 moves | two passes, 99% |
| random vs random | 7 × 7 | 53.4% | 42.1% | 4.5% | 73 moves | two passes, 99% |
| eye-aware vs eye-aware | 5 × 5 | 53.5% | 46.5% | 0% | 43 moves | two passes, 100% |
| eye-aware vs eye-aware | 7 × 7 | 53.8% | 46.2% | 0% | 73 moves | two passes, 100% |
| eye-aware (Black) vs random | 5 × 5 | 87.8% | 12.0% | 0.2% | | |

- Moving first helps even at random: Black wins a few points more
  often.
- Plain random games end early by accident: pass is always one of the
  choices, and as legal moves run out it becomes likely, so games stop
  with much of the board unresolved and the margins spread widely.
  Random play that passes only when forced is no better: players keep
  filling their own eyes and getting captured, 98% of games hit the
  move limit, and the result mostly reflects who moved last.
- Eye-aware games end naturally and decisively. On 5 × 5, 88% end with
  one side owning the whole board (whoever makes a group with two eyes
  captures everything else); on 7 × 7, 42%.
- Not filling your own eyes is already a big skill: eye-aware beats
  plain random 88% of the time.

## learn.py: a learned player

`policy.py` holds a small convolutional network (PyTorch). It sees the
board from the point of view of the player to move, as three layers
(that player's stones, the opponent's, the empty points), so one
network plays either colour. Three 3 × 3 convolutions of 32 channels
each look at every point and its neighbours (stacked, they see further);
a 1 × 1 convolution then scores each point, and the last layer's
average scores passing. Illegal moves are masked out before the softmax.

`learn.py` trains it with REINFORCE against a fixed opponent (by default
the eye-aware player), alternating colours. Each batch of 64 games, the
network samples its moves from its own probabilities; the reward is the
result (+1 win, −1 loss, 0 draw), and every one of its moves in a game
gets the game's result less the batch's average result as its
advantage. One Adam step (learning rate 0.001) then makes moves from
better-than-average games more likely. Every 20 batches it's tested
without sampling (its likeliest move), on 100 games as each colour, and
the best version is kept and saved with `--save`. `play.py` plays a
saved network as the `learned` player with `--model`:

```sh
uv run python go/learn.py --batches 200 --seed 1 --save go/runs/eye-aware-5x5-seed1.pt
uv run python go/play.py --black learned --white eye-aware --model go/runs/eye-aware-5x5-seed1.pt
uv run python go/play.py --window --black human --white learned --model go/runs/eye-aware-5x5-seed1.pt
```

Results on 5 × 5 (seed 1, komi 0):

- Untrained (always its likeliest move) it loses almost every game. After
  40 batches (2560 games, about 17 s) it wins about 90% against
  eye-aware as either colour; the best version (batch 180 of 200, about
  70 s) won 100% as Black and 98% as White in its test.
- On 500 fresh games: against eye-aware it wins 99.0% as Black and
  98.4% as White; against uniform random 100% and 99.6%.
- It opens in the centre (C3) as Black, the known best first move on
  5 × 5, and takes the centre as White when it's free.
- It hasn't learned to tidy up: it often passes with dead opponent
  stones still on the board, which count for the opponent under area
  scoring. Against random players that costs it nothing.
- Against itself (always its likeliest move) it plays the same game
  every time; Black wins by 12.

## selfplay.py: learned against learned

Against random players the network learns only what beats random
players. In self-play its opponent improves with it. Each batch of 64
games, the network plays either itself (both sides sampling, and both
sides' moves trained on, each with its own side's result) or, half the
time, a past copy of itself from a pool (a snapshot is added every 10
batches; only the current network's moves are trained on), alternating
colours. Each move's advantage is its side's result less the average
result of its colour in the batch, a separate baseline for Black and
White: between equals Black wins most games, which would otherwise make
every Black move look good. The loss is REINFORCE's, less a bonus for
the policy's entropy (how spread out its move probabilities are).

Self-play win rates mean nothing (it always plays its equal), so every
50 batches (`--eval-every`) it's tested, always taking its likeliest
move, against fixed yardsticks: the eye-aware player and the network it
started from (`--init`, or another with `--benchmark`). Two
deterministic networks would play the same game every time, so those
test games open with 2 random moves per side. The best version tested
is saved with `--save`, and the last beside it with `-last` added.

```sh
uv run python go/selfplay.py --init go/runs/eye-aware-5x5-seed1.pt --batches 300 --seed 1 --save go/runs/selfplay-5x5-seed1.pt
uv run python go/play.py --window --black human --white learned --model go/runs/selfplay-5x5-seed1.pt
```

Starting from the network trained against eye-aware (seed 1, 300
batches, about 130 s); win rates against that starting network, as
Black / as White (the starting network against itself scores 52% /
30%: Black's advantage is large between equals):

| batch | no entropy bonus | bonus 0.01 | bonus 0.03 (the default) |
|---|---|---|---|
| 100 | 70% / 30% | 68% / 28% | 64% / 40% |
| 200 | 72% / 52% | 68% / 48% | 76% / 42% |
| 250 | 66% / 38% | 76% / 38% | 74% / 54% |
| 300 | 40% / 28% | 62% / 26% | 74% / 52% |
| entropy at 300 | 0.27 | 0.49 | 1.02 |

- Without the bonus, self-play improved for about 200 batches and then
  collapsed below where it started: its entropy fell from 0.79 to 0.27
  as it narrowed into one style that beat its recent selves but lost to
  its starting self. A bonus of 0.01 only delayed that.
- With 0.03 the entropy held at about 1, and it kept improving: the
  best version (batch 250) wins 64% on average against its starting
  self, against 41% for the start against itself, and 96 to 100%
  against eye-aware throughout.
- These are single runs with noisy tests (50 games per colour, roughly
  ±7%).

## Tests

`uv run pytest` from the repo root runs them, with the other
experiments': captures, suicide, ko and superko, passing and the move
limit, scoring and komi, eyes, random games finishing legally, the
window (with synthetic clicks), and the network: encoding from the
mover's point of view, legal masks, only legal moves played, and a
REINFORCE update making moves from won games more likely.

Module names here are distinct from `cartpole/`'s (both directories are
on the tests' Python path, so a second `train.py` would collide).
