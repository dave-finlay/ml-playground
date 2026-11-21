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

Below the board, the window shows whose turn it is, with a stone of
their colour (and "White passed." when the last move was a pass, so a
pass doesn't look like nothing happening), the last five moves, passes
included (`5 B B4 · 6 W pass`), and the captures so far, or on your
turn what you can do. With `--record`, a game in the window is also
printed at the console as it's played: the players and komi, each move
(`14. W pass`, `15. B C5, capturing 1`), then the final position and
the result, ready to paste or replay.

`play.py` plays many games between two players and summarizes them, or
shows one game move by move:

```sh
uv run python go/play.py                                    # 1000 random games, 5x5
uv run python go/play.py --size 7 --black eye-aware --white eye-aware
uv run python go/play.py --watch --seed 3                   # one game, move by move
uv run python go/play.py --window --black eye-aware --white eye-aware
uv run python go/play.py --window --black human --white eye-aware   # you play Black
uv run python go/play.py --window --black random --white human      # you play White
uv run python go/play.py --window --record --black human --white eye-aware  # and print the moves
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

### Comparing networks

`play.py` can pit two saved networks against each other:
`--black-model` and `--white-model` give each `learned` player its own
network (`--model` is the default for both). Two networks always taking
their likeliest move would play the same game every time, so `--opening
2` makes each side's first two moves random (eye-aware); `--swap`
alternates colours game by game and reports each player's results as
Black, as White and overall, the fair comparison given Black's
advantage.

```sh
uv run python go/play.py --black learned --white learned --swap --opening 2 --games 400 \
    --black-model go/runs/value.pt --white-model go/runs/selfplay-5x5-seed1.pt
```

A 1000-batch self-play run with the critic (300 batches, then resumed
to 1000; best at batch 980, winning 80% as Black and 74% as White
against its starting network in its own tests) against earlier
networks, 400 games each:

| `value.pt` against... | as Black | as White | overall |
|---|---|---|---|
| the network trained against eye-aware (its start) | 92% | 69% | 80.5% |
| self-play without a critic, 300 batches | 82% | 59.5% | 71% |
| self-play with a critic, 300 batches | 88% | 57.5% | 73% |
| itself (a check: about even) | | | 53% / 47% |

It kept improving all the way to batch 1000, with no collapse (entropy
about 1.05 to 1.17 throughout), mostly by getting better as White
(30% to 74% against the start). Dave can still beat it with either
colour, but loses about half the time. After 300 batches the critic
looked like no help; over 1000 it kept climbing, but the run without a
critic hasn't been taken that far to compare.

### Training longer, and outgrowing the yardstick

Resumed from batch 1000 to 2000 (`--resume go/runs/value-resume.pt
--batches 2000 --save go/runs/value-2000.pt`, about 9 minutes), the
run's own tests against its starting network hovered around 76 to 80%
from batch 1000 on, as if it had stopped improving. Head to head (600
games, colours alternating, 2 random opening moves each) told a
different story:

| | as Black | as White | overall |
|---|---|---|---|
| `value-2000.pt` (best, batch 1820) against `value.pt` (batch 1000) | 81% | 51% | 66% |
| `value-2000-last.pt` (batch 2000) against `value.pt` | 82% | 49% | 66% |
| `value-2000.pt` against the start | 91% | 69% | 80% |
| `value.pt` against the start, for comparison | 92% | 69% | 80.5% |

- The extra 1000 batches made a real difference: the new network beats
  the batch-1000 one two games in three, and wins half its games as
  White against it playing Black.
- The start network had become too weak a yardstick: both networks beat
  it about 80% of the time, so progress above that didn't show. Tests
  need a yardstick a little below the current strength, such as the
  previous strong network (`--benchmark` on a fresh run), or a ladder
  of champions where each clear winner becomes the next yardstick.
- The "best" (picked from about 50 noisy tests, so flattered: the
  winner's curse) and the last network score the same head to head.
- Entropy drifted from about 1.03 to 0.87, still far from collapse; the
  critic improved to predicting the winner 85 to 90% of the time.

Against Dave (4 games), `value-2000.pt` won both its games as Black (by
3 and 16) and was wiped out as White both times (losing all its
stones). On 5 × 5 Black's advantage is enormous (perfect play wins the
whole board for Black), so White is lost against good play anyway, but
it should at least keep a group alive. With no komi, White's training
games are mostly lost whatever it does, a weak signal; komi (or the
more balanced 7 × 7 board, where the fair komi is believed to be about
9) would give White's moves a real win/loss signal.

### Komi

With no komi, Black wins most games between equals (and perfect play
wins the whole 5 × 5 board for Black), so White's training games are
mostly lost whatever it plays: a weak signal for White's moves. Komi
7.5 balances things: eye-aware random players split 49% / 51%, and
`value-2000.pt` playing itself wins 58% as Black instead of about 70%.

With komi the same position is worth 7.5 more points to White, so the
network needs to know who's to move. `PolicyNetwork(colour=True)` takes
a fourth input plane, all 1s when Black is to move and all 0s for White
(as AlphaGo Zero does). `selfplay.py` adds the plane whenever komi isn't
0 (and `learn.py` builds networks with it): an existing network gains it
with its first-layer weights for the plane at zero, so it plays exactly
as before until training puts the plane to use. Saved networks record
their komi, and `play.py` uses it by default (`--komi` overrides it;
networks trained with different komi need an explicit choice).

```sh
uv run python go/selfplay.py --init go/runs/value-2000-last.pt --benchmark go/runs/value-2000.pt \
    --value --komi 7.5 --batches 500 --seed 1 --save go/runs/komi.pt
uv run python go/play.py --window --black human --white learned --model go/runs/komi.pt
```

Training 500 batches like that (about 4 minutes), its tests against
`value-2000.pt` under komi 7.5 rose from 46% / 36% (as Black / as White)
to 66% / 54%, still improving at the end; Black's share of training
games fell from about 70% to 62%. Head to head under komi 7.5 (600
games):

| | as Black | as White | overall |
|---|---|---|---|
| `komi.pt` against `value-2000.pt` | 64% | 62% | 63% |
| `komi.pt` against itself | Black wins 52%, White 48% | | |

It plays both colours about equally well now (before, about 80% as
Black and 50% as White against the previous network), and the game is
balanced. Its entropy fell faster than in earlier runs (0.96 to 0.74),
worth watching in longer ones.

Against Dave, `komi.pt` as White lost by only 1.5 points (earlier
networks lost all their stones as White), and as Black it took the
whole board three times out of three, komi or not.

Resumed to 1500 batches, though, it stopped improving: its best test
stayed at batch 500 (so `komi-1500.pt` is the same network as
`komi.pt`), its tests wandered between 40% and 62% against
`value-2000.pt`, and head to head (600 games, komi 7.5) the batch-1500
network split 50% with `komi.pt` but won only 48% against `value-2000.pt`
(where `komi.pt` won 63%). It drifted to a different style rather than
a better one: Black's share of training games swung from 81% down to
30% as the colours' play kept overtaking each other, entropy fell to
about 0.52 to 0.70, and the critic's predictions worsened. Likely
causes: a learning rate too high for fine improvement (noisy REINFORCE
steps jostle a good policy around), an entropy bonus too weak for a
closer game, and a pool of 150 mostly old, easy snapshots. Nothing stops
the drift, which is the case for gating: keep a champion and only
replace it when a challenger clearly beats it.

### A ladder of champions

`ladder.py` stops self-play drifting by keeping its opponents fixed
until the learner has clearly beaten them. The champion starts as the
`--init` network and the learner as a copy. Each batch, the learner
plays the current champion (half the games) or a randomly chosen
earlier champion, with only its own moves trained on (per-colour
baseline, or a critic with `--value`). Every 50 batches it plays the
champion at the gate, 400 games with both taking their likeliest moves
and random two-move openings; at 55% or more it's promoted, becoming
the new champion and the next rung, `champion-N.pt` in `--dir`, with
`ladder.csv` logging each promotion. Its defaults are steadier than
`selfplay.py`'s: learning rate 0.0003 and entropy bonus 0.05.

```sh
uv run python go/ladder.py --init go/runs/komi.pt --komi 7.5 --value \
    --batches 1000 --seed 1 --dir go/runs/ladder-komi
```

From `komi.pt` (1000 batches, about 7 minutes) it made no promotions:
the gate results stayed between 45% and 54.5% (53% to 54.5% in the last
three gates). Meanwhile the learner's share of training games rose from
56% to 88%. The champions sample their moves in training games (for
variety), which makes them much weaker than at the gate, where they
take their likeliest moves: the learner learned to exploit a noisy
champion's mistakes, which barely helped against the real one. The
ladder did its job, refusing to promote a learner that wasn't better.
The fix to try is to have training opponents play as at the gate
(likeliest moves, random openings for variety), so training optimizes
what the gate measures.

Dave on `komi.pt`: he can beat it as Black, but any simple mistake
loses, and as White it's almost impossible to win. That fits 5 × 5:
perfect Black play wins the whole board, by 17.5 even after komi 7.5, so
against strong enough Black play White is lost from the start. 7 × 7,
believed balanced at a komi of about 9, would be the fairer test.

### Carrying a run on

With `--save go/runs/NAME.pt`, a resume file `go/runs/NAME-resume.pt` is
written beside the networks at every test. `--resume` carries the run
on from it, up to `--batches` in total, exactly as if it had never
stopped: the network, the optimizer's running averages, the pool of past
copies, the best version so far and the random number generators all
carry over, so 300 batches then a resume to 1000 ends with the same
network as 1000 batches in one go. It keeps the run's settings (only
`--batches` and `--save` may be given with it), and without `--save` it
keeps writing to the same files.

```sh
uv run python go/selfplay.py --init go/runs/eye-aware-5x5-seed1.pt --batches 300 --seed 1 --value --save go/runs/value.pt
uv run python go/selfplay.py --resume go/runs/value-resume.pt --batches 1000
```

To start a new run from a saved network instead (fresh optimizer and
pool, new settings allowed), use `--init` with the `-last` file, and
`--benchmark` to keep testing against the same yardstick.

### A critic (value head)

With `--value`, the network also has a value head: from the same
convolutions, each feature averaged over the board, then a linear layer
and tanh give V, the expected result for the player to move (−1 to +1).
Averaging suits it: area scoring is a sum over points of whose they
are. Each move's advantage becomes its result z less the critic's
estimate of the position before it, z − V(board), instead of less its
colour's average (V should learn Black's advantage itself), and the
critic learns by regression toward the results of the same recorded
moves, (V − z)², weighted by `--value-weight` (default 0.5).
`--value-detached` trains the critic without its loss reaching the
shared convolutions. A network without a value head (like one trained
by `learn.py`) gets a fresh one, choosing moves exactly as before.

Same start, seed and settings as above (300 batches); best win rate
against the starting network, averaged over both colours:

| | best | critic predicts the winner | value loss |
|---|---|---|---|
| no critic (per-colour baseline) | 64% (batch 250) | | |
| shared critic, weight 0.5 | 52% (batch 300) | 77% → 86% | 1.68 → 0.44 |
| shared critic, weight 1.0 | 45% (batch 300) | 76% → 84% | 1.68 → 0.43 |
| detached critic, weight 0.5 | 61% (batch 250) | 68% → 77% | 0.86 → 0.68 |

- The critic learns to judge positions (predicting the winner about 85%
  of the time), but sharing the convolutions slowed the policy's
  improvement, and more so with the heavier weight: the value loss bends
  the shared features toward judging positions at the expense of
  choosing moves. Detaching it recovered most of the gap, at the cost
  of a less accurate critic.
- No critic beat the simple per-colour baseline within 300 batches. Its
  advantages are also small when it's confidently right, so many moves
  get little push. Next to try: the critic learning from the past
  copies' moves too, and an advantage from the change in V across each
  move (−V after the move − V before it).
- Single runs with noisy tests (about ±7%); small differences need
  several seeds to judge.

## mcts.py: looking ahead with tree search

The networks choose moves by pattern, one look at the board, with no
lookahead. `mcts.py` is a Monte Carlo tree search player, with no
learning at all. For each move it runs many simulations, each of which:

1. selects a path down the tree of continuations, at each step taking
   the move with the best UCT score, wins / visits + 1.4 ×
   √(ln parent visits / visits), balancing moves that have done well
   against ones tried less often;
2. expands one new position at the edge of the tree;
3. evaluates it by playing out to the end with random moves that never
   fill their own eyes (trying empty points in random order rather than
   listing every legal move, about 0.4 ms a playout), scored by the
   rules with komi;
4. backs the result up along the path, as a win for the side that made
   each move or a loss.

It then plays its most-visited move. `play.py` has it as the `mcts`
player, with `--simulations` per move (default 400; about 0.2 s a move,
1000 about 0.45 s). `Game.copy()` gives each simulation its own copy of
the position.

```sh
uv run python go/play.py --window --black human --white mcts --simulations 1000 --komi 7.5
uv run python go/play.py --black mcts --white learned --model go/runs/komi.pt --simulations 1000 --swap --games 60
```

Against `komi.pt` under komi 7.5 (60 games each, colours alternating):

| `mcts` | as Black | as White | overall |
|---|---|---|---|
| 100 simulations | 10% | 17% | 13% |
| 400 simulations | 33% | 27% | 30% |
| 1000 simulations | 77% | 43% | 60% |

So `komi.pt`, one pass through the network per move, plays about as
well as 700 to 900 simulations of plain search, about a third of a
second of thinking a move, and search keeps getting stronger with more
simulations while the network plays at one fixed strength. With no
learning, just the rules, random playouts and counting, 1000
simulations beat the best network. (60 games give about ±6%.)

## guided.py: search guided by a network

As in AlphaZero, the network does two jobs inside the search. Its
policy gives each move a prior P, and selection uses PUCT instead of
UCT, Q + 1.5 × P × √(parent visits) / (1 + visits), so simulations go to
moves the network thinks are worth a look (Q is the move's average value
so far). Its value head judges each new position directly instead of a
random playout (a network without one falls back to playouts). When the
search reaches a new position, the network evaluates it once: every
legal move becomes a child with its prior, and the value is backed up
the path with its sign flipping at each level. It plays the move it
visited most. `play.py` has it as the `guided` player, with `--model`
and `--simulations`; `--black-simulations` and `--white-simulations`
give two searchers different budgets.

```sh
uv run python go/play.py --window --black human --white guided --model go/runs/komi.pt --simulations 100
uv run python go/play.py --black guided --white learned --model go/runs/komi.pt --simulations 100 --swap --opening 2 --games 200
```

With `komi.pt` (komi 7.5, which `play.py` reads from the network):

| guided search against... | as Black | as White | overall |
|---|---|---|---|
| `komi.pt` alone, 25 simulations (200 games) | 76% | 57% | 66.5% |
| `komi.pt` alone, 100 simulations | 87% | 81% | 84% |
| `komi.pt` alone, 400 simulations | 92% | 83% | 87.5% |
| plain `mcts`, both 100 simulations (60 games) | 100% | 100% | 100% |
| plain `mcts`, both 400 simulations | 93% | 93% | 93% |
| plain `mcts` with 1000, guided with 25 | 67% | 40% | 53% |
| plain `mcts` with 1000, guided with 100 | 73% | 50% | 62% |

- A little lookahead goes a long way: 25 simulations (about 0.01 s a
  move) make the network beat its unsearched self two games in three,
  and 100 make it win 84%, as White too.
- With the same budget, guidance makes search far stronger than random
  exploration and playouts (100% and 93%); guided search with 25
  simulations matches plain search with 1000, 40 times fewer.
- Against Dave as Black (on 5 × 5, the winning side), guided search as
  White with 100 simulations won 2 of 4 games (by 4.5 and 3.5, keeping a
  living group), where `komi.pt` alone lost as White even at its best.
  Its losses were total (every stone captured), at 400 simulations too:
  more search explores the value head's misjudgements of life and death
  more thoroughly rather than correcting them.
- Next: AlphaZero's training loop, where self-play games are played by
  this search and the network learns to imitate its choices (the visit
  counts) and predict the results, so search and network improve each
  other.

## symmetry.py: training on every orientation of the board

A Go position is worth the same turned or flipped, but the networks
don't know it: convolutions treat a pattern alike wherever it is, not
its mirror image. After C3 C2, `komi.pt` gives B2 an 80% prior and the
identical D2 under 2%. `symmetry_gap` measures this: the average total
variation distance between a network's move probabilities for a
position and for its 7 turned versions (turned back), on positions from
random games (0 is symmetric). A fresh network is 0.004; the first
REINFORCE network was already 0.23, and `komi.pt` is 0.29.

`selfplay.py --augment all` trains on each recorded move in all 8
orientations (planes, legal mask and move turned together, the advantage
unchanged); `--augment random` uses one at random. Every test now prints
the gap.

**Unweighted, it collapses.** REINFORCE's update is only right on
average for moves turning up as often as the network would play them; a
turned move turns up as often as the network played the original, so
unless the network is symmetric, most of the training examples are
off-policy. From `komi.pt`, entropy fell from 0.76 to 0.06 within 140
batches and the network ended up losing even to eye-aware random play.
From a fresh, symmetric network it collapsed too (entropy 2.2 to 0.3 by
batch 300, the gap lurching to 0.59): a slight tilt is enough.
Augmentation itself works: imitating a fixed set of moves, it cut the
gap from 0.29 to 0.15, where plain imitation raised it to 0.40.

**Weighted, it's valid but barely symmetric.** `--importance-cap`
(default 1) weights each turned move's policy update by rho =
pi(turned move | turned board) / pi(move | board), capped at 1
(importance sampling, as in PPO); the critic learns from every
orientation unweighted (plain regression, valid on any board). The
batch lines show the mean rho. But weighting makes the turned moves count
as often as the network would play them, which is ordinary on-policy
REINFORCE, and that has no pull towards symmetry: as the network tilts,
rho falls (0.99 to 0.75) and the lesson fades. From scratch with komi
7.5 and a critic, against the same run without augmentation (same seed,
3000 batches):

| | policy gap | value gap | training time |
|---|---|---|---|
| no augmentation (`fresh-none`) | 0.34 | 0.10 | 37 minutes |
| augmented, weighted (`fresh-all-last`) | 0.30 | 0.07 | 106 minutes |

Head to head (400 games, random two-move openings):

| | overall |
|---|---|
| `fresh-none` against `komi.pt` | 54% |
| `fresh-all-last` against `komi.pt` | 58.5% |
| `fresh-all-last` against `fresh-none` | 59.5% |
| `fresh-all-4000-last` (1000 more batches) against `fresh-all-last` | 57% |
| `fresh-all-4000-last` against `komi.pt` | 61% |
| `fresh-all-4000-last` against `fresh-none` | 66.5% |

So the weighted runs were the strongest yet, probably from a steadier
gradient and a better critic rather than symmetry, and they kept
improving where earlier continued runs drifted (the tests, 100 games
each, looked flat throughout: head-to-heads between versions are the
measure to trust). But the gain is modest for three times the training,
and it doesn't touch the bot's real weaknesses. Against Dave with
`fresh-all-4000-last`:

- As Black he beat guided search (100 simulations) with near-perfect
  play; White was lost from move 2, and the network's own value still
  gave it about 45% to 51% with its big group about to die.
- As White he won against 100 simulations: at move 9 the policy gave the
  vital point E2 a prior of 0.03 (D5, played, 0.77), and 100
  simulations never looked at it (400 do). A tactical miss.
- As White he won against 1000 simulations: the network-guided search,
  even at 10,000 simulations, kept Black at 78% to 86% while White made
  a living group of 10 points that, with komi, beats Black's 15. Plain
  search with random playouts (which count every game to the end) saw
  it. A misjudgement of the score, which more search can't fix: it only
  explores the value head's view more thoroughly.

Verdict: a modest gain, and the wrong tool for what costs games. The
lessons: REINFORCE can't absorb augmentation (it needs targets that are
facts about the position), and the bot's weaknesses are reading and
judgement, which AlphaZero's training targets.

## Tests

`uv run pytest` from the repo root runs them, with the other
experiments': captures, suicide, ko and superko, passing and the move
limit, scoring and komi, eyes, random games finishing legally, the
window (with synthetic clicks), and the network: encoding from the
mover's point of view, legal masks, only legal moves played, and a
REINFORCE update making moves from won games more likely.

Module names here are distinct from `cartpole/`'s (both directories are
on the tests' Python path, so a second `train.py` would collide).
