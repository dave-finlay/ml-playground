# go

Learning to play Go on a small board (5 × 5 or 7 × 7), step by step:
random players first, then a learned player against random, then
learned against learned. The rules are written from scratch.

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

Two players so far:

- `random`: uniform over all legal moves, passing included.
- `eye-aware`: random, but never fills one of its own eyes, and passes
  only when nothing else is left. An eye is an empty point surrounded
  by its own stones, where the opponent holds at most one diagonal (none
  on the edge); otherwise it's a false eye. This is the standard random
  player of Go programs, and the playout policy Monte Carlo tree search
  uses.

`play.py` plays many games between two players and summarizes them, or
shows one game move by move:

```sh
uv run python go/play.py                                    # 1000 random games, 5x5
uv run python go/play.py --size 7 --black eye-aware --white eye-aware
uv run python go/play.py --watch --seed 3                   # one game, move by move
```

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

## Tests

`uv run pytest` from the repo root runs them, with the other
experiments': captures, suicide, ko and superko, passing and the move
limit, scoring and komi, eyes, and random games finishing legally.

Module names here are distinct from `cartpole/`'s (both directories are
on the tests' Python path, so a second `train.py` would collide).
