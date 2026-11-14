"""Train a Go network up a ladder of champions.

Plain self-play can drift: its opponent changes every batch, and nothing
stops it wandering to a different style that's no better. Here the
opponents stay fixed until the learner has clearly beaten them:

- The champion starts as the --init network, and the learner as a copy.
- Each batch the learner plays games against the current champion
  (half of them by default) or a randomly chosen earlier champion, so it
  can't just learn to exploit one opponent. The champions sample their
  moves too, so games vary. Only the learner's moves are trained on
  (REINFORCE, as in selfplay.py: a per-colour baseline, or a critic with
  --value; with an entropy bonus).
- Every gate_every batches, the learner plays the champion (both taking
  their likeliest moves, with random openings so the games differ):
  if it wins at least promote_at of the games, it's promoted. It becomes
  the new champion (the old one joins the earlier champions) and is
  saved as the next rung, champion-N.pt in --dir.

Each gate also reports the learner against the first champion, which
shows progress overall, and ladder.csv in --dir records the promotions.

Run from the repo root:

    uv run python go/ladder.py --init go/runs/komi.pt --komi 7.5 --value \\
        --batches 1000 --seed 1 --dir go/runs/ladder-komi
    uv run python go/play.py --window --black human --white learned \\
        --model go/runs/ladder-komi/champion-3.pt
"""

import argparse
import copy
import csv
import os
import random
import time

import numpy as np
import torch

from game import BLACK, WHITE
from policy import (
    NetworkPlayer,
    actor_critic_loss,
    load_network,
    policy_gradient_loss,
    save_network,
    upgraded,
    with_value_head,
)
from selfplay import (
    advantages_by_colour,
    moves_and_results,
    play_training_game,
    win_rates,
)


def frozen(network):
    """A copy of network to play as a fixed opponent."""
    return copy.deepcopy(network).eval()


def head_to_head(network, opponent, games, size, komi, seed, opening):
    """network's win rate against opponent, both taking their likeliest
    moves: (as Black, as White, overall), over games / 2 games per colour,
    each opening with `opening` random moves per side."""
    black, white = win_rates(
        network,
        lambda s: NetworkPlayer(opponent),
        games,
        size,
        komi,
        seed,
        opening=opening,
    )
    return black, white, (black + white) / 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--init", required=True, help="the first champion's network")
    parser.add_argument(
        "--dir", required=True, help="where to save the champions and ladder.csv"
    )
    parser.add_argument("--size", type=int, default=5, help="board size (default: 5)")
    parser.add_argument("--komi", type=float, default=0.0, help="komi (default: 0)")
    parser.add_argument(
        "--batches", type=int, default=500, help="updates (default: 500)"
    )
    parser.add_argument(
        "--games", type=int, default=64, help="games per update (default: 64)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Adam step size; smaller than selfplay.py's, for steadier steps "
        "(default: 3e-4)",
    )
    parser.add_argument(
        "--entropy-bonus",
        type=float,
        default=0.05,
        help="reward for keeping move probabilities spread out (default: 0.05)",
    )
    parser.add_argument(
        "--champion-fraction",
        type=float,
        default=0.5,
        help="share of training games against the current champion rather than "
        "an earlier one (default: 0.5)",
    )
    parser.add_argument(
        "--gate-every",
        type=int,
        default=50,
        help="batches between attempts at promotion (default: 50)",
    )
    parser.add_argument(
        "--gate-games",
        type=int,
        default=400,
        help="games against the champion at each gate (default: 400)",
    )
    parser.add_argument(
        "--promote-at",
        type=float,
        default=0.55,
        help="win rate against the champion needed for promotion (default: 0.55)",
    )
    parser.add_argument(
        "--opening",
        type=int,
        default=2,
        help="random opening moves per side in gate games (default: 2)",
    )
    parser.add_argument(
        "--value", action="store_true", help="use a critic (see selfplay.py)"
    )
    parser.add_argument(
        "--value-weight",
        type=float,
        default=0.5,
        help="with --value, the weight of the critic's loss (default: 0.5)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else int(time.time())
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pick = random.Random(seed)

    learner = load_network(args.init)
    learner.train()
    if args.value and not learner.has_value:
        learner = with_value_head(learner)
    if args.komi and not learner.has_colour:
        learner = upgraded(learner, colour=True)
        print(
            f"Adding a colour plane: with komi {args.komi:g}, a position's "
            "value depends on who's to move"
        )
    optimizer = torch.optim.Adam(learner.parameters(), lr=args.learning_rate)
    first = frozen(learner)
    champion, earlier = frozen(learner), []
    os.makedirs(args.dir, exist_ok=True)
    save_network(
        os.path.join(args.dir, "champion-0.pt"),
        champion,
        {"komi": args.komi, "batch": 0, "rung": 0},
    )
    log_path = os.path.join(args.dir, "ladder.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow(
            [
                "rung",
                "batch",
                "win_rate_vs_previous",
                "as_black",
                "as_white",
                "win_rate_vs_first",
            ]
        )
    gate_seed = seed + 1_000_003
    rung = 0

    print(
        f"Champion 0: {args.init}. Gate every {args.gate_every} batches: "
        f"{args.gate_games} games, promotion at {100 * args.promote_at:.0f}%."
    )
    print(
        f"{'batch':>5}  {'games':>6}  {'won':>5}  {'entropy':>7}  champions"
        + ("  value loss, sign right" if learner.has_value else "")
    )
    start = time.perf_counter()
    for batch in range(1, args.batches + 1):
        sides, wins = [], []
        for i in range(args.games):
            colour = BLACK if (batch + i) % 2 else WHITE
            use_champion = not earlier or pick.random() < args.champion_fraction
            opponent = champion if use_champion else pick.choice(earlier)
            game_sides = play_training_game(
                learner, opponent, colour, args.size, args.komi, rng
            )
            sides += game_sides
            wins.append(game_sides[0][2] > 0)
        critic = ""
        if learner.has_value:
            records, results = moves_and_results(sides)
            loss, entropy, value_loss, right = actor_critic_loss(
                learner,
                records,
                results,
                args.value_weight,
                args.entropy_bonus,
            )
            critic = f"  {value_loss:10.3f}  {100 * right:9.0f}%"
        else:
            records, advantages = advantages_by_colour(sides)
            loss, entropy = policy_gradient_loss(
                learner, records, advantages, args.entropy_bonus
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0 or batch == 1:
            print(
                f"{batch:5}  {batch * args.games:6}  {100 * np.mean(wins):4.0f}%  "
                f"{entropy:7.2f}  {len(earlier) + 1:9}{critic}"
            )
        if batch % args.gate_every == 0:
            black, white, rate = head_to_head(
                learner,
                champion,
                args.gate_games,
                args.size,
                args.komi,
                gate_seed + batch,
                args.opening,
            )
            versus_first = rate
            if rung:
                _, _, versus_first = head_to_head(
                    learner,
                    first,
                    args.gate_games,
                    args.size,
                    args.komi,
                    gate_seed + batch + 1,
                    args.opening,
                )
            line = (
                f"      gate at batch {batch}: vs champion {rung} "
                f"{100 * rate:.1f}% ({100 * black:.0f}% as Black, "
                f"{100 * white:.0f}% as White)"
                + (f"; vs champion 0 {100 * versus_first:.1f}%" if rung else "")
            )
            if rate >= args.promote_at:
                earlier.append(champion)
                champion = frozen(learner)
                rung += 1
                path = os.path.join(args.dir, f"champion-{rung}.pt")
                save_network(
                    path,
                    champion,
                    {
                        "komi": args.komi,
                        "batch": batch,
                        "rung": rung,
                        "win_rate_vs_previous": rate,
                    },
                )
                with open(log_path, "a", newline="") as f:
                    csv.writer(f, lineterminator="\n").writerow(
                        [rung, batch, rate, black, white, versus_first]
                    )
                line += f"  -> promoted: champion {rung}"
            print(line)
    save_network(
        os.path.join(args.dir, "learner.pt"),
        learner,
        {"komi": args.komi, "batch": args.batches},
    )
    print(
        f"\n{rung} promotion(s) in {args.batches} batches "
        f"({time.perf_counter() - start:.0f} s); the champions are in {args.dir}"
    )


if __name__ == "__main__":
    main()
