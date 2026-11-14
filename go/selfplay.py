"""Train a Go-playing network by self-play, with REINFORCE.

Each batch of games, the network plays either itself or a past copy of
itself from a pool (a snapshot is added every pool_every batches), half
and half by default, alternating colours. It samples its moves from its
own probabilities. Against itself both sides' moves are trained on;
against a past copy only its own.

Each move's advantage is its side's result (+1 win, -1 loss, 0 draw)
less the average result of that colour in the batch: a separate
baseline for Black and White, so Black's first-move advantage doesn't
make all of Black's moves look good. The update is REINFORCE's (see
policy.policy_gradient_loss), with an optional bonus for keeping the
move probabilities spread out (entropy). With --value the network has a
critic (a value head), and each move's advantage is instead its result
less the critic's estimate of the position before it (see
policy.actor_critic_loss); the critic learns from the same moves. Without
the entropy bonus, self-play narrows
into one style that beats its recent selves but loses to older ones
(a 300-batch run on 5 x 5 rose and then fell below where it started).

Win rates against itself say nothing (it always plays its equal), so
every eval_every batches it's tested, always taking its likeliest move,
against fixed yardsticks: the eye-aware random player, and the network
it started from (or a network given with --benchmark). Two deterministic
networks would play the same game every time, so each network-against-
network test game opens with a few random moves.

Run from the repo root:

    uv run python go/selfplay.py --init go/runs/eye-aware-5x5-seed1.pt --seed 1
    uv run python go/selfplay.py --batches 300 --save go/runs/selfplay.pt
    uv run python go/selfplay.py --resume go/runs/selfplay-resume.pt --batches 1000

With --save, a resume file is written beside the network at every test;
--resume carries the run on from there exactly as if it had never
stopped (the network, the optimizer's running averages, the pool of past
copies, the best so far and the random number generators all carry
over).
"""

import argparse
import copy
import os
import random
import time

import numpy as np
import torch

from game import BLACK, WHITE, Game, other
from learn import result_for
from play import play_game
from players import EyeAwareRandomPlayer, RandomOpening
from policy import (
    NetworkPlayer,
    PolicyNetwork,
    actor_critic_loss,
    load_network,
    policy_gradient_loss,
    save_network,
    upgraded,
    with_value_head,
)


def play_training_game(network, opponent_network, colour, size, komi, rng):
    """Play one training game: network (sampling) as colour, against
    opponent_network (sampling), or against itself if that's None.
    Returns [(records, result)]: the moves to train on for each side the
    network played, with that side's result."""
    own, theirs = [], []
    player = NetworkPlayer(network, sample=True, rng=rng, record=own)
    if opponent_network is None:
        opponent = NetworkPlayer(network, sample=True, rng=rng, record=theirs)
    else:
        opponent = NetworkPlayer(opponent_network, sample=True, rng=rng)
    black, white = (player, opponent) if colour == BLACK else (opponent, player)
    game = play_game(Game(size, komi=komi), black, white)
    sides = [(own, colour, result_for(game, colour))]
    if opponent_network is None:
        sides.append((theirs, other(colour), result_for(game, other(colour))))
    return sides


def moves_and_results(sides):
    """Flatten sides into (records, results): every recorded move, and the
    result of the side that made it."""
    records, results = [], []
    for side_records, _, result in sides:
        records += side_records
        results += [result] * len(side_records)
    return records, results


def advantages_by_colour(sides):
    """Each side's result less the average result of sides of its colour:
    returns (records, advantages), flattened over every recorded move."""
    means = {}
    for colour in (BLACK, WHITE):
        results = [result for _, side_colour, result in sides if side_colour == colour]
        means[colour] = float(np.mean(results)) if results else 0.0
    records, advantages = [], []
    for side_records, colour, result in sides:
        records += side_records
        advantages += [result - means[colour]] * len(side_records)
    return records, advantages


def win_rates(network, make_opponent, games, size, komi, seed, opening=0):
    """Win rates of network (likeliest moves) against make_opponent(seed):
    (as Black, as White), over games / 2 games each. With opening, both
    sides' first `opening` moves are random."""
    rates = {}
    for colour in (BLACK, WHITE):
        wins = 0
        for i in range(games // 2):
            game_seed = seed + 1000 * colour + i
            player = NetworkPlayer(network)
            opponent = make_opponent(game_seed)
            if opening:
                player = RandomOpening(player, opening, game_seed)
                opponent = RandomOpening(opponent, opening, game_seed + 1)
            black, white = (player, opponent) if colour == BLACK else (opponent, player)
            game = play_game(Game(size, komi=komi), black, white)
            wins += game.winner() == colour
        rates[colour] = wins / (games // 2)
    return rates[BLACK], rates[WHITE]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=5, help="board size (default: 5)")
    parser.add_argument("--komi", type=float, default=0.0, help="komi (default: 0)")
    parser.add_argument(
        "--init", default=None, help="network to start from (default: a fresh one)"
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="network to test against (default: the one started from)",
    )
    parser.add_argument(
        "--batches", type=int, default=200, help="updates (default: 200)"
    )
    parser.add_argument(
        "--games", type=int, default=64, help="games per update (default: 64)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam step size (default: 1e-3)",
    )
    parser.add_argument(
        "--pool-every",
        type=int,
        default=10,
        help="batches between snapshots added to the pool (default: 10)",
    )
    parser.add_argument(
        "--pool-fraction",
        type=float,
        default=0.5,
        help="share of games against a past copy rather than itself (default: 0.5)",
    )
    parser.add_argument(
        "--entropy-bonus",
        type=float,
        default=0.03,
        help="reward for keeping move probabilities spread out; without it "
        "self-play collapses into a narrow style (default: 0.03)",
    )
    parser.add_argument(
        "--eval-every", type=int, default=20, help="batches between tests (default: 20)"
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=100,
        help="games per test opponent (default: 100)",
    )
    parser.add_argument(
        "--opening",
        type=int,
        default=2,
        help="random opening moves per side in network-against-network tests "
        "(default: 2)",
    )
    parser.add_argument(
        "--value",
        action="store_true",
        help="add a critic (value head): each move's advantage becomes its "
        "result less the critic's estimate before it, instead of less its "
        "colour's average",
    )
    parser.add_argument(
        "--value-weight",
        type=float,
        default=0.5,
        help="with --value, the weight of the critic's loss (default: 0.5)",
    )
    parser.add_argument(
        "--value-detached",
        action="store_true",
        help="with --value, train the critic without its loss changing the "
        "shared convolutions",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--resume",
        default=None,
        help="carry on a run from its resume file (written beside --save) up to "
        "--batches in total, with its settings; only --batches and --save may "
        "be given with it",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="where to save the best network tested (its last version goes "
        "beside it, with -last added to the name, and a resume file with -resume)",
    )
    args = parser.parse_args()
    if args.resume:
        allowed = {"resume", "batches", "save", "help"}
        changed = [
            "--" + action.dest.replace("_", "-")
            for action in parser._actions
            if action.dest not in allowed
            and getattr(args, action.dest) != parser.get_default(action.dest)
        ]
        if changed:
            parser.error(
                "--resume carries on with the saved settings: drop " + " ".join(changed)
            )
    return args


def resume_path(save):
    """Where the resume file for a run saved at save goes."""
    root, extension = os.path.splitext(save)
    return f"{root}-resume{extension or '.pt'}"


def network_from(saved):
    """A network from a {"settings", "weights"} dict."""
    network = PolicyNetwork(**saved["settings"])
    network.load_state_dict(saved["weights"])
    return network


def save_networks(save, network, best, batch, komi):
    """Save the best network tested to save, and network (the last) beside
    it with -last added to the name, recording the komi trained with."""
    os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
    root, extension = os.path.splitext(save)
    last = f"{root}-last{extension or '.pt'}"
    save_network(last, network, {"batch": batch, "komi": komi})
    best_network = network_from(
        {"settings": network.settings(), "weights": best["weights"]}
    )
    save_network(save, best_network, {"batch": best["batch"], "komi": komi})
    return last


def main():
    args = parse_args()
    if args.resume:
        state = torch.load(args.resume, weights_only=False)
        settings = state["settings"]
        if args.batches <= state["batch"]:
            raise SystemExit(
                f"selfplay.py: error: {args.resume} is already at batch "
                f"{state['batch']}; give --batches more than that"
            )
        args = argparse.Namespace(
            **{
                **settings,
                "batches": args.batches,
                "resume": args.resume,
                "save": args.save or state["save"],
            }
        )
        seed = state["seed"]
    else:
        state = None
        seed = args.seed if args.seed is not None else int(time.time())
        torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pick = random.Random(seed)

    if state:
        network = network_from(state["network"])
        benchmark = network_from(state["benchmark"]) if state["benchmark"] else None
        if benchmark is not None:
            benchmark.eval()
    else:
        if args.init:
            network = load_network(args.init)
            network.train()
        else:
            network = PolicyNetwork(args.size)
        if args.value and not network.has_value:
            network = with_value_head(network)
        if args.komi and not network.has_colour:
            network = upgraded(network, colour=True)
            print(
                f"Adding a colour plane: with komi {args.komi:g}, a position's "
                "value depends on who's to move"
            )
        benchmark_path = args.benchmark or args.init
        benchmark = load_network(benchmark_path) if benchmark_path else None
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    pool = [copy.deepcopy(network).eval()]
    best = {"score": -1.0, "batch": 0, "weights": copy.deepcopy(network.state_dict())}
    first_batch = 1
    if state:
        optimizer.load_state_dict(state["optimizer"])
        pool = []
        for weights in state["pool"]:
            past = network_from({"settings": network.settings(), "weights": weights})
            pool.append(past.eval())
        best = state["best"]
        rng.bit_generator.state = state["rng"]
        pick.setstate(state["pick"])
        torch.set_rng_state(state["torch"])
        first_batch = state["batch"] + 1
    eval_seed = seed + 1_000_003

    def write_resume(batch):
        """Save everything needed to carry on exactly from after batch."""
        if not args.save:
            return
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        settings = {
            k: v
            for k, v in vars(args).items()
            if k not in ("batches", "resume", "save")
        }
        torch.save(
            {
                "settings": settings,
                "seed": seed,
                "batch": batch,
                "save": args.save,
                "network": {
                    "settings": network.settings(),
                    "weights": network.state_dict(),
                },
                "benchmark": None
                if benchmark is None
                else {
                    "settings": benchmark.settings(),
                    "weights": benchmark.state_dict(),
                },
                "optimizer": optimizer.state_dict(),
                "pool": [past.state_dict() for past in pool],
                "best": best,
                "rng": rng.bit_generator.state,
                "pick": pick.getstate(),
                "torch": torch.get_rng_state(),
            },
            resume_path(args.save),
        )

    def test(label, batch=0):
        """Print the yardstick win rates, and keep the network if it scores
        best so far: by its average win rate against the benchmark network
        (or against eye-aware without one)."""
        b, w = win_rates(
            network,
            EyeAwareRandomPlayer,
            args.eval_games,
            args.size,
            args.komi,
            eval_seed,
        )
        score = (b + w) / 2
        line = f"{label}: vs eye-aware {100 * b:3.0f}% / {100 * w:3.0f}%"
        if benchmark is not None:
            b, w = win_rates(
                network,
                lambda s: NetworkPlayer(benchmark),
                args.eval_games,
                args.size,
                args.komi,
                eval_seed,
                opening=args.opening,
            )
            line += f";  vs start {100 * b:3.0f}% / {100 * w:3.0f}%"
            score = (b + w) / 2
        if batch and score >= best["score"]:
            best.update(
                score=score, batch=batch, weights=copy.deepcopy(network.state_dict())
            )
            line += "  (best so far)"
        print(line)

    print("Tests: win rates as Black / as White, always taking the likeliest move")
    if state:
        print(
            f"Resuming {args.resume} after batch {state['batch']} (best so far: "
            f"batch {best['batch']}, {100 * best['score']:.0f}%), to batch "
            f"{args.batches}"
        )
    else:
        test("Start")
    print(
        f"{'batch':>5}  {'games':>6}  {'Black won':>9}  {'length':>6}  "
        f"{'entropy':>7}  pool"
        + ("  value loss, sign right" if network.has_value else "")
    )
    start = time.perf_counter()
    for batch in range(first_batch, args.batches + 1):
        sides, black_wins, lengths = [], [], []
        for i in range(args.games):
            colour = BLACK if (batch + i) % 2 else WHITE
            against_pool = pick.random() < args.pool_fraction
            opponent = pick.choice(pool) if against_pool else None
            game_sides = play_training_game(
                network, opponent, colour, args.size, args.komi, rng
            )
            sides += game_sides
            first = game_sides[0]
            black_result = first[2] if first[1] == BLACK else -first[2]
            black_wins.append(black_result > 0)
            lengths.append(sum(len(records) for records, _, _ in game_sides))
        critic = ""
        if network.has_value:
            records, results = moves_and_results(sides)
            loss, entropy, value_loss, right = actor_critic_loss(
                network,
                records,
                results,
                args.value_weight,
                args.entropy_bonus,
                args.value_detached,
            )
            critic = f"  {value_loss:10.3f}  {100 * right:9.0f}%"
        else:
            records, advantages = advantages_by_colour(sides)
            loss, entropy = policy_gradient_loss(
                network, records, advantages, args.entropy_bonus
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % args.pool_every == 0:
            pool.append(copy.deepcopy(network).eval())
        if batch % 10 == 0 or batch == 1:
            print(
                f"{batch:5}  {batch * args.games:6}  "
                f"{100 * np.mean(black_wins):8.0f}%  {np.mean(lengths):6.1f}  "
                f"{entropy:7.2f}  {len(pool):4}{critic}"
            )
        if batch % args.eval_every == 0 or batch == args.batches:
            test(f"      batch {batch}", batch)
            write_resume(batch)
    print(
        f"\nBest: batch {best['batch']}, scoring {100 * best['score']:.0f}% "
        f"({time.perf_counter() - start:.0f} s of training)"
    )
    if args.save:
        last = save_networks(args.save, network, best, args.batches, args.komi)
        print(
            f"Saved the best to {args.save}, the last to {last}, and what's "
            f"needed to carry on to {resume_path(args.save)}"
        )


if __name__ == "__main__":
    main()
