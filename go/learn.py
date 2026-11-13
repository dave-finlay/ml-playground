"""Train a Go-playing network with REINFORCE against a fixed opponent.

Each batch, the network plays games against the opponent, alternating
Black and White, choosing its moves at random by its own probabilities.
The reward is the game's result from the network's side: +1 for a win,
-1 for a loss, 0 for a draw. Every one of the network's moves in a game
shares that result, less the batch's average result (a baseline, as in
CartPole), as its advantage, and one update makes moves followed by
better-than-average results more likely:

    loss = -mean over the network's moves of advantage * log pi(move | board)

Every eval_every batches the network plays eval_games more games without
sampling (always its likeliest move), half as each colour, and the best
version so far is kept.

Run from the repo root:

    uv run python go/learn.py
    uv run python go/learn.py --batches 300 --seed 1 --save go/runs/first.pt
    uv run python go/play.py --window --black human --white learned \\
        --model go/runs/first.pt
"""

import argparse
import copy
import os
import time

import numpy as np
import torch

from game import BLACK, WHITE, Game
from play import play_game
from players import EyeAwareRandomPlayer, RandomPlayer
from policy import NetworkPlayer, PolicyNetwork, masked_logits, save_network

OPPONENTS = {"random": RandomPlayer, "eye-aware": EyeAwareRandomPlayer}


def result_for(game, colour):
    """+1 if colour won game, -1 if it lost, 0 for a draw."""
    winner = game.winner()
    return 0.0 if winner not in (BLACK, WHITE) else 1.0 if winner == colour else -1.0


def play_batch(network, opponent, games, size, komi, rng, first_colour=BLACK):
    """Play games against opponent, the network sampling its moves and
    alternating colours. Returns (moves, results): each game's recorded
    (board, mask, move) list for the network, and its result."""
    moves, results = [], []
    for i in range(games):
        colour = (
            first_colour if i % 2 == 0 else (WHITE if first_colour == BLACK else BLACK)
        )
        record = []
        learner = NetworkPlayer(network, sample=True, rng=rng, record=record)
        black, white = (learner, opponent) if colour == BLACK else (opponent, learner)
        game = play_game(Game(size, komi=komi), black, white)
        moves.append(record)
        results.append(result_for(game, colour))
    return moves, results


def reinforce_loss(network, moves, results):
    """The REINFORCE loss for a batch: minus the mean over the network's
    moves of (game result - batch mean result) * log pi(move | board).
    Also returns the policy's mean entropy over those moves."""
    results = np.array(results)
    advantages = results - results.mean()
    boards, masks, chosen, weights = [], [], [], []
    for record, advantage in zip(moves, advantages, strict=True):
        for board, mask, move in record:
            boards.append(board)
            masks.append(mask)
            chosen.append(move)
            weights.append(advantage)
    boards = torch.from_numpy(np.stack(boards))
    masks = torch.from_numpy(np.stack(masks))
    logits = masked_logits(network(boards), masks)
    log_probs = torch.log_softmax(logits, dim=1)
    taken = log_probs[torch.arange(len(chosen)), torch.tensor(chosen)]
    loss = -(torch.tensor(weights, dtype=torch.float32) * taken).mean()
    probs = log_probs.exp()
    entropy = -(probs * log_probs.clamp(min=-30)).sum(dim=1).mean()
    return loss, float(entropy.detach())


def evaluate(network, opponent_class, games, size, komi, seed):
    """Win rates of the network (likeliest moves, no sampling) against
    the opponent: (as Black, as White), over games / 2 games each."""
    rates = {}
    for colour in (BLACK, WHITE):
        opponent = opponent_class(seed + colour)
        player = NetworkPlayer(network)
        wins = 0
        for _ in range(games // 2):
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
        "--opponent",
        choices=OPPONENTS,
        default="eye-aware",
        help="who it plays and learns against (default: eye-aware)",
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
        "--channels", type=int, default=32, help="convolution width (default: 32)"
    )
    parser.add_argument(
        "--layers", type=int, default=3, help="convolutions (default: 3)"
    )
    parser.add_argument(
        "--eval-every", type=int, default=20, help="batches between tests (default: 20)"
    )
    parser.add_argument(
        "--eval-games", type=int, default=200, help="games per test (default: 200)"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--save",
        default=None,
        help="where to save the best network (e.g. go/runs/first.pt)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else int(time.time())
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    network = PolicyNetwork(args.size, args.channels, args.layers)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    opponent_class = OPPONENTS[args.opponent]
    opponent = opponent_class(seed + 1)
    eval_seed = seed + 1_000_003

    black_rate, white_rate = evaluate(
        network, opponent_class, args.eval_games, args.size, args.komi, eval_seed
    )
    print(
        f"Before training, without sampling: wins {100 * black_rate:.0f}% as Black, "
        f"{100 * white_rate:.0f}% as White against {args.opponent}"
    )
    print(f"{'batch':>5}  {'games':>6}  {'won':>5}  {'length':>6}  {'entropy':>7}")
    best, best_rate, best_batch = copy.deepcopy(network.state_dict()), -1.0, 0
    start = time.perf_counter()
    for batch in range(1, args.batches + 1):
        first = BLACK if batch % 2 else WHITE
        moves, results = play_batch(
            network, opponent, args.games, args.size, args.komi, rng, first
        )
        loss, entropy = reinforce_loss(network, moves, results)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        won = np.mean([r > 0 for r in results])
        length = np.mean([len(m) for m in moves])
        if batch % 10 == 0 or batch == 1:
            print(
                f"{batch:5}  {batch * args.games:6}  {100 * won:4.0f}%  "
                f"{length:6.1f}  {entropy:7.2f}"
            )
        if batch % args.eval_every == 0 or batch == args.batches:
            black_rate, white_rate = evaluate(
                network,
                opponent_class,
                args.eval_games,
                args.size,
                args.komi,
                eval_seed,
            )
            rate = (black_rate + white_rate) / 2
            mark = ""
            if rate >= best_rate:
                best, best_rate, best_batch = (
                    copy.deepcopy(network.state_dict()),
                    rate,
                    batch,
                )
                mark = "  (best so far)"
            print(
                f"      test without sampling: wins {100 * black_rate:.0f}% as Black, "
                f"{100 * white_rate:.0f}% as White{mark}"
            )
    seconds = time.perf_counter() - start
    print(
        f"\nBest: batch {best_batch}, winning {100 * best_rate:.0f}% on average "
        f"against {args.opponent} ({seconds:.0f} s of training)"
    )
    if args.save:
        network.load_state_dict(best)
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        save_network(args.save, network, {"batch": best_batch, "win_rate": best_rate})
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
