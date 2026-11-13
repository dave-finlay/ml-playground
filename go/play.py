"""Play games of Go between two players and summarize how they went.

Run from the repo root:

    uv run python go/play.py                      # 1000 random games on 5x5
    uv run python go/play.py --size 7 --games 500
    uv run python go/play.py --komi 2.5 --seed 1
    uv run python go/play.py --watch --seed 3     # one game, move by move
    uv run python go/play.py --black eye-aware --white eye-aware

Players: random (uniform over legal moves, passing included) and
eye-aware (random, but never filling its own eyes, passing only when
nothing else is left).
"""

import argparse
import statistics

from game import BLACK, EMPTY, WHITE, Game
from players import EyeAwareRandomPlayer, RandomPlayer

PLAYERS = {"random": RandomPlayer, "eye-aware": EyeAwareRandomPlayer}


def play_game(game, black, white, watch=False):
    """Play game to the end, black and white choosing the moves, and
    return it. With watch, print the board after every move."""
    while not game.over:
        mover = game.to_play
        move = (black if mover == BLACK else white)(game)
        captured_before = game.captures[mover]
        game.play(move)
        if watch:
            colour = "Black" if mover == BLACK else "White"
            captured = game.captures[mover] - captured_before
            took = f", capturing {captured}" if captured else ""
            print(f"Move {len(game.moves)}: {colour} {game.name(move)}{took}")
            print(game)
            print()
    return game


def describe_result(game):
    margin = game.margin()
    if margin == 0:
        return "a draw"
    winner = "Black" if margin > 0 else "White"
    return f"{winner} by {abs(margin):g}"


def ending(game):
    return "two passes" if game.passes_in_a_row >= 2 else "move limit"


def print_summary(games):
    """Print statistics over finished games."""
    count = len(games)
    winners = [g.winner() for g in games]
    margins = [g.margin() for g in games]
    lengths = [len(g.moves) for g in games]
    endings = [ending(g) for g in games]
    captures = [g.captures[BLACK] + g.captures[WHITE] for g in games]
    first = games[0]
    print(
        f"{count} games on {first.size}x{first.size}, komi {first.komi:g}, "
        f"move limit {first.max_moves}:"
    )
    print(
        f"  Black won {100 * winners.count(BLACK) / count:.1f}%, White "
        f"{100 * winners.count(WHITE) / count:.1f}%, draws "
        f"{100 * winners.count(EMPTY) / count:.1f}%"
    )
    quartiles = statistics.quantiles(margins, n=4)
    print(
        f"  Margin (Black minus White, komi included): mean "
        f"{statistics.mean(margins):+.1f}, median {statistics.median(margins):+g}, "
        f"middle half {quartiles[0]:+g} to {quartiles[2]:+g}, range "
        f"{min(margins):+g} to {max(margins):+g}"
    )
    print(
        f"  Length: mean {statistics.mean(lengths):.0f} moves, median "
        f"{statistics.median(lengths):g}, longest {max(lengths)}"
    )
    print(
        f"  Ended by two passes {100 * endings.count('two passes') / count:.0f}%, "
        f"by the move limit {100 * endings.count('move limit') / count:.0f}%"
    )
    print(f"  Stones captured per game: mean {statistics.mean(captures):.1f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=5, help="board size (default: 5)")
    parser.add_argument(
        "--komi", type=float, default=0.0, help="points White gets (default: 0)"
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
        help="moves after which a game ends (default: 4 per point)",
    )
    parser.add_argument(
        "--games", type=int, default=1000, help="games to play (default: 1000)"
    )
    parser.add_argument(
        "--black",
        choices=PLAYERS,
        default="random",
        help="Black's player (default: random)",
    )
    parser.add_argument(
        "--white",
        choices=PLAYERS,
        default="random",
        help="White's player (default: random)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed (default: none)"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="play one game and show the board after every move",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    black = PLAYERS[args.black](None if seed is None else seed)
    white = PLAYERS[args.white](None if seed is None else seed + 1)

    def new_game():
        return Game(args.size, komi=args.komi, max_moves=args.max_moves)

    if args.watch:
        game = play_game(new_game(), black, white, watch=True)
        area = game.area()
        print(
            f"Game over after {len(game.moves)} moves ({ending(game)}): Black "
            f"{area[BLACK]}, White {area[WHITE]} + komi {game.komi:g}: "
            f"{describe_result(game)}."
        )
        return
    print_summary([play_game(new_game(), black, white) for _ in range(args.games)])


if __name__ == "__main__":
    main()
