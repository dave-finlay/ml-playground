"""Play games of Go between two players and summarize how they went.

Run from the repo root:

    uv run python go/play.py                      # 1000 random games on 5x5
    uv run python go/play.py --size 7 --games 500
    uv run python go/play.py --komi 2.5 --seed 1
    uv run python go/play.py --watch --seed 3     # one game, move by move
    uv run python go/play.py --window --black eye-aware --white eye-aware
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


def describe_move(game, move, mover, captured):
    """A line like "Move 12: Black C3, capturing 2"."""
    colour = "Black" if mover == BLACK else "White"
    took = f", capturing {captured}" if captured else ""
    return f"Move {len(game.moves)}: {colour} {game.name(move)}{took}"


def play_game(game, black, white, after_move=None):
    """Play game to the end, black and white choosing the moves, and
    return it. after_move, if given, is called after every move with
    (game, move, mover, stones captured); returning False stops the
    game early."""
    while not game.over:
        mover = game.to_play
        move = (black if mover == BLACK else white)(game)
        captured_before = game.captures[mover]
        game.play(move)
        if after_move is not None:
            captured = game.captures[mover] - captured_before
            if after_move(game, move, mover, captured) is False:
                break
    return game


def print_move(game, move, mover, captured):
    """after_move for play_game: print the move and the board."""
    print(describe_move(game, move, mover, captured))
    print(game)
    print()


def final_caption(game):
    area = game.area()
    return (
        f"Game over after {len(game.moves)} moves ({ending(game)}): Black "
        f"{area[BLACK]}, White {area[WHITE]} + komi {game.komi:g}: "
        f"{describe_result(game)}."
    )


def watch_in_window(game, black, white, delay):
    """Play game in a pygame window, pausing delay seconds after each
    move, then show the result with territory until a key is pressed."""
    from view import BoardView  # only needed with a window

    view = BoardView(game.size)
    names = {BLACK: "Black", WHITE: "White"}
    view.show(game, caption=f"{names[BLACK]} to play", seconds=delay)

    def show_move(game, move, mover, captured):
        line = describe_move(game, move, mover, captured)
        tally = (
            f"Captured so far: Black {game.captures[BLACK]}, "
            f"White {game.captures[WHITE]}"
        )
        return view.show(game, move, f"{line}\n{tally}", seconds=delay)

    play_game(game, black, white, after_move=show_move)
    if game.over:
        area = game.area()
        caption = (
            f"{describe_result(game).capitalize()} after {len(game.moves)} moves\n"
            f"Black {area[BLACK]}, White {area[WHITE]} + komi {game.komi:g}"
            " (any key to close)"
        )
        view.finish(game, caption)
        print(final_caption(game))
    else:
        view.close()
        print(f"Stopped after {len(game.moves)} moves.")


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
        help="play one game and print the board after every move",
    )
    parser.add_argument(
        "--window",
        action="store_true",
        help="play one game in a window (space pauses, Escape stops)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="with --window, seconds between moves (default: 0.5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    black = PLAYERS[args.black](None if seed is None else seed)
    white = PLAYERS[args.white](None if seed is None else seed + 1)

    def new_game():
        return Game(args.size, komi=args.komi, max_moves=args.max_moves)

    if args.window:
        watch_in_window(new_game(), black, white, args.delay)
        return
    if args.watch:
        game = play_game(new_game(), black, white, after_move=print_move)
        print(final_caption(game))
        return
    print_summary([play_game(new_game(), black, white) for _ in range(args.games)])


if __name__ == "__main__":
    main()
