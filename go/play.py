"""Play games of Go between two players and summarize how they went.

Run from the repo root:

    uv run python go/play.py                      # 1000 random games on 5x5
    uv run python go/play.py --size 7 --games 500
    uv run python go/play.py --komi 2.5 --seed 1
    uv run python go/play.py --watch --seed 3     # one game, move by move
    uv run python go/play.py --black eye-aware --white eye-aware
    uv run python go/play.py --window --black eye-aware --white eye-aware
    uv run python go/play.py --window --black human --white eye-aware
    uv run python go/play.py --window --record --black human --white guided \\
        --model go/runs/komi.pt --simulations 100   # moves also printed
    uv run python go/play.py --window --black human --white learned \\
        --model go/runs/first.pt
    uv run python go/play.py --black learned --white learned --swap --opening 2 \\
        --black-model go/runs/value.pt --white-model go/runs/selfplay.pt

Players: random (uniform over legal moves, passing included),
eye-aware (random, but never filling its own eyes, passing only when
nothing else is left), human (you, clicking in the window: needs
--window) and learned (a network trained by learn.py: needs --model).
"""

import argparse
import contextlib
import statistics

from game import BLACK, EMPTY, WHITE, Game
from mcts import MCTSPlayer
from players import EyeAwareRandomPlayer, HumanPlayer, RandomOpening, RandomPlayer

# Computer players, by name. "human" (you, clicking in the window) and
# "learned" (a network trained by learn.py, from --model) are also
# choices for --black and --white.
PLAYERS = {"random": RandomPlayer, "eye-aware": EyeAwareRandomPlayer}
HUMAN = "human"
LEARNED = "learned"
MCTS = "mcts"  # tree search with eye-aware playouts: --simulations per move
GUIDED = "guided"  # tree search guided by a network (--model): --simulations
NETWORK_PLAYERS = (LEARNED, GUIDED)  # players that need a network file


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


def record_move(game, move, mover, captured):
    """after_move for --record: print the move as a numbered line, like
    "12. B C3, capturing 2"."""
    took = f", capturing {captured}" if captured else ""
    colour = "B" if mover == BLACK else "W"
    print(f"{len(game.moves):3}. {colour} {game.name(move)}{took}", flush=True)


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


def make_player(name, seed, view=None, model=None, simulations=400):
    """The player called name, seeded with seed; a human player asks
    through view (a BoardView), a learned one plays the network saved at
    model, always taking its likeliest move, and mcts searches with
    simulations simulations per move."""
    if name == HUMAN:
        return HumanPlayer(view)
    if name == MCTS:
        return MCTSPlayer(simulations, seed=seed)
    if name == GUIDED:
        from guided import GuidedSearchPlayer  # needs PyTorch
        from policy import load_network

        return GuidedSearchPlayer(load_network(model), simulations, seed=seed)
    if name == LEARNED:
        from policy import NetworkPlayer, load_network  # needs PyTorch

        return NetworkPlayer(load_network(model))
    return PLAYERS[name](seed)


def watch_in_window(
    game,
    black_name,
    white_name,
    seed,
    delay,
    black_model=None,
    white_model=None,
    simulations=(400, 400),
    record=False,
):
    """Play game in a pygame window between the players named (either may
    be "human", clicking in the window), pausing delay seconds after each
    computer move, then show the result with territory until a key is
    pressed. With record, also print the game at the console as it's
    played: the players, each move, and the final position and result."""
    from view import BoardView, QuitGame, turn_caption  # only with a window

    view = BoardView(game.size)
    black = make_player(
        black_name, None if seed is None else seed, view, black_model, simulations[0]
    )
    white = make_player(
        white_name,
        None if seed is None else seed + 1,
        view,
        white_model,
        simulations[1],
    )
    humans = {
        colour
        for colour, name in ((BLACK, black_name), (WHITE, white_name))
        if name == HUMAN
    }
    if record:
        print(
            f"Game on {game.size}x{game.size}, komi {game.komi:g}: Black "
            f"{describe_player(black_name, black_model, simulations[0])}, White "
            f"{describe_player(white_name, white_model, simulations[1])}",
            flush=True,
        )
    if BLACK not in humans:
        view.show(game, caption=turn_caption(game), seconds=delay)

    def show_move(game, move, mover, captured):
        if record:
            record_move(game, move, mover, captured)
        pause = 0.0 if mover in humans else delay
        caption = turn_caption(game, you=game.to_play in humans)
        return view.show(game, move, caption, seconds=pause)

    with contextlib.suppress(QuitGame):  # the window closed on your turn
        play_game(game, black, white, after_move=show_move)
    if game.over:
        area = game.area()
        caption = (
            f"{describe_result(game).capitalize()} after {len(game.moves)} moves\n"
            f"Black {area[BLACK]}, White {area[WHITE]} + komi {game.komi:g}"
            " (any key to close)"
        )
        view.finish(game, caption)
        if record:
            print(game)
        print(final_caption(game))
    else:
        view.close()
        if record:
            print(game)
        print(f"Stopped after {len(game.moves)} moves.")


def describe_result(game):
    margin = game.margin()
    if margin == 0:
        return "a draw"
    winner = "Black" if margin > 0 else "White"
    return f"{winner} by {abs(margin):g}"


def ending(game):
    return "two passes" if game.passes_in_a_row >= 2 else "move limit"


def print_head_to_head(games, names):
    """With --swap: each player's results as Black, as White and overall.
    names maps (game index parity) to who was Black: names[0] played Black
    in the even-numbered games and White in the odd ones."""
    first, second = names
    print(f"Head to head, colours alternating ({len(games)} games):")
    for player, black_parity in ((first, 0), (second, 1)):
        as_black = [g for i, g in enumerate(games) if i % 2 == black_parity]
        as_white = [g for i, g in enumerate(games) if i % 2 != black_parity]
        black_wins = sum(g.winner() == BLACK for g in as_black)
        white_wins = sum(g.winner() == WHITE for g in as_white)
        print(
            f"  {player:30} won {100 * black_wins / max(len(as_black), 1):5.1f}% "
            f"as Black, {100 * white_wins / max(len(as_white), 1):5.1f}% as White, "
            f"{100 * (black_wins + white_wins) / len(games):5.1f}% overall"
        )


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
        "--komi",
        type=float,
        default=None,
        help="points White gets (default: the komi a learned player was "
        "trained with, otherwise 0)",
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
        choices=[*PLAYERS, HUMAN, LEARNED, MCTS, GUIDED],
        default="random",
        help="Black's player (default: random)",
    )
    parser.add_argument(
        "--white",
        choices=[*PLAYERS, HUMAN, LEARNED, MCTS, GUIDED],
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
        "--record",
        action="store_true",
        help="with --window, print the game at the console as it's played "
        "(players, every move, the final position)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="with --window, seconds between moves (default: 0.5)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="the network file for learned players (saved by learn.py or selfplay.py)",
    )
    parser.add_argument(
        "--black-model",
        default=None,
        help="a different network for a learned Black (overrides --model)",
    )
    parser.add_argument(
        "--white-model",
        default=None,
        help="a different network for a learned White (overrides --model)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=400,
        help="for mcts players, simulations per move: more is stronger and "
        "slower (default: 400)",
    )
    parser.add_argument(
        "--black-simulations",
        type=int,
        default=None,
        help="simulations for a searching Black (overrides --simulations)",
    )
    parser.add_argument(
        "--white-simulations",
        type=int,
        default=None,
        help="simulations for a searching White (overrides --simulations)",
    )
    parser.add_argument(
        "--opening",
        type=int,
        default=0,
        help="random moves per side to start each game, so deterministic "
        "players don't repeat one game (default: 0)",
    )
    parser.add_argument(
        "--swap",
        action="store_true",
        help="alternate colours game by game and report each player's results",
    )
    args = parser.parse_args()
    for colour in ("black", "white"):
        name, model = getattr(args, colour), getattr(args, f"{colour}_model")
        if name in NETWORK_PLAYERS and not (model or args.model):
            parser.error(f"a {name} {colour} player needs --model or --{colour}-model")
        if model and name not in NETWORK_PLAYERS:
            parser.error(f"--{colour}-model needs --{colour} learned or guided")
    if args.model and not {args.black, args.white} & set(NETWORK_PLAYERS):
        parser.error("--model needs a learned or guided player")
    if HUMAN in (args.black, args.white) and not args.window:
        parser.error("a human player needs --window (you play by clicking)")
    if args.opening and HUMAN in (args.black, args.white):
        parser.error("--opening is for computer players, not human ones")
    if args.record and not args.window:
        parser.error("--record is for --window games (--watch prints every move)")
    if args.swap and (args.window or args.watch):
        parser.error("--swap is for many games, not --window or --watch")
    searchers = {MCTS, GUIDED}
    if args.simulations != 400 and not {args.black, args.white} & searchers:
        parser.error("--simulations needs an mcts or guided player")
    for colour in ("black", "white"):
        if getattr(args, f"{colour}_simulations") is not None:
            if getattr(args, colour) not in searchers:
                parser.error(f"--{colour}-simulations needs --{colour} mcts or guided")
        else:
            setattr(args, f"{colour}_simulations", args.simulations)
    if args.komi is None:
        args.komi = models_komi(parser, args)
    return args


def models_komi(parser, args):
    """The komi the learned players' networks were trained with (0 if none
    recorded one); an error if they disagree."""
    from policy import saved_komi  # needs PyTorch

    models = []
    for colour in ("black", "white"):
        if getattr(args, colour) in NETWORK_PLAYERS:
            models.append(getattr(args, f"{colour}_model") or args.model)
    komis = {saved_komi(m) for m in models} - {None}
    if len(komis) > 1:
        parser.error(
            f"the networks were trained with different komi ({sorted(komis)}); "
            "choose one with --komi"
        )
    return komis.pop() if komis else 0.0


def describe_player(name, model, simulations=400):
    if name == LEARNED:
        return f"learned ({model})"
    if name == GUIDED:
        return f"guided ({model}, {simulations} simulations)"
    if name == MCTS:
        return f"mcts ({simulations} simulations)"
    return name


def main():
    args = parse_args()
    seed = args.seed
    black_model = args.black_model or args.model
    white_model = args.white_model or args.model

    def new_game():
        return Game(args.size, komi=args.komi, max_moves=args.max_moves)

    if args.window:
        watch_in_window(
            new_game(),
            args.black,
            args.white,
            seed,
            args.delay,
            black_model,
            white_model,
            (args.black_simulations, args.white_simulations),
            args.record,
        )
        return
    black = make_player(
        args.black,
        None if seed is None else seed,
        model=black_model,
        simulations=args.black_simulations,
    )
    white = make_player(
        args.white,
        None if seed is None else seed + 1,
        model=white_model,
        simulations=args.white_simulations,
    )

    def with_opening(player, game_number, offset):
        if not args.opening:
            return player
        base = 0 if seed is None else seed
        return RandomOpening(player, args.opening, base + 7919 * game_number + offset)

    if args.watch:
        game = play_game(
            new_game(),
            with_opening(black, 0, 0),
            with_opening(white, 0, 1),
            after_move=print_move,
        )
        print(final_caption(game))
        return
    games = []
    for i in range(args.games):
        first, second = with_opening(black, i, 0), with_opening(white, i, 1)
        if args.swap and i % 2:
            first, second = second, first
        games.append(play_game(new_game(), first, second))
    print_summary(games)
    if args.swap:
        print_head_to_head(
            games,
            (
                describe_player(args.black, black_model, args.black_simulations),
                describe_player(args.white, white_model, args.white_simulations),
            ),
        )
    if args.black == args.white == LEARNED and not args.opening:
        print(
            "  (Both players always take their likeliest move, so every game is "
            "the same; --opening varies them.)"
        )


if __name__ == "__main__":
    main()
