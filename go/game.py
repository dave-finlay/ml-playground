"""The rules of Go, for a small square board.

Close to the Tromp-Taylor rules, the simplest complete rules of Go:

- Players alternate placing a stone on an empty point, Black first, or
  passing.
- After a stone is placed, any opponent group (stones of one colour
  joined horizontally or vertically) left with no liberties (adjacent
  empty points) is captured and removed.
- Suicide is illegal: a move may not leave the mover's own group with
  no liberties after captures. (Tromp-Taylor allows suicide; Chinese and
  Japanese rules don't, and forbidding it removes pointless moves.)
- Positional superko: a move may not recreate any earlier board
  position. This includes the basic ko rule (no immediately retaking a
  single stone), and guarantees the game can't cycle.
- The game ends after two passes in a row, or after max_moves moves.
- Area scoring: each player scores their stones on the board plus the
  empty points from which only their stones can be reached (their
  territory). White adds komi, compensation for moving second.

Points are numbered 0 to size * size - 1, row by row from the top left;
PASS (size * size) is the pass move.
"""

EMPTY, BLACK, WHITE = 0, 1, 2
SYMBOLS = {EMPTY: ".", BLACK: "X", WHITE: "O"}
COLUMNS = "ABCDEFGHJKLMNOPQRST"  # Go skips I


def other(colour):
    """The opposing colour."""
    return WHITE if colour == BLACK else BLACK


class IllegalMove(ValueError):
    """A move the rules don't allow."""


class Game:
    """A game of Go in progress.

    size is the board's width and height (up to 19), komi the points
    White gets for moving second, and max_moves the move limit after
    which the game ends and is scored as it stands (default: 4 moves per
    point, ample for sensible play; random play can go on much longer).
    """

    def __init__(self, size=5, komi=0.0, max_moves=None):
        if not 2 <= size <= len(COLUMNS):
            raise ValueError(f"size should be 2 to {len(COLUMNS)}")
        self.size = size
        self.komi = komi
        self.max_moves = 4 * size * size if max_moves is None else max_moves
        self.pass_move = size * size
        self.board = [EMPTY] * (size * size)
        self.to_play = BLACK
        self.moves = []  # every move played, PASS included
        self.captures = {BLACK: 0, WHITE: 0}  # stones each colour has captured
        self.passes_in_a_row = 0
        self._seen = {tuple(self.board)}  # positions so far, for superko
        self._neighbours = [self._neighbours_of(p) for p in range(size * size)]

    @staticmethod
    def from_rows(rows, to_play=BLACK, komi=0.0, max_moves=None):
        """A game starting from a position given as rows of text, top row
        first: X for Black, O for White, . for empty, spaces ignored. Only
        that position counts as seen before, for superko."""
        cells = ["".join(row.split()) for row in rows]
        size = len(cells)
        if any(len(row) != size for row in cells):
            raise ValueError("the rows should make a square")
        game = Game(size, komi=komi, max_moves=max_moves)
        symbols = {symbol: colour for colour, symbol in SYMBOLS.items()}
        game.board = [symbols[c] for row in cells for c in row]
        game.to_play = to_play
        game._seen = {tuple(game.board)}
        return game

    def _neighbours_of(self, point):
        row, column = divmod(point, self.size)
        neighbours = []
        if row > 0:
            neighbours.append(point - self.size)
        if row < self.size - 1:
            neighbours.append(point + self.size)
        if column > 0:
            neighbours.append(point - 1)
        if column < self.size - 1:
            neighbours.append(point + 1)
        return neighbours

    def neighbours(self, point):
        """The points horizontally or vertically next to point."""
        return self._neighbours[point]

    def group(self, point, board=None):
        """Return (stones, liberties) for the group at point: the set of
        points in it, and the set of empty points next to it."""
        board = self.board if board is None else board
        colour = board[point]
        stones, liberties, frontier = {point}, set(), [point]
        while frontier:
            for neighbour in self._neighbours[frontier.pop()]:
                if board[neighbour] == colour and neighbour not in stones:
                    stones.add(neighbour)
                    frontier.append(neighbour)
                elif board[neighbour] == EMPTY:
                    liberties.add(neighbour)
        return stones, liberties

    def _result_of(self, point):
        """The board after the player to move plays at point, and how many
        stones that captures, or (None, reason) if the move is illegal."""
        if point == self.pass_move:
            return list(self.board), 0
        if not 0 <= point < self.pass_move:
            return None, "not a point on the board"
        if self.board[point] != EMPTY:
            return None, "occupied"
        board = list(self.board)
        board[point] = self.to_play
        captured = 0
        for neighbour in self._neighbours[point]:
            if board[neighbour] == other(self.to_play):
                stones, liberties = self.group(neighbour, board)
                if not liberties:
                    for stone in stones:
                        board[stone] = EMPTY
                    captured += len(stones)
        if not self.group(point, board)[1]:
            return None, "suicide"
        if tuple(board) in self._seen:
            return None, "repeats an earlier position (superko)"
        return board, captured

    def why_illegal(self, point):
        """Why the player to move may not play at point (like "occupied"
        or "suicide"), or None if they may."""
        if self.over:
            return "the game is over"
        board, reason = self._result_of(point)
        return None if board is not None else reason

    def is_legal(self, point):
        """Whether the player to move may play at point (PASS is always
        legal while the game goes on)."""
        return not self.over and self._result_of(point)[0] is not None

    def legal_moves(self):
        """Every legal move for the player to move, PASS last."""
        if self.over:
            return []
        return [p for p in range(self.pass_move + 1) if self._result_of(p)[0]]

    def play(self, point):
        """Play point (or PASS) for the player to move. Raises IllegalMove
        if the rules don't allow it."""
        if self.over:
            raise IllegalMove("the game is over")
        board, captured = self._result_of(point)
        if board is None:
            raise IllegalMove(f"{self.name(point)}: {captured}")
        if point == self.pass_move:
            self.passes_in_a_row += 1
        else:
            self.passes_in_a_row = 0
            self.board = board
            self._seen.add(tuple(board))
            self.captures[self.to_play] += captured
        self.moves.append(point)
        self.to_play = other(self.to_play)

    @property
    def over(self):
        """Whether the game has ended: two passes in a row, or the move
        limit reached."""
        return self.passes_in_a_row >= 2 or len(self.moves) >= self.max_moves

    def territory(self):
        """Return {point: colour} for the empty points that count for a
        colour: those in empty regions bordered only by that colour's
        stones. Empty points in regions touching both colours (or none)
        are left out."""
        territory = {}
        counted = set()
        for point in range(self.pass_move):
            if self.board[point] != EMPTY or point in counted:
                continue
            region, borders, frontier = {point}, set(), [point]
            while frontier:
                for neighbour in self._neighbours[frontier.pop()]:
                    if self.board[neighbour] == EMPTY:
                        if neighbour not in region:
                            region.add(neighbour)
                            frontier.append(neighbour)
                    else:
                        borders.add(self.board[neighbour])
            counted |= region
            if len(borders) == 1:
                owner = borders.pop()
                territory.update(dict.fromkeys(region, owner))
        return territory

    def area(self):
        """Return {BLACK: points, WHITE: points}: each colour's stones plus
        its territory (see territory)."""
        owners = list(self.territory().values())
        return {
            colour: self.board.count(colour) + owners.count(colour)
            for colour in (BLACK, WHITE)
        }

    def margin(self):
        """Black's score minus White's (komi included): positive means
        Black is ahead."""
        area = self.area()
        return area[BLACK] - area[WHITE] - self.komi

    def winner(self):
        """BLACK or WHITE by the current score, or EMPTY for a draw."""
        margin = self.margin()
        return BLACK if margin > 0 else WHITE if margin < 0 else EMPTY

    def name(self, point):
        """A point's name, like C3 (columns A, B, ... skipping I; rows
        numbered from the bottom), or "pass"."""
        if point == self.pass_move:
            return "pass"
        row, column = divmod(point, self.size)
        return f"{COLUMNS[column]}{self.size - row}"

    def __str__(self):
        rows = []
        for row in range(self.size):
            cells = " ".join(
                SYMBOLS[self.board[row * self.size + column]]
                for column in range(self.size)
            )
            rows.append(f"{self.size - row:2} {cells}")
        rows.append("   " + " ".join(COLUMNS[: self.size]))
        return "\n".join(rows)
