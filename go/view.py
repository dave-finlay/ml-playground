"""A pygame window showing a game of Go as it's played.

play.py uses it with --window. The board is drawn wood-coloured with
its grid, the stones, a red dot on the latest stone, and a caption with
the move and the captures so far. When the game ends, each side's
territory is marked with small squares of its colour and the caption
gives the score.

While a game plays: space pauses and resumes, and closing the window
(or Escape) stops it. When it's a person's turn (ask_move), they click
a point to play there, or press P to pass.
"""

import pygame

from game import BLACK, COLUMNS, WHITE

COLOUR_NAMES = {BLACK: "Black", WHITE: "White"}

BACKGROUND = (219, 178, 98)
LINE = (40, 30, 20)
BLACK_STONE = (25, 25, 25)
WHITE_STONE = (240, 240, 235)
LAST_MOVE = (200, 30, 30)
CAPTION = (30, 30, 30)
CAPTION_HEIGHT = 56


def board_geometry(size, cell):
    """Return (margin, board_pixels): the space around the grid (room for
    the edge stones and the coordinates beyond them), and the width and
    height of the board area."""
    margin = int(cell * 1.3)
    return margin, 2 * margin + (size - 1) * cell


def star_points(size):
    """The marked points: the centre, and on boards of 7 or more the
    points three in from each corner's edges (or two on 7 x 7)."""
    centre = size // 2
    points = [(centre, centre)] if size % 2 else []
    if size >= 7:
        inset = 2 if size < 13 else 3
        far = size - 1 - inset
        points += [(inset, inset), (inset, far), (far, inset), (far, far)]
    return points


def draw_board(surface, game, cell, last_move=None, caption="", final=False):
    """Draw game's position on surface: grid, coordinates, stones, a dot on
    last_move (a point), the caption below, and with final, territory."""
    size = game.size
    margin, pixels = board_geometry(size, cell)
    surface.fill(BACKGROUND)

    def centre(point):
        row, column = divmod(point, size)
        return margin + column * cell, margin + row * cell

    end = margin + (size - 1) * cell
    for i in range(size):
        offset = margin + i * cell
        pygame.draw.line(surface, LINE, (margin, offset), (end, offset), 2)
        pygame.draw.line(surface, LINE, (offset, margin), (offset, end), 2)
    for row, column in star_points(size):
        star = (margin + column * cell, margin + row * cell)
        pygame.draw.circle(surface, LINE, star, max(3, cell // 14))

    font = pygame.font.SysFont(None, max(18, cell // 3))
    label = int(cell * 0.35)  # from the window's edge, clear of edge stones
    for i in range(size):
        letter = font.render(COLUMNS[i], True, LINE)
        number = font.render(str(size - i), True, LINE)
        x = margin + i * cell
        y = margin + i * cell
        surface.blit(letter, letter.get_rect(center=(x, label)))
        surface.blit(letter, letter.get_rect(center=(x, pixels - label)))
        surface.blit(number, number.get_rect(center=(label, y)))
        surface.blit(number, number.get_rect(center=(pixels - label, y)))

    radius = int(cell * 0.46)
    for point, colour in enumerate(game.board):
        if colour == BLACK:
            pygame.draw.circle(surface, BLACK_STONE, centre(point), radius)
        elif colour == WHITE:
            pygame.draw.circle(surface, WHITE_STONE, centre(point), radius)
            pygame.draw.circle(surface, LINE, centre(point), radius, 1)
    if last_move is not None and last_move != game.pass_move:
        pygame.draw.circle(surface, LAST_MOVE, centre(last_move), max(4, cell // 9))

    if final:
        half = max(5, cell // 7)
        for point, colour in game.territory().items():
            x, y = centre(point)
            square = pygame.Rect(x - half, y - half, 2 * half, 2 * half)
            fill = BLACK_STONE if colour == BLACK else WHITE_STONE
            pygame.draw.rect(surface, fill, square)
            pygame.draw.rect(surface, LINE, square, 1)

    caption_font = pygame.font.SysFont(None, 26)
    for i, line in enumerate(caption.split("\n")[:2]):
        text = caption_font.render(line, True, CAPTION)
        surface.blit(text, (12, pixels + 6 + 24 * i))


class QuitGame(Exception):
    """The window was closed (or Escape pressed) while waiting for a move."""


class BoardView:
    """A window showing a game. Call show() after each move, then
    finish() at the end; it handles the window's events meanwhile."""

    def __init__(self, size, cell=None, title="Go"):
        pygame.init()
        self.cell = cell or (90 if size <= 5 else 72 if size <= 9 else 40)
        _, pixels = board_geometry(size, self.cell)
        self.screen = pygame.display.set_mode((pixels, pixels + CAPTION_HEIGHT))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.closed = False
        self.size = size
        self.margin, _ = board_geometry(size, self.cell)

    def show(self, game, last_move=None, caption="", seconds=0.0, final=False):
        """Draw the position, then keep the window responsive for seconds
        (longer while paused with space). Returns False once the window
        has been closed (or Escape pressed)."""
        draw_board(self.screen, game, self.cell, last_move, caption, final)
        pygame.display.flip()
        paused, waited = False, 0.0
        while not self.closed and (paused or waited < seconds):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    self.closed = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    paused = not paused
            waited += self.clock.tick(60) / 1000
        return not self.closed

    def point_at(self, position):
        """The point nearest the pixel position (x, y), or None if it's not
        close enough to any."""
        x, y = position
        column = round((x - self.margin) / self.cell)
        row = round((y - self.margin) / self.cell)
        if not (0 <= row < self.size and 0 <= column < self.size):
            return None
        centre = (self.margin + column * self.cell, self.margin + row * self.cell)
        if (x - centre[0]) ** 2 + (y - centre[1]) ** 2 > (0.48 * self.cell) ** 2:
            return None
        return row * self.size + column

    def ask_move(self, game, last_move=None):
        """Wait for the person to move for the player to move: a click on a
        point plays there, P passes. An illegal click says why and waits
        for another. Raises QuitGame if the window is closed (or Escape
        pressed) instead."""
        if self.closed:
            raise QuitGame
        colour = COLOUR_NAMES[game.to_play]
        prompt = (
            f"{colour} (you) to play: click a point, or P to pass\n"
            "Capture dead stones before passing: the board scores as it stands"
        )
        last_move = (
            last_move
            if last_move is not None
            else (game.moves[-1] if game.moves else None)
        )
        draw_board(self.screen, game, self.cell, last_move, prompt)
        pygame.display.flip()
        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                self.closed = True
                raise QuitGame
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                return game.pass_move
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                point = self.point_at(event.pos)
                if point is None:
                    continue
                reason = game.why_illegal(point)
                if reason is None:
                    return point
                retry = (
                    f"{game.name(point)} isn't allowed: {reason}\n"
                    f"{colour} (you) to play: click a point, or P to pass"
                )
                draw_board(self.screen, game, self.cell, last_move, retry)
                pygame.display.flip()

    def finish(self, game, caption):
        """Show the final position with territory until the window is
        closed or a key is pressed."""
        if self.closed:
            return
        draw_board(self.screen, game, self.cell, None, caption, final=True)
        pygame.display.flip()
        while True:
            event = pygame.event.wait()
            if event.type in (pygame.QUIT, pygame.KEYDOWN):
                break
        self.close()

    def close(self):
        pygame.quit()
