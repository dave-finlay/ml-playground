"""A neural network that chooses moves in Go, and a player that uses it.

The network sees the board from the point of view of the player to
move, as three layers of size x size: that player's stones, the
opponent's stones, and the empty points. So one network plays either
colour. It outputs a score (a logit) for each point and for passing;
illegal moves are masked out before the softmax turns scores into
probabilities.

It's a small convolutional network: a few 3 x 3 convolutions, each
looking at a point and its neighbours, so stacked layers see further
across the board; then a 1 x 1 convolution scores each point, and the
average of the last layer's features scores passing.
"""

import numpy as np
import torch
from torch import nn

from game import EMPTY, other


def encode(game):
    """The board as the network sees it: an array of shape (3, size, size)
    with 1s marking the mover's stones, the opponent's stones, and the
    empty points."""
    board = np.array(game.board).reshape(game.size, game.size)
    mover = game.to_play
    return np.stack([board == mover, board == other(mover), board == EMPTY]).astype(
        np.float32
    )


def legal_mask(game):
    """A boolean array over every move (each point, then pass): True where
    the move is legal."""
    mask = np.zeros(game.pass_move + 1, dtype=bool)
    mask[game.legal_moves()] = True
    return mask


def masked_logits(logits, mask):
    """logits with illegal moves pushed to minus infinity (in effect), so
    the softmax gives them probability 0."""
    return logits.masked_fill(~mask, -1e9)


class PolicyNetwork(nn.Module):
    """Scores every move given encoded boards: input shape (batch, 3,
    size, size), output (batch, size * size + 1), the last column for
    passing. channels is the width of each convolution, layers how many
    there are."""

    def __init__(self, size, channels=32, layers=3):
        super().__init__()
        self.size, self.channels, self.layers = size, channels, layers
        convolutions = []
        width = 3
        for _ in range(layers):
            convolutions += [nn.Conv2d(width, channels, 3, padding=1), nn.ReLU()]
            width = channels
        self.body = nn.Sequential(*convolutions)
        self.point_scores = nn.Conv2d(channels, 1, 1)
        self.pass_score = nn.Linear(channels, 1)

    def forward(self, boards):
        features = self.body(boards)
        points = self.point_scores(features).flatten(1)
        passing = self.pass_score(features.mean(dim=(2, 3)))
        return torch.cat([points, passing], dim=1)

    def settings(self):
        """What's needed to build the same network again."""
        return {"size": self.size, "channels": self.channels, "layers": self.layers}


def save_network(path, network, extra=None):
    """Save the network's weights and settings (and anything in extra)."""
    torch.save(
        {
            "settings": network.settings(),
            "weights": network.state_dict(),
            **(extra or {}),
        },
        path,
    )


def load_network(path):
    """A network saved by save_network."""
    saved = torch.load(path, weights_only=True)
    network = PolicyNetwork(**saved["settings"])
    network.load_state_dict(saved["weights"])
    network.eval()
    return network


class NetworkPlayer:
    """Plays with a PolicyNetwork. With sample, it picks each move at
    random by the network's probabilities (for training); otherwise it
    takes the likeliest legal move. If record is a list, each move's
    (board, legal mask, move) is appended to it."""

    def __init__(self, network, sample=False, rng=None, record=None):
        self.network = network
        self.sample = sample
        self.rng = rng if rng is not None else np.random.default_rng()
        self.record = record

    def __call__(self, game):
        board, mask = encode(game), legal_mask(game)
        with torch.no_grad():
            logits = self.network(torch.from_numpy(board)[None])[0]
            logits = masked_logits(logits, torch.from_numpy(mask))
        if self.sample:
            probs = torch.softmax(logits, dim=0).numpy().astype(np.float64)
            move = int(self.rng.choice(len(probs), p=probs / probs.sum()))
        else:
            move = int(torch.argmax(logits))
        if self.record is not None:
            self.record.append((board, mask, move))
        return move


def policy_gradient_loss(network, records, advantages, entropy_bonus=0.0):
    """The REINFORCE loss for recorded moves: minus the mean over them of
    advantage * log pi(move | board), less entropy_bonus times the mean
    entropy of the policy (a reward for keeping its probabilities spread
    out, so it keeps exploring).

    records are (board, mask, move) tuples as NetworkPlayer records them,
    and advantages one number per record. Returns (loss, mean entropy).
    """
    boards = torch.from_numpy(np.stack([board for board, _, _ in records]))
    masks = torch.from_numpy(np.stack([mask for _, mask, _ in records]))
    chosen = torch.tensor([move for _, _, move in records])
    log_probs = torch.log_softmax(masked_logits(network(boards), masks), dim=1)
    taken = log_probs[torch.arange(len(records)), chosen]
    weights = torch.tensor(np.asarray(advantages), dtype=torch.float32)
    entropy = -(log_probs.exp() * log_probs.clamp(min=-30)).sum(dim=1).mean()
    loss = -(weights * taken).mean() - entropy_bonus * entropy
    return loss, float(entropy.detach())
