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

from game import BLACK, EMPTY, other


def encode(game, colour=False):
    """The board as the network sees it: an array of shape (3, size, size)
    with 1s marking the mover's stones, the opponent's stones, and the
    empty points. With colour, a fourth plane says who's to move: all 1s
    for Black, all 0s for White (with komi, the same position is worth
    more to White)."""
    board = np.array(game.board).reshape(game.size, game.size)
    mover = game.to_play
    planes = [board == mover, board == other(mover), board == EMPTY]
    if colour:
        planes.append(np.full_like(board, mover == BLACK, dtype=bool))
    return np.stack(planes).astype(np.float32)


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
    there are.

    With colour, it takes a fourth input plane saying who's to move (see
    encode). With value, it also has a value head (a critic): from the same
    convolutions, it averages each feature over the board, and a linear
    layer and tanh give V, the expected result for the player to move,
    from -1 (a sure loss) to +1 (a sure win). heads() returns both.
    """

    def __init__(self, size, channels=32, layers=3, value=False, colour=False):
        super().__init__()
        self.size, self.channels, self.layers = size, channels, layers
        self.has_value = value
        self.has_colour = colour
        convolutions = []
        width = 4 if colour else 3
        for _ in range(layers):
            convolutions += [nn.Conv2d(width, channels, 3, padding=1), nn.ReLU()]
            width = channels
        self.body = nn.Sequential(*convolutions)
        self.point_scores = nn.Conv2d(channels, 1, 1)
        self.pass_score = nn.Linear(channels, 1)
        self.value_score = nn.Linear(channels, 1) if value else None

    def heads(self, boards, detach_value=False):
        """Return (move scores, values): values has shape (batch,), or is
        None without a value head. With detach_value, the value head reads
        the shared features without its training changing them (so the
        critic can't disturb what the policy uses)."""
        features = self.body(boards)
        average = features.mean(dim=(2, 3))
        points = self.point_scores(features).flatten(1)
        scores = torch.cat([points, self.pass_score(average)], dim=1)
        if self.value_score is None:
            return scores, None
        if detach_value:
            average = average.detach()
        return scores, torch.tanh(self.value_score(average)).squeeze(1)

    def forward(self, boards):
        return self.heads(boards)[0]

    def settings(self):
        """What's needed to build the same network again."""
        return {
            "size": self.size,
            "channels": self.channels,
            "layers": self.layers,
            "value": self.has_value,
            "colour": self.has_colour,
        }


def upgraded(network, value=None, colour=None):
    """A copy of network gaining a value head (value) or a colour input
    plane (colour), choosing moves exactly as before: a new value head
    starts untrained, and the first convolution's weights for a new
    colour plane start at zero, so the plane has no effect until
    training gives it one."""
    settings = network.settings()
    if value is not None:
        settings["value"] = value
    if colour is not None:
        settings["colour"] = colour
    copy = PolicyNetwork(**settings)
    weights = dict(network.state_dict())
    first = "body.0.weight"
    old, new = weights[first], copy.state_dict()[first]
    if old.shape != new.shape:  # adding the colour plane
        grown = torch.zeros_like(new)
        grown[:, : old.shape[1]] = old
        weights[first] = grown
    copy.load_state_dict(weights, strict=False)
    return copy


def with_value_head(network):
    """A copy of network with a value head (see upgraded)."""
    return upgraded(network, value=True)


def saved_komi(path):
    """The komi a saved network was trained with, or None if not recorded."""
    return torch.load(path, weights_only=True).get("komi")


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
        board, mask = encode(game, self.network.has_colour), legal_mask(game)
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


def policy_gradient_loss(
    network, records, advantages, entropy_bonus=0.0, policy_weights=None
):
    """The REINFORCE loss for recorded moves: minus the mean over them of
    advantage * log pi(move | board), less entropy_bonus times the mean
    entropy of the policy (a reward for keeping its probabilities spread
    out, so it keeps exploring).

    records are (board, mask, move) tuples as NetworkPlayer records them,
    and advantages one number per record. policy_weights, if given, scale
    each record's term (importance weights for turned moves, see
    symmetry.py). Returns (loss, mean entropy).
    """
    boards = torch.from_numpy(np.stack([board for board, _, _ in records]))
    masks = torch.from_numpy(np.stack([mask for _, mask, _ in records]))
    chosen = torch.tensor([move for _, _, move in records])
    log_probs = torch.log_softmax(masked_logits(network(boards), masks), dim=1)
    taken = log_probs[torch.arange(len(records)), chosen]
    weights = torch.tensor(np.asarray(advantages), dtype=torch.float32)
    if policy_weights is not None:
        weights = weights * torch.tensor(
            np.asarray(policy_weights), dtype=torch.float32
        )
    entropy = -(log_probs.exp() * log_probs.clamp(min=-30)).sum(dim=1).mean()
    loss = -(weights * taken).mean() - entropy_bonus * entropy
    return loss, float(entropy.detach())


def actor_critic_loss(
    network,
    records,
    results,
    value_weight=0.5,
    entropy_bonus=0.0,
    detach_value=False,
    policy_weights=None,
):
    """The loss with a critic (the network's value head), for recorded
    moves and each one's result z (+1, -1 or 0 for the side that made it):

        advantage = z - V(board)    (V held fixed for the policy's part)
        loss = -mean(advantage * log pi(move | board))
               + value_weight * mean((V(board) - z)^2)
               - entropy_bonus * mean entropy

    With detach_value, the critic's loss trains only the value head, not
    the shared convolutions. policy_weights, if given, scale each record's
    policy term, not the critic's (importance weights for turned moves,
    see symmetry.py: the critic's regression is valid on any board).
    Returns (loss, mean entropy, value loss, how often V's sign matched a
    decided result).
    """
    boards = torch.from_numpy(np.stack([board for board, _, _ in records]))
    masks = torch.from_numpy(np.stack([mask for _, mask, _ in records]))
    chosen = torch.tensor([move for _, _, move in records])
    targets = torch.tensor(np.asarray(results), dtype=torch.float32)
    scores, values = network.heads(boards, detach_value)
    log_probs = torch.log_softmax(masked_logits(scores, masks), dim=1)
    taken = log_probs[torch.arange(len(records)), chosen]
    advantages = targets - values.detach()
    if policy_weights is not None:
        advantages = advantages * torch.tensor(
            np.asarray(policy_weights), dtype=torch.float32
        )
    entropy = -(log_probs.exp() * log_probs.clamp(min=-30)).sum(dim=1).mean()
    value_loss = ((values - targets) ** 2).mean()
    loss = (
        -(advantages * taken).mean()
        + value_weight * value_loss
        - entropy_bonus * entropy
    )
    decided = targets != 0
    correct = (torch.sign(values) == targets)[decided].float().mean()
    return loss, float(entropy.detach()), float(value_loss.detach()), float(correct)
