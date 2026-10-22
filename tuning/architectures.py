"""Tunable supersets of the repo architectures.

LSTMNetX / TransformerX mirror train_lstm_normal.LSTMNet and
optimize_tr_gcn.TransformerFlex exactly, adding the knobs those classes do
not expose (LSTM depth, Transformer dropout). At the incumbent settings
(layers=1 / dropout=0.0) they are structurally identical to the repo
classes and their state_dict keys match, so a tuned checkpoint stays
loadable by the repo classes whenever the extra knobs sit at defaults;
the winning config JSON always records the knob values either way.
"""
import torch
import torch.nn as nn


class LSTMNetX(nn.Module):
    """LSTMNet with optional stacking. num_layers=1 == LSTMNet exactly."""

    def __init__(self, hidden=16, dropout=0.0, num_layers=1, n_feat=9, leads=7):
        super().__init__()
        # nn.LSTM applies `dropout` only between stacked layers; with
        # num_layers=1 the repo applies dropout after the RNN instead.
        self.rnn = nn.LSTM(n_feat, hidden, num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, leads)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(self.drop(h[:, -1]))


class TransformerX(nn.Module):
    """TransformerFlex with dropout on attention + feed-forward output.

    Dropout is applied functionally (F.dropout) so the module tree — and
    therefore every state_dict key — is identical to TransformerFlex at
    ANY dropout value; tuned checkpoints always load into the repo class.
    """

    def __init__(self, d=16, layers=1, heads=2, T=29, dropout=0.0,
                 n_feat=9, leads=7):
        super().__init__()
        self._p = float(dropout)
        self.emb = nn.Linear(n_feat, d)
        self.pos = nn.Parameter(torch.zeros(1, T, d))
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "att": nn.MultiheadAttention(d, heads, batch_first=True,
                                             dropout=dropout),
                "n1": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                   nn.Linear(2 * d, d)),
                "n2": nn.LayerNorm(d)}))
        self.head = nn.Linear(d, leads)

    def forward(self, x):
        h = self.emb(x) + self.pos
        for b in self.blocks:
            a, _ = b["att"](h, h, h)
            h = b["n1"](h + a)
            h = b["n2"](h + nn.functional.dropout(b["ff"](h), p=self._p,
                                                  training=self.training))
        return self.head(h[:, -1])


def build_model(family, params, window):
    """params: flat dict from the search space (see spaces.py)."""
    if family == "LSTM":
        return LSTMNetX(hidden=int(params["hidden"]),
                        dropout=float(params["dropout"]),
                        num_layers=int(params["layers"]))
    if family == "Transformer":
        d = int(params["d"]); heads = int(params["heads"])
        return TransformerX(d=d, layers=int(params["layers"]), heads=heads,
                            T=int(window), dropout=float(params["dropout"]))
    raise ValueError(f"unknown family {family}")
