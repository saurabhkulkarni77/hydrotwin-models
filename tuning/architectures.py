"""Tunable supersets of the repo architectures.

LSTMNetX / TransformerX mirror train_lstm_normal.LSTMNet and
optimize_tr_gcn.TransformerFlex exactly, adding the knobs those classes do
not expose (LSTM depth, Transformer dropout). At the incumbent settings
(layers=1 / dropout=0.0) they are structurally identical to the repo
classes and their state_dict keys match, so a tuned checkpoint stays
loadable by the repo classes whenever the extra knobs sit at defaults;
the winning config JSON always records the knob values either way.

TFTX mirrors train_sota8.TFTLite (incumbent d=48, heads=4) and PatchTSTX
mirrors train_new_arch.PatchTST (incumbent d=64, patch=7, stride=7,
layers=2, heads=4, window=29) under the same rule: identical module tree
and state_dict keys at incumbent settings, extra knobs (LSTM depth /
patch length / dropout) recorded in the winning config. Dropout is
applied functionally so it never changes the key set. PILSTMv1 is
LSTMNet with a physics loss (train_pilstm.py) — architecturally it IS
LSTMNetX; the physics lives in trainable.py, not here. GP is an exact
GP regression (no torch module); see trainable.fit_gp.
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


class GRNX(nn.Module):
    """train_sota8.GRN, unchanged (fc1/fc2/gate/norm — same keys)."""

    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = nn.functional.elu(self.fc1(x))
        h = self.fc2(h)
        return self.norm(x + torch.sigmoid(self.gate(x)) * h)


class TFTX(nn.Module):
    """TFTLite with LSTM depth + dropout knobs.

    At lstm_layers=1 / dropout=0.0 the module tree and every state_dict
    key are identical to train_sota8.TFTLite(d, heads): emb -> LSTM ->
    GRN -> MultiheadAttention -> LayerNorm -> head. Dropout is the
    attention module's own parameter plus one functional drop after the
    GRN, so no key ever changes with the dropout value.
    """

    def __init__(self, d=48, heads=4, lstm_layers=1, dropout=0.0,
                 n_feat=9, leads=7):
        super().__init__()
        self._p = float(dropout)
        self.emb = nn.Linear(n_feat, d)
        self.lstm = nn.LSTM(d, d, num_layers=int(lstm_layers),
                            batch_first=True)
        self.grn = GRNX(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True,
                                         dropout=dropout)
        self.n = nn.LayerNorm(d)
        self.head = nn.Linear(d, leads)

    def forward(self, x):
        h = self.emb(x)
        h, _ = self.lstm(h)
        h = self.grn(h)
        h = nn.functional.dropout(h, p=self._p, training=self.training)
        a, _ = self.att(h, h, h)
        return self.head(self.n(h + a)[:, -1])


class PatchTSTX(nn.Module):
    """PatchTST with patch length / depth / dropout knobs.

    At patch=7, stride=patch, layers=2, dropout=0.0 and window=29 the
    module tree and state_dict keys are identical to
    train_new_arch.PatchTST(d, patch, stride, layers, heads) — including
    the head width d*npatch, which follows from the window.
    """

    def __init__(self, d=64, patch=7, layers=2, heads=4, window=29,
                 dropout=0.0, n_feat=9, leads=7):
        super().__init__()
        self.patch = int(patch)
        self.stride = int(patch)
        self.npatch = (int(window) - self.patch) // self.stride + 1
        self._p = float(dropout)
        self.emb = nn.Linear(self.patch * n_feat, d)
        self.pos = nn.Parameter(torch.zeros(1, self.npatch, d))
        self.blocks = nn.ModuleList()
        for _ in range(int(layers)):
            self.blocks.append(nn.ModuleDict({
                "att": nn.MultiheadAttention(d, heads, batch_first=True,
                                             dropout=dropout),
                "n1": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                   nn.Linear(2 * d, d)),
                "n2": nn.LayerNorm(d)}))
        self.head = nn.Linear(d * self.npatch, leads)

    def forward(self, x):
        B = x.shape[0]
        p = x.unfold(1, self.patch, self.stride).permute(0, 1, 3, 2) \
            .reshape(B, self.npatch, -1)
        h = self.emb(p) + self.pos
        for b in self.blocks:
            a, _ = b["att"](h, h, h)
            h = b["n1"](h + a)
            h = b["n2"](h + nn.functional.dropout(
                b["ff"](h), p=self._p, training=self.training))
        return self.head(h.reshape(B, -1))


def build_model(family, params, window):
    """params: flat dict from the search space (see spaces.py)."""
    if family == "LSTM":
        return LSTMNetX(hidden=int(params["hidden"]),
                        dropout=float(params["dropout"]),
                        num_layers=int(params["layers"]))
    if family == "PILSTMv1":
        # Same architecture as production PILSTMv1 (train_pilstm.py's
        # LSTMNet); the physics is in the training loss (trainable.py).
        return LSTMNetX(hidden=int(params["hidden"]),
                        dropout=float(params["dropout"]),
                        num_layers=int(params["layers"]))
    if family == "Transformer":
        d = int(params["d"]); heads = int(params["heads"])
        return TransformerX(d=d, layers=int(params["layers"]), heads=heads,
                            T=int(window), dropout=float(params["dropout"]))
    if family == "TFT":
        d = int(params["d"]); heads = int(params["heads"])
        return TFTX(d=d, heads=heads, lstm_layers=int(params["layers"]),
                    dropout=float(params["dropout"]))
    if family == "PatchTST":
        d = int(params["d"]); heads = int(params["heads"])
        return PatchTSTX(d=d, patch=int(params["patch"]),
                         layers=int(params["layers"]), heads=heads,
                         window=int(window),
                         dropout=float(params["dropout"]))
    raise ValueError(f"unknown family {family}")
