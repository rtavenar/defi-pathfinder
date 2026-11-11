# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Défi Pathfinder — notebook de démarrage
#
# Ce notebook télécharge les données, montre quelques exemples, calcule une
# baseline naïve, entraîne un petit U-Net et produit un fichier de soumission.
#
# Placez `pathfinder_data.py` dans le même dossier que ce notebook.
# Sur Colab : `!wget https://raw.githubusercontent.com/rtavenar/defi-pathfinder/main/pathfinder_data.py`

# %%
import matplotlib.pyplot as plt
import numpy as np

from pathfinder_data import dice_score, load_test, load_train, save_submission

X, Y = load_train()
X_test = load_test()
print("train :", X.shape, Y.shape, "| test :", X_test.shape)
print("fraction de pixels positifs dans les masques :", Y.mean().round(4))

# %% [markdown]
# ## Visualiser quelques exemples
# En haut l'image d'entrée, en bas le masque à prédire.

# %%
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for k in range(4):
    axes[0, k].imshow(X[k], cmap="gray")
    axes[1, k].imshow(Y[k], cmap="gray")
    for ax in axes[:, k]:
        ax.set_axis_off()
axes[0, 0].set_title("entrée")
axes[1, 0].set_title("cible")
plt.tight_layout()

# %% [markdown]
# ## Découpage train / validation
# Le jeu de test n'a pas de masques : on garde une partie du train pour
# estimer le score localement avant de soumettre.

# %%
rng = np.random.default_rng(0)
perm = rng.permutation(len(X))
n_val = 500
idx_val, idx_tr = perm[:n_val], perm[n_val:]

# %% [markdown]
# ## Baseline 1 : seuillage de l'image
# On prédit « chemin » pour tout pixel allumé. Ce score est un plancher :
# il prédit *toutes* les courbes, y compris les distractrices.

# %%
def threshold_baseline(images, thr=127):
    flip = images.mean() > 127  # au cas où le fond serait clair
    return (images < thr) if flip else (images > thr)


print("Dice (validation), seuillage :", round(dice_score(threshold_baseline(X[idx_val]), Y[idx_val]), 4))

# %% [markdown]
# ## Baseline 2 : un petit U-Net (PyTorch)
# Un GPU est vivement conseillé (Colab, Kaggle notebooks...).
# Les images sont converties en float **par batch** pour limiter la mémoire.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device :", device)


class Block(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, widths=(16, 32, 64, 128)):
        super().__init__()
        self.down = nn.ModuleList()
        c = 1
        for w in widths:
            self.down.append(Block(c, w))
            c = w
        self.up = nn.ModuleList()
        self.upconv = nn.ModuleList()
        for w in reversed(widths[:-1]):
            self.upconv.append(nn.ConvTranspose2d(c, w, 2, stride=2))
            self.up.append(Block(2 * w, w))
            c = w
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for upconv, block in zip(self.upconv, self.up):
            x = upconv(x)
            x = block(torch.cat([x, skips.pop()], dim=1))
        return self.head(x)  # logits


def to_tensor(batch):
    return torch.from_numpy(batch).float().div_(255.0).unsqueeze(1).to(device)


def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


@torch.no_grad()
def predict(model, images, batch_size=16):
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        out.append(torch.sigmoid(model(to_tensor(images[i:i + batch_size]))).squeeze(1).cpu().numpy())
    return np.concatenate(out)


# %%
model = UNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epochs, batch_size = 5, 8

for epoch in range(n_epochs):
    model.train()
    order = rng.permutation(idx_tr)
    losses = []
    for i in range(0, len(order), batch_size):
        b = np.sort(order[i:i + batch_size])
        xb, yb = to_tensor(X[b]), torch.from_numpy(Y[b]).float().unsqueeze(1).to(device)
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb) + soft_dice_loss(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    val_dice = dice_score(predict(model, X[idx_val]) > 0.5, Y[idx_val])
    print(f"époque {epoch + 1} : loss={np.mean(losses):.4f}  Dice val={val_dice:.4f}")

# %% [markdown]
# ## Prédire sur le test et soumettre
# `save_submission` écrit `submission.zip`. Pour soumettre : ouvrez une issue
# « Soumission » sur le dépôt du défi et glissez-déposez ce fichier.

# %%
pred_test = predict(model, X_test)
save_submission(pred_test, "submission.zip", expected_shape=X_test.shape)
