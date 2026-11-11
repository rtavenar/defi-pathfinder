"""Données du défi Pathfinder : téléchargement, chargement, soumission.

    from pathfinder_data import load_train, load_test, save_submission, dice_score

    X, Y = load_train()      # X : (5000, 512, 512) uint8 ; Y : (5000, 512, 512) uint8 dans {0, 1}
    X_test = load_test()     # (2000, 512, 512) uint8
    ...
    save_submission(pred)    # pred : (2000, 512, 512), binaire ou probabilités -> submission.zip

Ne dépend que de numpy (pas de torch, pas de Hugging Face).
"""

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

BASE_URL = "https://github.com/rtavenar/defi-pathfinder/releases/download/data-v1/"
FILES = ["train_images.npz", "train_masks.npz", "test_images.npz"]
DEFAULT_DIR = Path("data")


def _download(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")

    last = [-1]

    def hook(n_blocks, block_size, total):
        if total > 0:
            done = min(n_blocks * block_size, total)
            pct = int(100 * done / total)
            if pct // 5 != last[0] // 5:
                last[0] = pct
                sys.stdout.write(f"\r  {dest.name} : {done / 1e6:6.1f} / {total / 1e6:.1f} Mo ({pct} %)")
                sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    print()
    tmp.rename(dest)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(data_dir=DEFAULT_DIR, base_url=BASE_URL):
    """Télécharge les fichiers du défi dans data_dir (ne refait rien s'ils sont déjà là)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    sums_path = data_dir / "SHA256SUMS.txt"
    if not sums_path.exists():
        _download(base_url + "SHA256SUMS.txt", sums_path)
    expected = dict(line.split()[::-1] for line in sums_path.read_text().splitlines() if line.strip())
    for name in FILES:
        path = data_dir / name
        if path.exists() and _sha256(path) == expected[name]:
            continue
        print(f"Téléchargement de {name}")
        _download(base_url + name, path)
        if _sha256(path) != expected[name]:
            path.unlink()
            raise IOError(f"{name} corrompu, relancez download()")
    return data_dir


def unpack_masks(packed, shape):
    return np.unpackbits(packed, axis=-1, count=int(shape[-1])).astype(np.uint8)


def load_train(data_dir=DEFAULT_DIR):
    """Renvoie (X, Y) : images d'entrée et masques cibles, en uint8."""
    data_dir = download(data_dir)
    X = np.load(data_dir / "train_images.npz")["images"]
    m = np.load(data_dir / "train_masks.npz")
    Y = unpack_masks(m["masks"], m["shape"])
    return X, Y


def load_test(data_dir=DEFAULT_DIR):
    """Renvoie X_test : images d'entrée du jeu de test (masques non fournis)."""
    data_dir = download(data_dir)
    return np.load(data_dir / "test_images.npz")["images"]


def dice_score(pred, target):
    """Dice moyen par image (métrique officielle). pred et target binaires, (N, H, W).

    Pour une image où prédiction et cible sont toutes deux vides, le Dice vaut 1.
    """
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    if pred.shape != target.shape:
        raise ValueError(f"Formes différentes : {pred.shape} vs {target.shape}")
    if pred.ndim == 2:
        pred, target = pred[None], target[None]
    inter = (pred & target).sum(axis=(1, 2))
    denom = pred.sum(axis=(1, 2)) + target.sum(axis=(1, 2))
    dice = np.where(denom == 0, 1.0, 2 * inter / np.maximum(denom, 1))
    return float(dice.mean())


def save_submission(pred, path="submission.zip", threshold=0.5, expected_shape=None):
    """Écrit une soumission. pred : (N_test, H, W), masques binaires ou probabilités.

    Les valeurs > threshold sont considérées comme appartenant au chemin.
    """
    pred = np.asarray(pred)
    if pred.dtype != bool:
        pred = pred > threshold
    if pred.ndim != 3:
        raise ValueError(f"pred doit être de forme (N, H, W), reçu {pred.shape}")
    if expected_shape is not None and tuple(pred.shape) != tuple(expected_shape):
        raise ValueError(f"Forme attendue {expected_shape}, reçu {pred.shape}")
    buf = io.BytesIO()
    np.savez(buf, masks=np.packbits(pred, axis=-1), shape=np.array(pred.shape))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("submission.npz", buf.getvalue())
    size = Path(path).stat().st_size / 1e6
    print(f"Soumission écrite dans {path} ({size:.1f} Mo)")
    if size > 25:
        print("Attention : GitHub refuse les pièces jointes de plus de 25 Mo.")
    return path


def load_submission(path):
    """Relit un fichier de soumission -> (N, H, W) uint8. Utilisé par le scoring."""
    with zipfile.ZipFile(path) as z:
        if "submission.npz" not in z.namelist():
            raise ValueError("submission.npz absent de l'archive")
        info = z.getinfo("submission.npz")
        if info.file_size > 200e6:
            raise ValueError("Archive trop volumineuse une fois décompressée")
        data = np.load(io.BytesIO(z.read("submission.npz")), allow_pickle=False)
        return unpack_masks(data["masks"], data["shape"])
