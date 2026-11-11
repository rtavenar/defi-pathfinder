"""Préparation des données du défi Pathfinder (à lancer UNE fois, par l'enseignant).

Récupère un sous-échantillon aléatoire de Pathfinder-X2 (Tylersuard/PathfinderX2),
en tire un sous-échantillon train / test et écrit :

    release/train_images.npz   -> à publier (GitHub Release)
    release/train_masks.npz    -> à publier
    release/test_images.npz    -> à publier
    release/SHA256SUMS.txt     -> à publier (le loader vérifie les fichiers)
    private/test_masks.npz     -> NE PAS publier (vérité terrain en clair)
    private/GT_KEY.txt         -> NE PAS publier (clé à mettre en secret GitHub)
    scoring/test_masks.enc     -> vérité terrain chiffrée, à committer
    docs/assets/sample_*.png   -> exemples affichés sur la page du défi

Usage :
    pip install pillow numpy cryptography
    python prepare_data.py --n-train 5000 --n-test 2000
    python prepare_data.py --describe     # diagnostic : contenu des archives
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image



# Pathfinder-X2 est un dépôt Hugging Face « à script » (PathfinderX2.py) que les
# versions récentes de `datasets` refusent d'exécuter, et son export Parquet est vide.
# Les données sont en fait deux archives zip sur S3 (images, masques ; même nom
# de fichier pour une image et son masque). Plutôt que de télécharger ces archives
# en entier (plusieurs Go), on lit leur index puis on ne récupère que les fichiers
# voulus, par requêtes HTTP partielles (en-tête Range).
S3_IMAGES = "https://pathfinder-x2.s3.us-west-1.amazonaws.com/Pathfinder-X2+images.zip"
S3_MASKS = "https://pathfinder-x2.s3.us-west-1.amazonaws.com/Pathfinder-X2+masks.zip"
MASK_PREFIXES = ("seg_", "mask_", "annotation_", "annot_")


def _http_range(url, start, end, retries=4):
    """Octets [start, end] (inclus) de l'URL."""
    import time
    import urllib.request

    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise RuntimeError(f"Le serveur ignore l'en-tête Range ({r.status})")
                return r.read()
        except RuntimeError:
            raise
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


class HTTPRangeFile:
    """Fichier distant en lecture seule, suffisant pour zipfile.ZipFile."""

    def __init__(self, url, block=4 << 20):
        import urllib.request

        self.url, self.block, self.pos = url, block, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self._cache_start, self._cache = -1, b""

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            bstart = (self.pos // self.block) * self.block
            if bstart != self._cache_start:
                self._cache = _http_range(self.url, bstart, min(bstart + self.block, self.size) - 1)
                self._cache_start = bstart
            chunk = self._cache[self.pos - bstart:self.pos - bstart + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)


def _zip_index(url):
    import zipfile

    f = HTTPRangeFile(url)
    print(f"  index de {url.rsplit('/', 1)[-1]} ({f.size / 1e9:.1f} Go)")
    z = zipfile.ZipFile(f)
    keep = [zi for zi in z.infolist() if zi.filename.lower().endswith(".png")
            and "__MACOSX" not in zi.filename and not Path(zi.filename).name.startswith("._")]
    return f.size, keep


def _fetch_member(url, size, zi):
    """Contenu décompressé d'un membre de zip, en une requête Range."""
    import struct
    import zlib

    start = zi.header_offset
    end = min(start + 30 + len(zi.filename.encode()) + 1024 + zi.compress_size, size) - 1
    data = _http_range(url, start, end)
    if data[:4] != b"PK\x03\x04":
        raise RuntimeError(f"En-tête local invalide pour {zi.filename}")
    n_name, n_extra = struct.unpack("<HH", data[26:30])
    s = 30 + n_name + n_extra
    if len(data) < s + zi.compress_size:
        data += _http_range(url, start + len(data), start + s + zi.compress_size - 1)
    comp = data[s:s + zi.compress_size]
    if zi.compress_type == 0:
        return comp
    if zi.compress_type == 8:
        return zlib.decompress(comp, -15)
    raise RuntimeError(f"Compression {zi.compress_type} non gérée ({zi.filename})")


def _stem(name, strip_prefix=False):
    stem = Path(name).stem
    if strip_prefix:
        for p in MASK_PREFIXES:
            if stem.startswith(p):
                return stem[len(p):]
    return stem


def iter_s3_samples(n, seed=0, skip=0, workers=16, images_url=S3_IMAGES, masks_url=S3_MASKS):
    """Tire n paires (image, masque) au hasard dans les archives et les itère."""
    from concurrent.futures import ThreadPoolExecutor

    size_i, imgs = _zip_index(images_url)
    size_m, masks = _zip_index(masks_url)
    by_stem = {_stem(z.filename): z for z in masks}
    if not set(by_stem) & {_stem(z.filename) for z in imgs[:1000]}:
        by_stem = {_stem(z.filename, strip_prefix=True): z for z in masks}
    pairs = sorted(((zi, by_stem[_stem(zi.filename)]) for zi in imgs
                    if _stem(zi.filename) in by_stem), key=lambda p: p[0].filename)
    print(f"  {len(imgs)} images, {len(masks)} masques, {len(pairs)} paires appariées")
    if len(pairs) < skip + n:
        raise RuntimeError(f"Seulement {len(pairs)} paires disponibles")
    chosen = np.random.default_rng(seed + 12345).permutation(len(pairs))[skip:skip + n]
    print(f"  exemples : {pairs[chosen[0]][0].filename} <-> {pairs[chosen[0]][1].filename}")

    def fetch(k):
        zi, zm = pairs[k]
        return (Image.open(io.BytesIO(_fetch_member(images_url, size_i, zi))),
                Image.open(io.BytesIO(_fetch_member(masks_url, size_m, zm))))

    with ThreadPoolExecutor(workers) as pool:
        yield from pool.map(fetch, chosen)


def describe_archives(images_url=S3_IMAGES, masks_url=S3_MASKS):
    """Affiche le début de l'index des deux archives (diagnostic)."""
    for url in (images_url, masks_url):
        size, infos = _zip_index(url)
        print(f"  {len(infos)} fichiers PNG ; premiers :")
        for zi in infos[:5]:
            print(f"    {zi.filename}  ({zi.file_size} octets, compression {zi.compress_type})")


def to_arrays(img, ann, size):
    img = img.convert("L")
    ann = ann.convert("L")
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
        ann = ann.resize((size, size), Image.NEAREST)
    x = np.asarray(img, dtype=np.uint8)
    a = np.asarray(ann, dtype=np.uint8)
    # Les fichiers servis peuvent être des JPEG : on binarise le masque.
    y = (a > 127).astype(np.uint8)
    return x, y, a


def pack_masks(y):
    """(N, H, W) {0,1} -> (N, H, W/8) uint8, 8x plus léger."""
    return np.packbits(y.astype(bool), axis=-1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_sample_png(x, y, path_prefix):
    Image.fromarray(x).save(f"{path_prefix}_image.png", optimize=True)
    Image.fromarray((y * 255).astype(np.uint8)).save(f"{path_prefix}_mask.png", optimize=True)


def build(samples, n_train, n_test, size, root, seed=0, n_examples=4):
    root = Path(root)
    (root / "release").mkdir(parents=True, exist_ok=True)
    (root / "private").mkdir(parents=True, exist_ok=True)
    (root / "scoring").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "assets").mkdir(parents=True, exist_ok=True)

    n_total = n_train + n_test
    X, Y = None, None
    raw_values = set()
    i = -1
    for i, (img, ann) in enumerate(samples):
        if i >= n_total:
            break
        x, y, a = to_arrays(img, ann, size)
        if X is None:
            X = np.empty((n_total,) + x.shape, dtype=np.uint8)
            Y = np.empty((n_total,) + y.shape, dtype=np.uint8)
        X[i], Y[i] = x, y
        if i < 50:
            raw_values |= set(np.unique(a).tolist())
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_total} exemples lus")
    else:
        i += 1
    if X is None or i < n_total:
        raise RuntimeError(f"Seulement {0 if X is None else i} exemples disponibles")

    rng = np.random.default_rng(seed)
    # Rotation / symétrie aléatoire de chaque exemple (la tâche y est invariante) :
    # les images ne sont plus identiques octet à octet à celles de Hugging Face,
    # ce qui complique une recherche naïve des masques de test dans le jeu d'origine.
    for j in range(n_total):
        k, flip = rng.integers(4), rng.integers(2)
        X[j], Y[j] = np.rot90(X[j], k).copy(), np.rot90(Y[j], k).copy()
        if flip:
            X[j], Y[j] = X[j][:, ::-1].copy(), Y[j][:, ::-1].copy()
    perm = rng.permutation(n_total)
    tr, te = perm[:n_train], perm[n_train:]
    X_tr, Y_tr, X_te, Y_te = X[tr], Y[tr], X[te], Y[te]

    # Découpage public / privé du test (style Kaggle) : 50 / 50
    test_perm = rng.permutation(n_test)
    public_idx = np.sort(test_perm[: n_test // 2])
    private_idx = np.sort(test_perm[n_test // 2:])

    rel = root / "release"
    np.savez_compressed(rel / "train_images.npz", images=X_tr)
    np.savez_compressed(rel / "train_masks.npz", masks=pack_masks(Y_tr), shape=np.array(Y_tr.shape))
    np.savez_compressed(rel / "test_images.npz", images=X_te)
    with open(rel / "SHA256SUMS.txt", "w") as f:
        for name in ["train_images.npz", "train_masks.npz", "test_images.npz"]:
            f.write(f"{sha256(rel / name)}  {name}\n")

    np.savez_compressed(root / "private" / "test_masks.npz", masks=Y_te,
                        public_idx=public_idx, private_idx=private_idx)

    # Vérité terrain chiffrée pour la GitHub Action
    from cryptography.fernet import Fernet

    key_file = root / "private" / "GT_KEY.txt"
    if key_file.exists():
        key = key_file.read_text().strip().encode()
    else:
        key = Fernet.generate_key()
        key_file.write_text(key.decode() + "\n")
    buf = io.BytesIO()
    np.savez_compressed(buf, masks=pack_masks(Y_te), shape=np.array(Y_te.shape),
                        public_idx=public_idx, private_idx=private_idx)
    (root / "scoring" / "test_masks.enc").write_bytes(Fernet(key).encrypt(buf.getvalue()))

    for k in range(n_examples):
        save_sample_png(X_tr[k], Y_tr[k], root / "docs" / "assets" / f"sample_{k}")

    stats = {
        "n_train": int(n_train), "n_test": int(n_test),
        "image_shape": list(X_tr.shape[1:]),
        "mask_positive_fraction": float(Y.mean()),
        "raw_annotation_values_first50": sorted(int(v) for v in raw_values)[:20],
        "n_raw_annotation_values_first50": len(raw_values),
        "files_MB": {p.name: round(p.stat().st_size / 1e6, 1) for p in rel.glob("*.npz")},
    }
    (root / "private" / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--size", type=int, default=None,
                   help="Redimensionner (ex. 256). Par défaut : taille native (512).")
    p.add_argument("--skip", type=int, default=0,
                   help="Décaler le tirage (pour obtenir d'autres exemples)")
    p.add_argument("--workers", type=int, default=16, help="Téléchargements parallèles")
    p.add_argument("--images-url", default=S3_IMAGES)
    p.add_argument("--masks-url", default=S3_MASKS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default=".")
    p.add_argument("--describe", action="store_true",
                   help="Afficher le début de l'index des archives et quitter")
    args = p.parse_args()
    if args.describe:
        describe_archives(args.images_url, args.masks_url)
        return
    samples = iter_s3_samples(args.n_train + args.n_test, seed=args.seed, skip=args.skip,
                              workers=args.workers, images_url=args.images_url,
                              masks_url=args.masks_url)
    build(samples, args.n_train, args.n_test, args.size, args.root, seed=args.seed)
    print("\nÀ vérifier : ouvrir docs/assets/sample_*_mask.png et private/stats.json"
          " (le masque doit bien être binaire).")


if __name__ == "__main__":
    main()
