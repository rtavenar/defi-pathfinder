"""Classement final sur le test PRIVÉ (à lancer en local par l'enseignant, après la date limite).

Pour chaque équipe, on retient sa meilleure soumission sur le test public
(option --select last : sa dernière soumission), et on l'évalue sur la moitié
privée du test, jamais vue par le leaderboard.

    python scoring/final_ranking.py                 # lit private/test_masks.npz
    python scoring/final_ranking.py --select last
"""

import argparse
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pathfinder_data import dice_score, load_submission  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--select", choices=["best", "last"], default="best")
    p.add_argument("--gt", default=str(ROOT / "private" / "test_masks.npz"))
    p.add_argument("--out", default=str(ROOT / "private" / "final_ranking.csv"))
    args = p.parse_args()

    gt = np.load(args.gt)
    masks, priv = gt["masks"], gt["private_idx"]
    subs = json.loads((ROOT / "docs" / "submissions.json").read_text())

    chosen = {}
    for s in subs:
        t = s["team"]
        if args.select == "last" or t not in chosen or s["public_score"] > chosen[t]["public_score"]:
            chosen[t] = s

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for team, s in chosen.items():
            path = Path(tmp) / f"{s['issue']}.zip"
            urllib.request.urlretrieve(s["url"], path)
            pred = load_submission(path)
            rows.append((team, s["public_score"], dice_score(pred[priv], masks[priv]), s["issue"]))
            print(f"{team:40s} public={s['public_score']:.4f} privé={rows[-1][2]:.4f}")

    rows.sort(key=lambda r: -r[2])
    with open(args.out, "w") as f:
        f.write("rang,equipe,score_public,score_prive,issue\n")
        for i, (team, pub, prv, issue) in enumerate(rows, 1):
            f.write(f'{i},"{team}",{pub:.5f},{prv:.5f},{issue}\n')
    print(f"\nClassement écrit dans {args.out}")


if __name__ == "__main__":
    main()
