"""Évalue une soumission postée en issue GitHub (lancé par la GitHub Action).

Variables d'environnement :
    ISSUE_BODY, ISSUE_USER, ISSUE_NUMBER, GT_KEY
Écrit comment.md (réponse postée sur l'issue) et met à jour
docs/submissions.json et docs/leaderboard.json.
Code de sortie 0 si la soumission est acceptée, 1 sinon.
"""

import datetime as dt
import io
import json
import os
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pathfinder_data import dice_score, load_submission, unpack_masks  # noqa: E402

CONFIG = json.loads((ROOT / "scoring" / "config.json").read_text())
SUBMISSIONS = ROOT / "docs" / "submissions.json"
LEADERBOARD = ROOT / "docs" / "leaderboard.json"
ALLOWED_URL = re.compile(
    r"https://(github\.com/user-attachments/files/|github\.com/[\w.-]+/[\w.-]+/files/"
    r"|objects\.githubusercontent\.com/)[^\s)]+"
)


def parse_issue_form(body):
    """Découpe le corps d'une issue form ("### Titre\n\nvaleur") en dictionnaire."""
    fields, current = {}, None
    for line in body.splitlines():
        if line.startswith("### "):
            current = line[4:].strip()
            fields[current] = []
        elif current is not None:
            fields[current].append(line)
    return {k: "\n".join(v).strip() for k, v in fields.items()}


def load_ground_truth():
    from cryptography.fernet import Fernet

    blob = (ROOT / "scoring" / "test_masks.enc").read_bytes()
    data = np.load(io.BytesIO(Fernet(os.environ["GT_KEY"].strip().encode()).decrypt(blob)))
    return unpack_masks(data["masks"], data["shape"]), data["public_idx"], data["private_idx"]


def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def is_baseline(team):
    """Les noms d'équipe commençant par « Baseline » sont réservés à l'enseignant."""
    return team.strip().lower().startswith("baseline")


def build_leaderboard(subs):
    best = {}
    for s in subs:
        if s["team"] not in best or s["public_score"] > best[s["team"]]["public_score"]:
            best[s["team"]] = s
    rows = sorted(best.values(), key=lambda s: (-s["public_score"], s["date"]))
    counts = {}
    for s in subs:
        counts[s["team"]] = counts.get(s["team"], 0) + 1
    return {
        "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "metric": CONFIG["metric_name"],
        "rows": [
            {"team": r["team"], "score": r["public_score"], "date": r["date"],
             "issue": r["issue"], "description": r.get("description", ""),
             "n_submissions": counts[r["team"]],
             "baseline": is_baseline(r["team"])}
            for r in rows
        ],
    }


def fail(msg):
    Path("comment.md").write_text(f"❌ **Soumission refusée** : {msg}\n")
    print(msg)
    sys.exit(1)


def main():
    fields = parse_issue_form(os.environ.get("ISSUE_BODY", ""))
    user = os.environ.get("ISSUE_USER", "?")
    issue = int(os.environ.get("ISSUE_NUMBER", "0"))

    team = fields.get(CONFIG["field_team"], "").strip()
    if not team or team == "_No response_":
        fail("nom d'équipe manquant.")
    team = re.sub(r"\s+", " ", team)[:40]
    admins = {u.lower() for u in CONFIG.get("admins", [])}
    if is_baseline(team) and user.lower() not in admins:
        fail("les noms d'équipe commençant par « Baseline » sont réservés.")
    desc = fields.get(CONFIG["field_description"], "")
    desc = "" if desc == "_No response_" else desc.replace("\n", " ")[:120]

    allowed_file = ROOT / "scoring" / "allowed_users.txt"
    if allowed_file.exists():
        allowed = {line.strip().lower() for line in allowed_file.read_text().splitlines()
                   if line.strip() and not line.strip().startswith("#")}
        if allowed and user.lower() not in allowed:
            fail(f"le compte GitHub `{user}` n'est pas inscrit au défi.")

    subs = read_json(SUBMISSIONS, [])
    # Même équipe quelle que soit la casse : on reprend l'orthographe de la 1re soumission
    known = {s["team"].lower(): s["team"] for s in subs}
    team = known.get(team.lower(), team)
    now = dt.datetime.now(dt.timezone.utc)
    if now > dt.datetime.fromisoformat(CONFIG["deadline"]):
        fail("la date limite du défi est dépassée.")
    recent = [s for s in subs if s["team"].lower() == team.lower()
              and now - dt.datetime.fromisoformat(s["date"]) < dt.timedelta(hours=24)]
    if len(recent) >= CONFIG["max_per_day"]:
        fail(f"l'équipe « {team} » a déjà fait {CONFIG['max_per_day']} soumissions "
             "dans les dernières 24 h.")

    m = ALLOWED_URL.search(fields.get(CONFIG["field_file"], ""))
    if not m:
        fail("aucun fichier joint trouvé. Glissez-déposez `submission.zip` dans le champ prévu.")
    url = m.group(0)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "submission.zip"
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(path, "wb") as f:
                f.write(r.read(30_000_001))
        except Exception as e:  # noqa: BLE001
            fail(f"téléchargement impossible ({e}).")
        if path.stat().st_size > 30_000_000:
            fail("fichier trop volumineux.")
        try:
            pred = load_submission(path)
        except Exception as e:  # noqa: BLE001
            fail(f"fichier illisible ({e}). Utilisez `save_submission()`.")

    gt, public_idx, _ = load_ground_truth()
    if pred.shape != gt.shape:
        fail(f"forme {tuple(pred.shape)} au lieu de {tuple(gt.shape)}.")
    score = dice_score(pred[public_idx], gt[public_idx])

    entry = {"team": team, "user": user, "issue": issue, "url": url,
             "date": now.isoformat(timespec="seconds"), "public_score": round(score, 5),
             "description": desc}
    subs.append(entry)
    SUBMISSIONS.write_text(json.dumps(subs, indent=1, ensure_ascii=False))
    lb = build_leaderboard(subs)
    LEADERBOARD.write_text(json.dumps(lb, indent=1, ensure_ascii=False))
    rank = next(i for i, r in enumerate(lb["rows"], 1) if r["team"] == team)
    left = CONFIG["max_per_day"] - len(recent) - 1
    Path("comment.md").write_text(
        f"✅ Soumission évaluée pour l'équipe **{team}**.\n\n"
        f"| | |\n|---|---|\n| {CONFIG['metric_name']} (test public) | **{score:.4f}** |\n"
        f"| Rang actuel de l'équipe (meilleure soumission) | {rank} / {len(lb['rows'])} |\n"
        f"| Soumissions restantes sur 24 h glissantes | {left} |\n\n"
        f"Le classement de la page du défi sera à jour d'ici quelques minutes (cache de GitHub).\n"
    )
    print(f"{team}: {score:.4f}")


if __name__ == "__main__":
    if "--rebuild" in sys.argv:
        # Régénère docs/leaderboard.json à partir de docs/submissions.json
        LEADERBOARD.write_text(json.dumps(build_leaderboard(read_json(SUBMISSIONS, [])),
                                          indent=1, ensure_ascii=False))
        print("docs/leaderboard.json régénéré")
    else:
        main()
