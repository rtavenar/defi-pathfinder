# Défi Pathfinder

Défi de segmentation image-vers-image sur un sous-échantillon de
[Pathfinder-X2](https://huggingface.co/datasets/Tylersuard/PathfinderX2) (CC BY 4.0).

- Page du défi : https://rtavenar.github.io/defi-pathfinder/
- Étudiant·es : `pathfinder_data.py` (données + soumission) et `starter.py` (notebook jupytext)

## Contenu du dépôt

| Chemin | Rôle |
|---|---|
| `docs/` | Site GitHub Pages (`index.html`, `leaderboard.json`, `submissions.json`, exemples) |
| `pathfinder_data.py` | Loader étudiant : téléchargement, `load_train`, `load_test`, `dice_score`, `save_submission` |
| `starter.py` | Notebook de démarrage (baseline seuillage + petit U-Net) |
| `prepare_data.py` | Préparation des données (enseignant, une seule fois) |
| `scoring/score_submission.py` | Évaluation d'une soumission, lancée par la GitHub Action |
| `scoring/final_ranking.py` | Classement final sur le test privé (enseignant, en local) |
| `scoring/config.json` | Date limite, nombre de soumissions par 24 h, noms des baselines |
| `scoring/allowed_users.txt` | Liste optionnelle des comptes GitHub autorisés |
| `scoring/test_masks.enc` | Vérité terrain du test, chiffrée (créée par `prepare_data.py`) |
| `.github/workflows/score.yml` | Action : issue « Soumission » → score → classement |

## Mise en place (enseignant)

### 1. Préparer les données

```bash
pip install pillow numpy cryptography
python prepare_data.py --n-train 5000 --n-test 2000
```

Les données de Pathfinder-X2 sont deux archives zip sur S3 (le dépôt Hugging Face
ne contient qu'un script de chargement, que `datasets` ne sait plus exécuter). Le
script lit l'index des deux archives, tire 7 000 paires image/masque au hasard et
ne télécharge que celles-ci, par requêtes HTTP partielles : pas besoin de
récupérer les archives complètes. `python prepare_data.py --describe` affiche le
début de l'index des archives. Il applique une rotation / symétrie aléatoire à chaque
exemple, et écrit :

- `release/` : fichiers publics, à attacher à une GitHub Release ;
- `private/` : vérité terrain en clair et clé de chiffrement (**jamais committés**, cf. `.gitignore`) ;
- `scoring/test_masks.enc` : vérité terrain chiffrée, à committer ;
- `docs/assets/sample_*.png` : exemples affichés sur la page.

**À vérifier** : `private/stats.json` (le masque doit être binaire et la fraction de
pixels positifs plausible) et les images `docs/assets/sample_*_mask.png` (le masque
doit bien être le serpent qui porte le disque). Le script binarise
l'annotation à 127 ; si l'annotation de Pathfinder-X2 code autre chose, c'est
`to_arrays()` qu'il faut adapter.

### 2. Créer le dépôt et publier les données

```bash
git init && git add . && git commit -m "Défi Pathfinder"
gh repo create rtavenar/defi-pathfinder --public --source . --push
gh release create data-v1 release/*.npz release/SHA256SUMS.txt \
    --title "Données v1" --notes "5000 train / 2000 test, 512x512"
```

Le dépôt doit être **public** : c'est ce qui rend les liens de téléchargement directs
et permet à l'Action de récupérer les pièces jointes des issues sans authentification.

### 3. Configurer GitHub

1. *Settings → Secrets and variables → Actions* : créer le secret `GT_KEY` avec le
   contenu de `private/GT_KEY.txt`.
2. *Settings → Pages* : source « Deploy from a branch », branche `main`, dossier `/docs`.
3. *Settings → Actions → General → Workflow permissions* : « Read and write permissions ».
4. *Issues → Labels* : créer le label `soumission` (sans lui, le formulaire ne
   l'applique pas et l'Action ne se déclenche pas).
5. Ajuster `scoring/config.json` (date limite, `max_per_day`) et les dates de la
   section Règles dans `docs/index.html`.
6. Optionnel : lister les comptes GitHub des étudiant·es dans `scoring/allowed_users.txt`.

### 4. Tester avec la baseline

Lancer `starter.py` jusqu'à la baseline par seuillage, écrire sa soumission
(`save_submission(threshold_baseline(X_test))`), puis ouvrir une issue
« Soumission » avec le nom d'équipe `Baseline (seuillage)`. Ce nom figure dans
`baseline_teams`, il apparaît donc en italique et hors rang dans le classement.
Même chose pour le U-Net avec `Baseline (U-Net)`.

### 5. Après la date limite

```bash
python scoring/final_ranking.py              # meilleure soumission publique de chaque équipe
python scoring/final_ranking.py --select last
```

Le classement final sur la moitié privée du test est écrit dans `private/final_ranking.csv`.

## Fonctionnement du classement

- Un·e étudiant·e ouvre une issue avec le formulaire « Soumission » et y dépose `submission.zip`.
- L'Action déchiffre la vérité terrain, calcule le Dice moyen sur la moitié
  **publique** du test, commente l'issue avec le score, la ferme, puis committe
  `docs/leaderboard.json` et `docs/submissions.json`.
- La page lit `leaderboard.json` directement sur `raw.githubusercontent.com` : elle
  est à jour sans attendre la reconstruction de GitHub Pages.
- Garde-fous : 5 soumissions par équipe sur 24 h glissantes, date limite, liste
  blanche optionnelle, taille de fichier limitée, seules les pièces jointes
  GitHub sont téléchargées, et le corps de l'issue n'est jamais interpolé dans le shell.
- Les scores privés ne sont jamais calculés en ligne : ils ne peuvent donc pas fuiter.

Limite connue : les images de test proviennent d'un jeu public. La rotation
aléatoire empêche une simple comparaison octet à octet, mais un·e étudiant·e
déterminé·e pourrait retrouver les masques dans Pathfinder-X2. La règle
l'interdit explicitement, et le code est demandé avec le rapport.
