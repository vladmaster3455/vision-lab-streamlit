# Valeurs exactes pour deploy Render

## A. Formulaire Render (Web Service)

Remplis comme ceci:

- Source Code: ton repo GitHub
- Branch: main
- Language: Python 3
- Root Directory: vide

Build Command:

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

Start Command:

gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 300

Health Check Path:

/healthz

Auto-Deploy:

On Commit

Instance Type:

Starter (recommande)

## B. Variables d environnement Render

Ajoute ces variables:

- PYTHON_VERSION = 3.10.13
- DEVICE = cpu
- ENABLE_DINO = 1
- YOLO_WEIGHTS = yolo11n.pt
- RTDETR_WEIGHTS = rtdetr-l.pt
- DINO_MODEL = dino-4scale_r50_8xb2-12e_coco
- DINO_CONFIG =
- DINO_WEIGHTS =

## C. Commandes Git minimales

Depuis le dossier du projet:

git init
git add app.py benchmark.py generate_report.py generate_pptx.py prepare_cryovirusdb.py render.yaml requirements.txt README.md DEPLOY_GITHUB.md RENDER_IMAGES.md RENDER_DEPLOY_VALUES.md
git commit -m "Initial project setup"
git branch -M main
git remote add origin https://github.com/VOTRE_COMPTE/VOTRE_REPO.git
git push -u origin main

## D. Erreur a eviter

Ne mets pas ce Start Command:

gunicorn your_application.wsgi

Le bon Start Command est celui de la section A avec app:app.
