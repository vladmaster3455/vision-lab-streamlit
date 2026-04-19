# Deploiement sur Hugging Face Spaces (gratuit)

Ce projet contient une version dediee Hugging Face dans le dossier hf_space.

## 1. Ce qui est deja pret

- app Hugging Face: hf_space/app.py
- dependances Hugging Face: hf_space/requirements.txt

## 2. Creer un Space

1. Ouvre Hugging Face
2. Clique sur New Space
3. Choisis:
   - SDK: Gradio
   - Visibility: Public ou Private
   - Hardware: CPU basic (gratuit)
4. Cree le Space

Nom conseille: `vision-lab-yolo-rtdetr-dino`

## 2bis. Fichiers a pousser

A la racine du Space, il faut au minimum:

- `app.py`
- `requirements.txt`
- `README.md` (front-matter Hugging Face)

## 3. Envoyer les fichiers du dossier hf_space

Depuis ton terminal, dans le projet:

1. cd hf_space
2. git init
3. git add app.py requirements.txt README.md
4. git commit -m "Init HF Space app"
5. git branch -M main
6. git remote add origin https://huggingface.co/spaces/VOTRE_USER/VOTRE_SPACE
7. git push -u origin main

Si le remote existe deja:

1. git add app.py requirements.txt README.md
2. git commit -m "Update HF Space"
3. git push

## 4. Lancement automatique

Hugging Face va installer requirements.txt puis demarrer app.py automatiquement.

## 4bis. Etapes de verification apres deploiement

1. Ouvrir le Space
2. Attendre le statut **Running**
3. Faire un test rapide avec YOLO
4. Tester RTDETR
5. Tester DINO

Si DINO ne repond pas, consulter l'onglet **Logs** du Space.

Si tu vois encore une erreur du type **No API found**:

1. Ouvre **Logs**
2. Force un **Restart** du Space
3. Fais un **Rebuild** si tu as modifie `requirements.txt`
4. Attends que le statut redevienne **Running** avant de tester

Si le log mentionne `gradio` ou `api_info`, le rebuild est indispensable apres modification de `requirements.txt`.

## 5. Important

- Cette version HF utilise YOLO, RTDETR et DINO.
- DINO repose sur MMDetection, donc le premier build peut etre plus long.
- Les poids sont telecharges automatiquement au premier lancement.

Variables optionnelles pour DINO (Settings > Variables and secrets du Space):

- DINO_MODEL (defaut: dino-4scale_r50_8xb2-12e_coco)
- DINO_WEIGHTS (optionnel, chemin/url d'un checkpoint custom)

Configuration recommandee dans Settings:

- Hardware: CPU Basic (gratuit)
- Sleep Time: par defaut
- Visibility: selon besoin (Public/Private)

## 6. Si le build echoue

- verifie que le Space est bien en SDK Gradio
- verifie que les fichiers pushes sont bien a la racine du Space: app.py et requirements.txt
- redemarre le Space depuis Settings si besoin
- si DINO echoue, regarde les logs build pour mmcv/mmdet puis redemarre apres rebuild
- garde `gradio==4.36.1` dans `requirements.txt` pour eviter le conflit avec le builder Spaces
