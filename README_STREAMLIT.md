# Deployment Streamlit

Cette version ne depend pas de hf_space.

## 1) Lancer en local

1. Active ton environnement Python
2. Installe les dependances
3. Lance Streamlit

Commandes:

```bash
pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py --server.port 8501
```

Option DINO (si tu veux activer DINO dans Streamlit):

```bash
pip install mmengine>=0.10.5 mmcv-lite>=2.1.0,<2.2.0 mmdet>=3.3.0
```

## 2) Deployer sur Streamlit Community Cloud

1. Pousse le projet sur GitHub
2. Ouvre Streamlit Community Cloud
3. New app
4. Choisis le repo et la branche
5. Main file path: streamlit_app.py
6. Python version: 3.11 (important)
7. Deploy

Important: garde le fichier `runtime.txt` dans le repo avec `python-3.11` pour eviter Python 3.14 sur Cloud.

## 3) Variables optionnelles

- DEVICE=cpu
- YOLO_WEIGHTS=yolo11n.pt
- RTDETR_WEIGHTS=rtdetr-l.pt
- DINO_MODEL=dino-4scale_r50_8xb2-12e_coco
- DINO_WEIGHTS=<checkpoint optionnel>

## 4) Si DINO ne marche pas

- verifie que mmengine, mmcv-lite et mmdet sont installes
- regarde les logs de build
- teste YOLO d abord pour confirmer que l app demarre

## 5) Note importante sur ton erreur

- Le `requirements.txt` principal contient beaucoup de dependances pour tout le projet, ce qui peut provoquer du backtracking pip.
- Pour Streamlit, utilise seulement `requirements_streamlit.txt`.

## 6) Erreur ImportError cv2 / ultralytics sur Cloud

Si tu vois une erreur avec `python3.14` dans la traceback:

1. verifie que `runtime.txt` est bien pousse avec `python-3.11`
2. verifie que le deploiement Streamlit pointe bien sur le bon repo et la bonne branche (celle qui contient `runtime.txt`)
3. ajoute aussi `.python-version` (valeur `3.11`) dans le meme repo pour renforcer le pin
4. sur Streamlit Cloud, ouvre les settings de l app et force Python 3.11 si propose
5. fais Reboot app puis Clear cache puis Redeploy
6. verifie dans les logs que l environnement est bien en Python 3.11

## 7) Erreur `libGL.so.1` manquante sur Streamlit Cloud

Si tu vois `cannot open shared object file: libGL.so.1`:

1. verifie que le fichier `packages.txt` est bien present dans le repo
2. contenu minimum conseille:
	- `libgl1`
3. n'ajoute `ffmpeg` ou `libglib2.0-0` que si un log le demande explicitement (sinon cela peut casser l'installation apt)
4. push `packages.txt`, puis fais Reboot app + Clear cache + Redeploy

Si tu vois `libgthread-2.0.so.0` manquante:

1. ajoute `libglib2.0-0t64` dans `packages.txt` (image Debian recente Streamlit)
2. garde `ffmpeg` absent (ne pas le remettre)
3. push, puis Reboot app + Clear cache + Redeploy

Si tu vois `Failed building wheel for pillow`:

1. verifie que `requirements.txt` contient `pillow>=11.3.0`
2. verifie que `runtime.txt` est `python-3.11`
3. fais Clear cache + Reboot + Redeploy

## 8) Ou sont les modeles (poids) sur Streamlit Cloud

Tu ne vois pas forcement les fichiers de poids dans ton repo GitHub, c est normal.

1. Les modeles Ultralytics (YOLO, RTDETR) se telechargent automatiquement au premier usage
2. Sur Streamlit Cloud, ils sont stockes dans le cache du conteneur, pas dans ton depot GitHub
3. En cas de redeploy, changement de machine ou clear cache, ils peuvent etre retelcharges
4. Donc ne pas voir les `.pt` dans le repo n est pas un bug

Si tu veux des poids fixes:

1. pousse les fichiers `.pt` dans le repo (ou via Git LFS)
2. configure les variables `YOLO_WEIGHTS` et `RTDETR_WEIGHTS` avec le chemin local dans le repo
