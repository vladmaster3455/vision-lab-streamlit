---
title: Vision Lab Object Detection
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Vision Lab - YOLO / RTDETR / DINO

Application de detection d'objets sur image avec 3 modeles:
- YOLO
- RTDETR
- DINO (via MMDetection)

## Utilisation

1. Charger une image
2. Choisir le modele
3. Regler le seuil de confiance
4. Cliquer sur **Lancer l'analyse**

La sortie affiche:
- image annotee
- tableau des detections
- resume du run

## Variables optionnelles

- `DINO_MODEL` (defaut: `dino-4scale_r50_8xb2-12e_coco`)
- `DINO_WEIGHTS` (optionnel, checkpoint custom)
