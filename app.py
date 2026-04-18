#!/usr/bin/env python3
"""
Application Flask simple pour tester YOLO, RT-DETR et DINO en direct.

Variables d'environnement utiles:
- YOLO_WEIGHTS=yolo11n.pt
- RTDETR_WEIGHTS=rtdetr-l.pt
- DINO_MODEL=dino-4scale_r50_8xb2-12e_coco
- DINO_CONFIG=optionnel si config locale
- DINO_WEIGHTS=optionnel pour checkpoint custom
- ENABLE_DINO=1
- DEVICE=cpu ou cuda:0
"""

from __future__ import annotations

import os
import uuid
import gc
from pathlib import Path
from typing import Dict

import torch
from flask import Flask, redirect, render_template_string, request, send_from_directory, url_for
from PIL import Image, ImageDraw
from ultralytics import RTDETR, YOLO
from werkzeug.utils import secure_filename

try:
    from mmdet.apis import DetInferencer, inference_detector, init_detector
except Exception:
    DetInferencer = None
    inference_detector = None
    init_detector = None


APP_ROOT = Path(__file__).resolve().parent
UPLOADS = APP_ROOT / "web_uploads"
RUNS = APP_ROOT / "web_runs"
EXAMPLE_DIR_CANDIDATES = [
    APP_ROOT / "data" / "processed" / "11060" / "images" / "test",
    APP_ROOT / "data" / "raw" / "11060" / "micrographs",
]
UPLOADS.mkdir(exist_ok=True)
RUNS.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}
DEVICE = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
ENABLE_DINO = os.getenv("ENABLE_DINO", "1") == "1"

MODEL_CACHE: Dict[str, object] = {}
EXAMPLE_IMAGE_MAP: Dict[str, Path] = {}


def discover_example_images(max_examples: int = 3):
    found = []
    for base in EXAMPLE_DIR_CANDIDATES:
        if not base.exists() or not base.is_dir():
            continue
        for candidate in sorted(base.iterdir()):
            if candidate.is_file() and candidate.suffix.lower().lstrip(".") in ALLOWED_EXTENSIONS:
                found.append(candidate)
                if len(found) >= max_examples:
                    break
        if len(found) >= max_examples:
            break
    mapping = {}
    for idx, path in enumerate(found, start=1):
        mapping[f"ex{idx}"] = path
    return mapping


EXAMPLE_IMAGE_MAP = discover_example_images()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clear_loaded_models(keep_key: str | None = None):
    keys_to_drop = [key for key in MODEL_CACHE.keys() if key != keep_key]
    for key in keys_to_drop:
        MODEL_CACHE.pop(key, None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_model(name: str):
    key = name.lower()
    if key in MODEL_CACHE:
        # Enforce single-model residency to cap memory on small instances.
        clear_loaded_models(keep_key=key)
        return MODEL_CACHE[key]

    clear_loaded_models()

    if key == "yolo":
        MODEL_CACHE[key] = YOLO(os.getenv("YOLO_WEIGHTS", "yolo11n.pt"))
    elif key == "rtdetr":
        MODEL_CACHE[key] = RTDETR(os.getenv("RTDETR_WEIGHTS", "rtdetr-l.pt"))
    elif key == "dino":
        if not ENABLE_DINO:
            raise RuntimeError("DINO est désactivé sur ce déploiement")
        dino_config = os.getenv("DINO_CONFIG", "").strip()
        dino_weights = os.getenv("DINO_WEIGHTS", "").strip() or None
        dino_model = os.getenv("DINO_MODEL", "dino-4scale_r50_8xb2-12e_coco")
        if dino_config and init_detector is not None and inference_detector is not None:
            MODEL_CACHE[key] = {
                "mode": "native",
                "model": init_detector(dino_config, dino_weights, device=DEVICE),
            }
        elif DetInferencer is not None:
            MODEL_CACHE[key] = {
                "mode": "inferencer",
                "model": DetInferencer(model=dino_model, weights=dino_weights, device=DEVICE),
            }
        else:
            raise RuntimeError("MMDetection / DetInferencer indisponible côté serveur")
    else:
        raise ValueError(f"Modèle inconnu: {name}")
    return MODEL_CACHE[key]


def predict_boxes(model_name: str, image_path: Path):
    model_name = model_name.lower()
    if model_name in {"yolo", "rtdetr"}:
        model = get_model(model_name)
        results = model.predict(source=str(image_path), device=DEVICE, conf=0.15, verbose=False)
        result = results[0]
        preds = []
        if result.boxes is None:
            return preds
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy() if result.boxes.cls is not None else [0] * len(scores)
        names = getattr(model, "names", {})

        for box, score, cls_id in zip(xyxy, scores, classes):
            x1, y1, x2, y2 = [float(v) for v in box]
            class_idx = int(cls_id)
            if isinstance(names, dict):
                label_name = str(names.get(class_idx, f"class_{class_idx}"))
            elif isinstance(names, (list, tuple)) and 0 <= class_idx < len(names):
                label_name = str(names[class_idx])
            else:
                label_name = f"class_{class_idx}"

            preds.append({"bbox_xyxy": [x1, y1, x2, y2], "score": float(score), "label": label_name})
        return preds

    if model_name == "dino":
        bundle = get_model("dino")
        preds = []
        if bundle["mode"] == "native":
            result = inference_detector(bundle["model"], str(image_path))
            instances = result.pred_instances
            if instances is None or len(instances) == 0:
                return preds
            bboxes = instances.bboxes.detach().cpu().numpy()
            scores = instances.scores.detach().cpu().numpy()
            labels = instances.labels.detach().cpu().numpy()
            classes = (getattr(bundle["model"], "dataset_meta", {}) or {}).get("classes", None)

            for bbox, score, label in zip(bboxes, scores, labels):
                if float(score) < 0.15:
                    continue
                x1, y1, x2, y2 = [float(v) for v in bbox]
                label_idx = int(label)
                if classes is not None and 0 <= label_idx < len(classes):
                    label_name = str(classes[label_idx])
                else:
                    label_name = f"class_{label_idx}"
                preds.append({"bbox_xyxy": [x1, y1, x2, y2], "score": float(score), "label": label_name})
            return preds

        outputs = bundle["model"](str(image_path), no_save_vis=True, no_save_pred=True)
        prediction = outputs["predictions"][0]
        bboxes = prediction.get("bboxes", [])
        scores = prediction.get("scores", [])
        labels = prediction.get("labels", [])
        label_names = prediction.get("label_names", [])

        for idx, (bbox, score) in enumerate(zip(bboxes, scores)):
            if float(score) < 0.15:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            if idx < len(label_names):
                label_name = str(label_names[idx])
            elif idx < len(labels):
                label_name = f"class_{int(labels[idx])}"
            else:
                label_name = "objet"
            preds.append({"bbox_xyxy": [x1, y1, x2, y2], "score": float(score), "label": label_name})
        return preds

    raise ValueError("Modèle non supporté")


def annotate_image(src: Path, preds, dst: Path):
    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)
    for pred in preds[:80]:
        x1, y1, x2, y2 = pred["bbox_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=(11, 95, 165), width=3)
        draw.text((x1 + 4, max(0, y1 - 14)), f"{pred['label']} {pred['score']:.2f}", fill=(255, 255, 255))
    img.save(dst)


HTML = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="theme-color" content="#07111f">
    <title>Vision Lab</title>
  <style>
        :root {
            --bg: #07111f;
            --panel: rgba(10, 19, 33, 0.84);
            --panel-strong: #0d1829;
            --line: rgba(148, 163, 184, 0.18);
            --text: #e5eefb;
            --muted: #94a3b8;
            --accent: #72d0ff;
            --accent-2: #7c5cff;
            --good: #38d39f;
            --shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            color: var(--text);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background:
                radial-gradient(circle at top left, rgba(124, 92, 255, 0.22), transparent 28%),
                radial-gradient(circle at top right, rgba(114, 208, 255, 0.18), transparent 22%),
                linear-gradient(180deg, #050b14 0%, #07111f 48%, #091322 100%);
            min-height: 100vh;
        }
        .wrap { max-width: 1260px; margin: 0 auto; padding: 28px 20px 40px; }
        .hero {
            display: grid;
            grid-template-columns: 1.25fr 0.75fr;
            gap: 20px;
            align-items: stretch;
            margin-bottom: 20px;
        }
        .card {
            background: linear-gradient(180deg, rgba(15, 25, 42, 0.92), rgba(10, 17, 30, 0.92));
            border: 1px solid var(--line);
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 24px;
            backdrop-filter: blur(12px);
        }
        .hero-main { padding: 28px; position: relative; overflow: hidden; }
        .hero-main::after {
            content: "";
            position: absolute;
            inset: auto -20px -60px auto;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(114, 208, 255, 0.24), transparent 68%);
            pointer-events: none;
        }
        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border: 1px solid var(--line);
            border-radius: 999px;
            color: var(--accent);
            font-size: 12px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            background: rgba(114, 208, 255, 0.08);
        }
        .success-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 12px;
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid rgba(56, 211, 159, 0.35);
            background: rgba(56, 211, 159, 0.12);
            color: #9ef2d4;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        h1 { margin: 16px 0 10px; font-size: clamp(2rem, 4vw, 3.4rem); line-height: 1.02; }
        .lead { margin: 0; color: #c7d5ea; font-size: 1.04rem; line-height: 1.6; max-width: 60ch; }
        .hero-side { display: grid; gap: 12px; }
        .stat {
            padding: 18px;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.03);
        }
        .stat .kpi { font-size: 1.45rem; font-weight: 700; margin: 6px 0 0; }
        .stat .label { color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
        .overview-card {
            position: sticky;
            top: 20px;
            align-self: start;
        }
        .title-row { display: flex; align-items: end; justify-content: space-between; gap: 16px; margin-bottom: 18px; }
        .title-row h2, .title-row h3 { margin: 0; }
        .title-row p { margin: 6px 0 0; color: var(--muted); }
        .form-grid { display: grid; gap: 16px; }
        .field label { display: block; margin-bottom: 8px; color: #dce7f6; font-weight: 600; }
        .field small { display: block; margin-top: 8px; color: var(--muted); }
        input[type="file"], select {
            width: 100%;
            padding: 13px 14px;
            border-radius: 14px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.04);
            color: var(--text);
            outline: none;
        }
        select option { color: #0a1120; }
        .cta-row { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
        button {
            padding: 13px 18px;
            border-radius: 14px;
            border: 0;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
            color: #05101d;
            font-weight: 800;
            cursor: pointer;
            box-shadow: 0 12px 28px rgba(124, 92, 255, 0.24);
        }
        .ghost {
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.03);
            color: var(--text);
            box-shadow: none;
        }
        .examples { display: grid; grid-template-columns: repeat(auto-fill, minmax(116px, 1fr)); gap: 8px; }
        .example-item {
            display: block;
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 7px;
            background: rgba(255,255,255,0.03);
            cursor: pointer;
            transition: transform .18s ease, border-color .18s ease, background .18s ease;
        }
        .example-item:hover { transform: translateY(-2px); border-color: rgba(114, 208, 255, 0.5); background: rgba(114, 208, 255, 0.06); }
        .example-item input { margin-bottom: 6px; }
        .example-item img { width: 100%; height: 72px; object-fit: cover; border-radius: 10px; display: block; margin-bottom: 6px; }
        .example-name { display: block; font-size: 10px; color: #d3def0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .hint { color: var(--muted); font-size: 13px; line-height: 1.5; }
        .img-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
        .panel-img {
            width: 100%;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: #050b14;
            display: block;
        }
        .pred {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 13px;
            line-height: 1.5;
            background: rgba(255,255,255,0.03);
            padding: 14px;
            border-radius: 16px;
            border: 1px solid var(--line);
            max-height: 320px;
            overflow: auto;
            color: #dce7f6;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .result-shell { display: grid; gap: 14px; scroll-margin-top: 16px; }
        .result-strip { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
        .result-chip {
            padding: 14px 16px;
            border-radius: 16px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.03);
        }
        .result-chip .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }
        .result-chip .value { margin-top: 8px; font-size: 1.02rem; font-weight: 700; }
        .result-list { display: grid; gap: 10px; }
        .result-item {
            display: grid;
            grid-template-columns: auto 1fr auto;
            gap: 12px;
            align-items: center;
            padding: 12px 14px;
            border-radius: 14px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.03);
        }
        .badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 74px;
            padding: 7px 10px;
            border-radius: 999px;
            background: rgba(114, 208, 255, 0.12);
            color: var(--accent);
            font-size: 12px;
            font-weight: 700;
        }
        .meta-stack { display: grid; gap: 2px; }
        .meta-stack .top { font-weight: 700; }
        .meta-stack .bottom { color: var(--muted); font-size: 12px; }
        .score-pill {
            padding: 7px 10px;
            border-radius: 999px;
            background: rgba(56, 211, 159, 0.12);
            color: var(--good);
            font-weight: 800;
            font-size: 12px;
        }
        .score-track {
            grid-column: 2 / 4;
            height: 8px;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.14);
            overflow: hidden;
            margin-top: -4px;
        }
        .score-fill {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, var(--accent) 0%, var(--good) 100%);
        }
        .output-meta { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 14px; }
        .mini {
            padding: 14px;
            border-radius: 16px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.03);
        }
        .mini .label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
        .mini .value { margin-top: 8px; font-weight: 700; font-size: 1.05rem; }
        .error { border-color: rgba(248, 113, 113, 0.35); background: rgba(248, 113, 113, 0.08); color: #ffd6d6; }
        .footer-note { margin-top: 18px; color: var(--muted); font-size: 12px; line-height: 1.5; }
        @media (max-width: 960px) {
            .hero, .grid, .img-grid, .output-meta { grid-template-columns: 1fr; }
            .wrap { padding: 16px; }
        }
  </style>
</head>
<body>
<div class="wrap">
    <section class="hero">
        <div class="card hero-main">
            <span class="eyebrow">Vision Lab</span>
            <h1>Analyse d’images par YOLO, RT-DETR et DINO</h1>
            <p class="lead">Importe une image, choisis un modèle, puis récupère une image annotée et une liste de prédictions. L’interface est conçue pour une démo propre sur des images du quotidien ou des images médicales.</p>
            <div class="footer-note"><strong>Domaine virus ciblé:</strong> particules virales en cryo-EM (CryoVirusDB / EMPIAR 11060). Sur images non médicales, YOLO et RT-DETR utilisent leurs classes générales (ex: personne, vélo, voiture, animal, etc.).</div>
            {% if has_results %}
            <div class="success-pill">Analyse terminée - résultats disponibles</div>
            {% endif %}
            <div class="footer-note">Mode démo: upload manuel ou sélection d’un exemple. Les modèles restent les mêmes, seule l’expérience utilisateur change.</div>
        </div>
        <div class="hero-side">
            <div class="stat card">
                <div class="label">Entrée</div>
                <div class="kpi">Image uploadée</div>
            </div>
            <div class="stat card">
                <div class="label">Sortie</div>
                <div class="kpi">Image annotée + objets détectés</div>
            </div>
            <div class="stat card">
                <div class="label">Usages</div>
                <div class="kpi">Général + médical</div>
            </div>
        </div>
    </section>

    <div class="grid">
        <section class="card overview-card">
            <div class="title-row">
                <div>
                    <h2>Tester une image</h2>
                    <p>Choisis un modèle puis envoie une image ou un exemple.</p>
                </div>
            </div>
            <form method="post" enctype="multipart/form-data" class="form-grid">
                <div class="field">
                    <label for="model">Modèle</label>
                    <select name="model" id="model">
                        <option value="yolo">YOLO</option>
                        <option value="rtdetr">RT-DETR</option>
                        {% if enable_dino %}
                        <option value="dino">DINO</option>
                        {% endif %}
                    </select>
                </div>

                <div class="field">
                    <label for="image">Uploader une image</label>
                    <input id="image" type="file" name="image" accept="image/*">
                    <small>PNG, JPG, JPEG, TIFF, BMP. Si tu choisis un exemple à droite, l’upload devient optionnel.</small>
                </div>

                <div class="field">
                    <label>Ou choisir 3 images d’exemple</label>
                    {% if examples %}
                    <div class="examples">
                        {% for ex in examples %}
                        <label class="example-item">
                            <input type="radio" name="example_key" value="{{ ex.key }}">
                            <img src="{{ ex.url }}" alt="{{ ex.name }}">
                            <span class="example-name">{{ ex.name }}</span>
                        </label>
                        {% endfor %}
                    </div>
                    {% else %}
                    <p class="hint">Aucun exemple local n’est disponible. Utilise l’upload manuel.</p>
                    {% endif %}
                </div>

                <div class="cta-row">
                    <button type="submit">Lancer l’analyse</button>
                    <button type="reset" class="ghost">Réinitialiser</button>
                </div>
            </form>
            <div class="footer-note">L’app peut analyser des objets généraux et des images biomédicales. Les résultats dépendent des poids chargés et du domaine visuel.</div>
        </section>

        <section class="card">
            <div class="title-row">
                <div>
                    <h3>Ce que l’app renvoie</h3>
                    <p>Un aperçu visuel et un résumé des résultats.</p>
                </div>
            </div>
            <div class="output-meta">
                <div class="mini">
                    <div class="label">Étape 1</div>
                    <div class="value">Image d’entrée</div>
                </div>
                <div class="mini">
                    <div class="label">Étape 2</div>
                    <div class="value">Boîtes de détection</div>
                </div>
                <div class="mini">
                    <div class="label">Étape 3</div>
                    <div class="value">Liste lisible</div>
                </div>
            </div>
            <div class="hint">Les images médicales fonctionnent bien pour la démo si tu utilises des exemples déjà présents dans le dataset ou un upload direct.</div>
        </section>
    </div>

  {% if original_url and output_url %}
    <section class="grid">
        <div class="card">
            <div class="title-row">
                <div>
                    <h3>Image source</h3>
                    <p>Image envoyée ou exemple choisi.</p>
                </div>
            </div>
            <img class="panel-img" src="{{ original_url }}" alt="source">
    </div>
        <div class="card">
            <div class="title-row">
                <div>
                    <h3>Détection annotée</h3>
                    <p>Résultat visuel produit par le modèle.</p>
                </div>
            </div>
            <img class="panel-img" src="{{ output_url }}" alt="output">
        </div>
    </section>

    <section class="card" id="analysis-results" style="margin-top: 20px;">
        <div class="title-row">
            <div>
                <h3>Prédictions</h3>
                <p>Résumé compact des objets détectés.</p>
            </div>
        </div>
        <div class="result-shell">
            <div class="result-strip">
                <div class="result-chip">
                    <div class="label">Nombre d'objets</div>
                    <div class="value">{{ prediction_count }}</div>
                </div>
                <div class="result-chip">
                    <div class="label">Score max</div>
                    <div class="value">{{ max_score }}</div>
                </div>
                <div class="result-chip">
                    <div class="label">Modèle</div>
                    <div class="value">{{ model_name_display }}</div>
                </div>
            </div>
            <div class="hint">{{ analysis_mode }}</div>
            {% if predictions_data %}
            <div class="result-list">
                {% for pred in predictions_data %}
                <div class="result-item">
                    <div class="badge">{{ pred.label }}</div>
                    <div class="meta-stack">
                        <div class="top">Détection {{ loop.index }}</div>
                        <div class="bottom">BBox: {{ pred.bbox_xyxy[0] | round(1) }}, {{ pred.bbox_xyxy[1] | round(1) }}, {{ pred.bbox_xyxy[2] | round(1) }}, {{ pred.bbox_xyxy[3] | round(1) }}</div>
                    </div>
                    <div class="score-pill">{{ '%.2f' | format(pred.score) }}</div>
                    <div class="score-track"><div class="score-fill" style="width: {{ (pred.score * 100) | round(0) }}%;"></div></div>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <div class="hint">Aucune détection au-dessus du seuil.</div>
            {% endif %}
        </div>
    </section>
  {% endif %}

    {% if has_results %}
    <script>
        window.addEventListener('load', function () {
            const target = document.getElementById('analysis-results');
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    </script>
    {% endif %}

  {% if error %}
    <section class="card error" style="margin-top: 20px;"><strong>Erreur:</strong> {{ error }}</section>
  {% endif %}
</div>
</body>
</html>
"""

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOADS, filename)


@app.route("/runs/<path:filename>")
def runs_file(filename):
    return send_from_directory(RUNS, filename)


@app.route("/healthz", methods=["GET"])
def healthz():
    return {"status": "ok"}, 200


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


@app.route("/examples/<path:key>")
def example_file(key):
    path = EXAMPLE_IMAGE_MAP.get(key)
    if path is None:
        return "Exemple introuvable", 404
    return send_from_directory(path.parent, path.name)


def build_example_payload():
    payload = []
    for key, path in EXAMPLE_IMAGE_MAP.items():
        payload.append({"key": key, "name": path.name, "url": url_for("example_file", key=key)})
    return payload


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(
            HTML,
            original_url=None,
            output_url=None,
            predictions_data=[],
            error=None,
            examples=build_example_payload(),
            prediction_count=0,
            max_score="0.00",
            model_name_display="-",
            analysis_mode="-",
            has_results=False,
            enable_dino=ENABLE_DINO,
        )

    try:
        file = request.files.get("image")
        example_key = request.form.get("example_key", "").strip()
        model_name = request.form.get("model", "yolo")
        input_path = None
        original_url = None

        if file and file.filename:
            if not allowed_file(file.filename):
                raise ValueError("Extension non supportée")
            uid = uuid.uuid4().hex[:12]
            filename = secure_filename(file.filename)
            ext = filename.rsplit(".", 1)[1].lower()
            input_name = f"{uid}_input.{ext}"
            input_path = UPLOADS / input_name
            file.save(input_path)
            original_url = url_for("uploaded_file", filename=input_name)
        elif example_key:
            input_path = EXAMPLE_IMAGE_MAP.get(example_key)
            if input_path is None:
                raise ValueError("Exemple sélectionné introuvable")
            original_url = url_for("example_file", key=example_key)
            uid = uuid.uuid4().hex[:12]
        else:
            raise ValueError("Envoyez une image ou choisissez un exemple")

        output_name = f"{uid}_{model_name}.png"
        output_path = RUNS / output_name

        preds = predict_boxes(model_name, input_path)
        annotate_image(input_path, preds, output_path)
        prediction_count = len(preds)
        max_score = f"{max((p['score'] for p in preds), default=0.0):.2f}"
        model_name_display = model_name.upper()
        if model_name in {"yolo", "rtdetr"}:
            analysis_mode = "Mode objets généraux: le modèle utilise ses classes connues (COCO) pour les images non médicales."
        else:
            analysis_mode = "Mode DINO: détection générique, les libellés dépendent du checkpoint et de la configuration active."

        return render_template_string(
            HTML,
            original_url=original_url,
            output_url=url_for("runs_file", filename=output_name),
            predictions_data=preds[:12],
            error=None,
            examples=build_example_payload(),
            prediction_count=prediction_count,
            max_score=max_score,
            model_name_display=model_name_display,
            analysis_mode=analysis_mode,
            has_results=True,
            enable_dino=ENABLE_DINO,
        )
    except Exception as e:
        return render_template_string(
            HTML,
            original_url=None,
            output_url=None,
            predictions_data=[],
            error=str(e),
            examples=build_example_payload(),
            prediction_count=0,
            max_score="0.00",
            model_name_display="-",
            analysis_mode="-",
            has_results=False,
            enable_dino=ENABLE_DINO,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=False)
