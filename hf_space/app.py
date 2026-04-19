#!/usr/bin/env python3
from __future__ import annotations

import gc
import os
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import RTDETR, YOLO

MMDET_IMPORT_ERROR: Exception | None = None
try:
    from mmdet.apis import DetInferencer
except Exception as e:  # pragma: no cover
    MMDET_IMPORT_ERROR = e
    DetInferencer = None

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_CACHE: Dict[str, object] = {}

MODEL_CONFIG = {
    "YOLO": "yolo11n.pt",
    "RTDETR": "rtdetr-l.pt",
    "DINO": os.getenv("DINO_MODEL", "dino-4scale_r50_8xb2-12e_coco"),
}


def clear_models(keep: str | None = None) -> None:
    to_drop = [k for k in MODEL_CACHE.keys() if k != keep]
    for k in to_drop:
        MODEL_CACHE.pop(k, None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_model(model_name: str):
    if model_name in MODEL_CACHE:
        clear_models(keep=model_name)
        return MODEL_CACHE[model_name]

    clear_models()
    weights = MODEL_CONFIG[model_name]
    if model_name == "YOLO":
        MODEL_CACHE[model_name] = YOLO(weights)
    elif model_name == "RTDETR":
        MODEL_CACHE[model_name] = RTDETR(weights)
    else:
        if DetInferencer is None:
            raise gr.Error(
                "DINO indisponible: MMDetection n'est pas charge dans ce Space. "
                f"Detail: {MMDET_IMPORT_ERROR}"
            )
        dino_weights = os.getenv("DINO_WEIGHTS", "").strip() or None
        MODEL_CACHE[model_name] = DetInferencer(model=weights, weights=dino_weights, device=DEVICE)
    return MODEL_CACHE[model_name]


def _predict_ultralytics(model_name: str, image: Image.Image, conf: float, max_boxes: int) -> Tuple[Image.Image, List[List[str]], str]:
    model = get_model(model_name)
    rgb = image.convert("RGB")
    arr = np.array(rgb)

    results = model.predict(source=arr, conf=float(conf), device=DEVICE, verbose=False)
    result = results[0]

    draw = ImageDraw.Draw(rgb)
    rows: List[List[str]] = []

    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy() if result.boxes.cls is not None else np.zeros(len(scores))
        names = getattr(model, "names", {})

        for idx, (box, score, cls_id) in enumerate(zip(xyxy, scores, classes), start=1):
            if idx > int(max_boxes):
                break
            x1, y1, x2, y2 = [float(v) for v in box]
            label_id = int(cls_id)
            if isinstance(names, dict):
                label_name = str(names.get(label_id, f"class_{label_id}"))
            elif isinstance(names, (list, tuple)) and 0 <= label_id < len(names):
                label_name = str(names[label_id])
            else:
                label_name = f"class_{label_id}"

            draw.rectangle([x1, y1, x2, y2], outline=(0, 170, 220), width=3)
            draw.text((x1 + 4, max(0, y1 - 16)), f"{label_name} {score:.2f}", fill=(255, 255, 255))
            rows.append([
                str(idx),
                label_name,
                f"{score:.3f}",
                f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]",
            ])

    summary = f"Modele: {model_name} | Device: {DEVICE} | Detections: {len(rows)}"
    return rgb, rows, summary


def _predict_dino(image: Image.Image, conf: float, max_boxes: int) -> Tuple[Image.Image, List[List[str]], str]:
    inferencer = get_model("DINO")
    rgb = image.convert("RGB")
    arr = np.array(rgb)
    outputs = inferencer(arr, no_save_vis=True, no_save_pred=True)
    prediction = outputs["predictions"][0]

    bboxes = prediction.get("bboxes", [])
    scores = prediction.get("scores", [])
    labels = prediction.get("labels", [])
    label_names = prediction.get("label_names", [])

    draw = ImageDraw.Draw(rgb)
    rows: List[List[str]] = []
    row_id = 0

    for i, (bbox, score) in enumerate(zip(bboxes, scores), start=1):
        score_f = float(score)
        if score_f < float(conf):
            continue
        row_id += 1
        if row_id > int(max_boxes):
            break

        x1, y1, x2, y2 = [float(v) for v in bbox]
        if i - 1 < len(label_names):
            label_name = str(label_names[i - 1])
        elif i - 1 < len(labels):
            label_name = f"class_{int(labels[i - 1])}"
        else:
            label_name = "objet"

        draw.rectangle([x1, y1, x2, y2], outline=(255, 147, 38), width=3)
        draw.text((x1 + 4, max(0, y1 - 16)), f"{label_name} {score_f:.2f}", fill=(255, 255, 255))
        rows.append([
            str(row_id),
            label_name,
            f"{score_f:.3f}",
            f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]",
        ])

    summary = f"Modele: DINO | Device: {DEVICE} | Detections: {len(rows)}"
    return rgb, rows, summary


def predict(image: Image.Image, model_name: str, conf: float, max_boxes: int) -> Tuple[Image.Image, List[List[str]], str]:
    if image is None:
        raise gr.Error("Merci de charger une image avant de lancer l'analyse.")
    if model_name == "DINO":
        return _predict_dino(image, conf, max_boxes)
    return _predict_ultralytics(model_name, image, conf, max_boxes)


with gr.Blocks(title="Vision Lab - Hugging Face") as demo:
    gr.Markdown("""
# Vision Lab sur Hugging Face
Demo detection d'objets avec YOLO, RTDETR et DINO.

- Charge une image
- Choisis un modele
- Lance l'analyse
""")

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="Image a analyser")
            model_in = gr.Dropdown(choices=["YOLO", "RTDETR", "DINO"], value="YOLO", label="Modele")
            conf_in = gr.Slider(minimum=0.05, maximum=0.95, value=0.25, step=0.01, label="Seuil de confiance")
            max_boxes_in = gr.Slider(minimum=5, maximum=200, value=80, step=1, label="Nombre max de boites affichees")
            run_btn = gr.Button("Lancer l'analyse")

        with gr.Column(scale=1):
            image_out = gr.Image(type="pil", label="Image annotee")
            table_out = gr.Dataframe(headers=["#", "Label", "Score", "BBox xyxy"], datatype=["str", "str", "str", "str"], label="Detections")
            summary_out = gr.Textbox(label="Resume")

    run_btn.click(
        fn=predict,
        inputs=[image_in, model_in, conf_in, max_boxes_in],
        outputs=[image_out, table_out, summary_out],
    )


if __name__ == "__main__":
    demo.launch()
