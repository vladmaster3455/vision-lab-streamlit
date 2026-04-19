#!/usr/bin/env python3
from __future__ import annotations

import gc
import os
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw

ULTRA_IMPORT_ERROR: Exception | None = None
try:
    from ultralytics import RTDETR, YOLO
except Exception as exc:  # pragma: no cover
    ULTRA_IMPORT_ERROR = exc
    RTDETR = None
    YOLO = None

MMDET_IMPORT_ERROR: Exception | None = None
try:
    from mmdet.apis import DetInferencer
except Exception as exc:  # pragma: no cover
    MMDET_IMPORT_ERROR = exc
    DetInferencer = None

DEVICE = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_CACHE: Dict[str, object] = {}

MODEL_CONFIG = {
    "YOLO": os.getenv("YOLO_WEIGHTS", "yolo11n.pt"),
    "RTDETR": os.getenv("RTDETR_WEIGHTS", "rtdetr-l.pt"),
    "DINO": os.getenv("DINO_MODEL", "dino-4scale_r50_8xb2-12e_coco"),
}

VIRUS_TRAINING_NOTE = (
    "Cible virus (projet): particules virales en cryo-EM issues de CryoVirusDB "
    "(EMPIAR-11060)."
)


def clear_models(keep: str | None = None) -> None:
    to_drop = [key for key in MODEL_CACHE if key != keep]
    for key in to_drop:
        MODEL_CACHE.pop(key, None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_model(model_name: str):
    if model_name in MODEL_CACHE:
        clear_models(keep=model_name)
        return MODEL_CACHE[model_name]

    clear_models()

    if model_name == "YOLO":
        if YOLO is None:
            raise RuntimeError(
                "YOLO indisponible: import ultralytics/cv2 impossible dans cet environnement. "
                f"Detail: {ULTRA_IMPORT_ERROR}"
            )
        MODEL_CACHE[model_name] = YOLO(MODEL_CONFIG[model_name])
    elif model_name == "RTDETR":
        if RTDETR is None:
            raise RuntimeError(
                "RTDETR indisponible: import ultralytics/cv2 impossible dans cet environnement. "
                f"Detail: {ULTRA_IMPORT_ERROR}"
            )
        MODEL_CACHE[model_name] = RTDETR(MODEL_CONFIG[model_name])
    elif model_name == "DINO":
        if DetInferencer is None:
            raise RuntimeError(
                "DINO indisponible: MMDetection non charge dans cet environnement. "
                f"Detail: {MMDET_IMPORT_ERROR}"
            )
        dino_weights = os.getenv("DINO_WEIGHTS", "").strip() or None
        MODEL_CACHE[model_name] = DetInferencer(
            model=MODEL_CONFIG[model_name],
            weights=dino_weights,
            device=DEVICE,
        )
    else:
        raise ValueError(f"Modele inconnu: {model_name}")

    return MODEL_CACHE[model_name]


def predict_ultralytics(model_name: str, image: Image.Image, conf: float, max_boxes: int):
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
                round(float(score), 3),
                f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]",
            ])

    table = pd.DataFrame(rows, columns=["#", "Classe", "Confiance", "BBox"])
    summary = f"Modele: {model_name} | Device: {DEVICE} | Detections: {len(rows)}"
    return rgb, table, summary


def predict_dino(image: Image.Image, conf: float, max_boxes: int):
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

    for idx, (bbox, score) in enumerate(zip(bboxes, scores), start=1):
        score_f = float(score)
        if score_f < float(conf):
            continue

        row_id += 1
        if row_id > int(max_boxes):
            break

        x1, y1, x2, y2 = [float(v) for v in bbox]
        if idx - 1 < len(label_names):
            label_name = str(label_names[idx - 1])
        elif idx - 1 < len(labels):
            label_name = f"class_{int(labels[idx - 1])}"
        else:
            label_name = "objet"

        draw.rectangle([x1, y1, x2, y2], outline=(255, 147, 38), width=3)
        draw.text((x1 + 4, max(0, y1 - 16)), f"{label_name} {score_f:.2f}", fill=(255, 255, 255))
        rows.append([
            str(row_id),
            label_name,
            round(score_f, 3),
            f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]",
        ])

    table = pd.DataFrame(rows, columns=["#", "Classe", "Confiance", "BBox"])
    summary = f"Modele: DINO | Device: {DEVICE} | Detections: {len(rows)}"
    return rgb, table, summary


def run_prediction(image: Image.Image, model_name: str, conf: float, max_boxes: int):
    if model_name == "DINO":
        return predict_dino(image, conf, max_boxes)
    return predict_ultralytics(model_name, image, conf, max_boxes)


def build_human_summary(table: pd.DataFrame, model_name: str, conf: float) -> tuple[str, str, float, str]:
    if table.empty:
        return (
            "Aucune detection au-dessus du seuil.",
            "Essaie de baisser legerement le seuil de confiance pour recuperer plus de boites.",
            0.0,
            "Aucune",
        )

    det_count = int(len(table))
    mean_conf = float(table["Confiance"].mean())
    labels = table["Classe"].astype(str).tolist()
    top_label = Counter(labels).most_common(1)[0][0]

    short_msg = (
        f"{det_count} detection(s) trouvee(s) avec {model_name} "
        f"(seuil {conf:.2f})."
    )
    tip_msg = (
        "Lecture rapide: la classe dominante est "
        f"'{top_label}', avec une confiance moyenne de {mean_conf:.2f}."
    )
    return short_msg, tip_msg, mean_conf, top_label


def main() -> None:
    st.set_page_config(page_title="Vision Lab Streamlit", page_icon="🔬", layout="wide")

    st.title("Vision Lab - Streamlit")
    st.caption("Detection d objets avec YOLO, RTDETR et DINO")
    st.markdown(
        "### Domaine d entrainement et usage\n"
        f"- {VIRUS_TRAINING_NOTE}\n"
        "- L application analyse aussi les images non-virus (objets generiques) avec les classes natives des modeles.\n"
        "- Si tu envoies des images liees aux virus cryo-EM, utilise de preference des poids specialises pour ce domaine."
    )

    with st.sidebar:
        st.header("Parametres")
        model_name = st.selectbox("Modele", ["YOLO", "RTDETR", "DINO"], index=0)
        conf = st.slider("Seuil de confiance", min_value=0.05, max_value=0.95, value=0.25, step=0.01)
        max_boxes = st.slider("Nombre max de boites", min_value=5, max_value=200, value=80, step=1)
        st.caption("Suggestion: commence avec YOLO pour valider le pipeline, puis teste DINO.")

    uploaded = st.file_uploader("Charge une image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Image source")
            st.image(image, use_column_width=True)

        if st.button("Lancer l analyse", type="primary"):
            with st.spinner("Inference en cours..."):
                try:
                    out_image, table, summary = run_prediction(image, model_name, conf, max_boxes)
                except Exception as exc:
                    st.error(f"Erreur: {exc}")
                    return

            with c2:
                st.subheader("Image annotee")
                st.image(out_image, use_column_width=True)

            st.subheader("Resultat de l analyse")
            short_msg, tip_msg, mean_conf, top_label = build_human_summary(table, model_name, conf)

            m1, m2, m3 = st.columns(3)
            m1.metric("Detections", int(len(table)))
            m2.metric("Confiance moyenne", f"{mean_conf:.2f}")
            m3.metric("Classe dominante", top_label)

            st.success(short_msg)
            st.caption(tip_msg)
            st.caption(summary)

            st.subheader("Detections detaillees")
            if table.empty:
                st.warning("Aucune boite a afficher. Essaie un seuil plus bas.")
            else:
                table_sorted = table.sort_values(by="Confiance", ascending=False).reset_index(drop=True)
                st.dataframe(table_sorted, use_container_width=True, hide_index=True)

            if model_name in {"YOLO", "RTDETR"}:
                st.caption(
                    "Ces modeles analysent toute image. Sur des images non-virus, ils retournent "
                    "les classes generiques d objets. Pour des classes virus, utilise des poids specialises."
                )


if __name__ == "__main__":
    main()
