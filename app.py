import csv
import io
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from deepface import DeepFace
from PIL import Image, ImageDraw

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "webp"]

st.set_page_config(page_title="Specific Person Face Matcher", layout="wide")
st.title("Specific Person Face Matcher")
st.caption(
    "Select the target person's face from a few reference photos, then scan mixed/group photos for matches. "
    "Images are processed in this app session and are not committed to GitHub."
)


def _suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"


def save_upload_to_temp(uploaded_file):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=_suffix(uploaded_file.name))
    temp.write(uploaded_file.getvalue())
    temp.flush()
    temp.close()
    return temp.name


def get_faces(uploaded_file):
    path = save_upload_to_temp(uploaded_file)
    try:
        reps = DeepFace.represent(
            img_path=path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
            normalization="base",
        )
        image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
        faces = []
        for index, rep in enumerate(reps):
            area = rep.get("facial_area", {})
            x = int(area.get("x", 0))
            y = int(area.get("y", 0))
            w = int(area.get("w", 0))
            h = int(area.get("h", 0))
            crop = image.crop((max(0, x), max(0, y), max(0, x + w), max(0, y + h)))
            faces.append(
                {
                    "index": index,
                    "embedding": np.asarray(rep["embedding"], dtype=np.float32),
                    "area": {"x": x, "y": y, "w": w, "h": h},
                    "crop": crop,
                }
            )
        return image, faces
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def draw_boxes(image: Image.Image, faces, scores=None, threshold=None):
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for i, face in enumerate(faces):
        area = face["area"]
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]
        label = f"Face {i}"
        if scores is not None:
            label += f" | {scores[i]:.3f}"
            if threshold is not None:
                label += " | MATCH" if scores[i] >= threshold else " | no match"
        draw.rectangle((x, y, x + w, y + h), width=4)
        draw.text((x + 4, max(0, y - 16)), label)
    return output


if "reference_results" not in st.session_state:
    st.session_state.reference_results = []
if "target_embeddings" not in st.session_state:
    st.session_state.target_embeddings = []
if "scan_results" not in st.session_state:
    st.session_state.scan_results = []

st.header("1. Choose the target person")
reference_uploads = st.file_uploader(
    "Upload a few reference photos. They may contain other people.",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
    key="reference_uploads",
)

if st.button("Detect faces in reference photos", disabled=not reference_uploads):
    st.session_state.reference_results = []
    st.session_state.target_embeddings = []
    for uploaded in reference_uploads:
        try:
            image, faces = get_faces(uploaded)
            st.session_state.reference_results.append(
                {"name": uploaded.name, "bytes": uploaded.getvalue(), "image": image, "faces": faces}
            )
        except Exception as exc:
            st.warning(f"Could not detect a face in {uploaded.name}: {exc}")

selected_embeddings = []
if st.session_state.reference_results:
    st.write("Select every detected face that is the target person.")
    for file_idx, item in enumerate(st.session_state.reference_results):
        st.subheader(item["name"])
        st.image(draw_boxes(item["image"], item["faces"]), use_container_width=True)
        cols = st.columns(min(4, max(1, len(item["faces"]))))
        for face_idx, face in enumerate(item["faces"]):
            with cols[face_idx % len(cols)]:
                st.image(face["crop"], caption=f"Face {face_idx}", width=180)
                chosen = st.checkbox(
                    "This is the target",
                    key=f"ref_{file_idx}_face_{face_idx}",
                )
                if chosen:
                    selected_embeddings.append(face["embedding"])

    if st.button("Build target profile", disabled=not selected_embeddings):
        st.session_state.target_embeddings = [e.copy() for e in selected_embeddings]
        st.success(f"Target profile built from {len(selected_embeddings)} selected face(s).")

if st.session_state.target_embeddings:
    st.info(f"Current target profile: {len(st.session_state.target_embeddings)} reference face(s).")

st.header("2. Scan mixed photos")
threshold = st.slider(
    "Cosine similarity threshold",
    min_value=0.40,
    max_value=0.95,
    value=0.70,
    step=0.01,
    help="Higher = stricter. Start around 0.70, then tune using obvious matches/non-matches from your own data.",
)

scan_uploads = st.file_uploader(
    "Upload photos to search. Group photos and unrelated people are fine.",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
    key="scan_uploads",
)

if st.button(
    "Scan photos",
    disabled=(not scan_uploads or not st.session_state.target_embeddings),
):
    st.session_state.scan_results = []
    progress = st.progress(0)
    refs = st.session_state.target_embeddings

    for idx, uploaded in enumerate(scan_uploads):
        try:
            image, faces = get_faces(uploaded)
            scores = []
            for face in faces:
                score = max(cosine_similarity(face["embedding"], ref) for ref in refs)
                scores.append(score)
            matched_indices = [i for i, score in enumerate(scores) if score >= threshold]
            st.session_state.scan_results.append(
                {
                    "name": uploaded.name,
                    "bytes": uploaded.getvalue(),
                    "image": image,
                    "faces": faces,
                    "scores": scores,
                    "matched_indices": matched_indices,
                }
            )
        except Exception as exc:
            st.session_state.scan_results.append(
                {
                    "name": uploaded.name,
                    "bytes": uploaded.getvalue(),
                    "error": str(exc),
                    "faces": [],
                    "scores": [],
                    "matched_indices": [],
                }
            )
        progress.progress((idx + 1) / len(scan_uploads))

if st.session_state.scan_results:
    matched_files = [r for r in st.session_state.scan_results if r.get("matched_indices")]
    st.subheader(f"Matches: {len(matched_files)} / {len(st.session_state.scan_results)} photos")

    rows = []
    for result in st.session_state.scan_results:
        if result.get("error"):
            rows.append(
                {
                    "filename": result["name"],
                    "face_index": "",
                    "similarity": "",
                    "match": False,
                    "error": result["error"],
                }
            )
            continue

        for face_idx, score in enumerate(result["scores"]):
            rows.append(
                {
                    "filename": result["name"],
                    "face_index": face_idx,
                    "similarity": round(score, 6),
                    "match": score >= threshold,
                    "error": "",
                }
            )

    for result in matched_files:
        st.write(f"**{result['name']}**")
        annotated = draw_boxes(
            result["image"], result["faces"], result["scores"], threshold=threshold
        )
        st.image(annotated, use_container_width=True)

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=["filename", "face_index", "similarity", "match", "error"],
    )
    writer.writeheader()
    writer.writerows(rows)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.csv", csv_buffer.getvalue())
        for result in matched_files:
            zf.writestr(f"matches/{result['name']}", result["bytes"])

    st.download_button(
        "Download matches + results.csv",
        data=zip_buffer.getvalue(),
        file_name="face_match_results.zip",
        mime="application/zip",
    )

st.divider()
st.caption(
    "Use only with the subject's permission. Similarity scores are model outputs, not calibrated probabilities. "
    "For better reliability, use several clear reference faces covering different angles and lighting."
)
