import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFilter, ImageOps

SUPPORTED_TYPES = ["jpg", "jpeg", "png", "webp"]
MAX_PROFILE_BYTES = 25 * 1024 * 1024

st.set_page_config(page_title="Target Face Replacer", layout="wide")
st.title("Target Face Replacer")
st.caption(
    "Upload a target recognition profile, a photo to search, and a replacement-face photo. "
    "The app finds the enrolled person and composites the replacement face over matching face regions."
)


def _suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"


def save_bytes_to_temp(data: bytes, filename: str) -> str:
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=_suffix(filename))
    temp.write(data)
    temp.flush()
    temp.close()
    return temp.name


def load_profile(uploaded_file):
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_PROFILE_BYTES:
        raise ValueError("Profile ZIP is unexpectedly large.")

    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
        members = set(zf.namelist())
        required = {"target_profile.npz", "metadata.json"}
        if not required.issubset(members):
            raise ValueError("This is not a compatible target-profile ZIP.")

        metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
        npz_bytes = zf.read("target_profile.npz")

    if metadata.get("format") != "specific-person-face-profile":
        raise ValueError("Unsupported profile format.")

    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as data:
        references = np.asarray(data["reference_embeddings"], dtype=np.float32)
        mean_embedding = np.asarray(data["mean_embedding"], dtype=np.float32)

    if references.ndim != 2 or references.shape[0] < 1:
        raise ValueError("Profile does not contain valid reference embeddings.")
    if mean_embedding.ndim != 1 or mean_embedding.shape[0] != references.shape[1]:
        raise ValueError("Profile embedding dimensions are inconsistent.")

    return metadata, references, mean_embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def detect_faces(image_bytes: bytes, filename: str, metadata):
    path = save_bytes_to_temp(image_bytes, filename)
    try:
        reps = DeepFace.represent(
            img_path=path,
            model_name=metadata.get("model_name", "Facenet512"),
            detector_backend=metadata.get("detector_backend", "retinaface"),
            enforce_detection=True,
            align=bool(metadata.get("align", True)),
            normalization=metadata.get("normalization", "base"),
        )
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    faces = []
    for index, rep in enumerate(reps):
        area = rep.get("facial_area", {})
        x = int(area.get("x", 0))
        y = int(area.get("y", 0))
        w = int(area.get("w", 0))
        h = int(area.get("h", 0))
        if w <= 0 or h <= 0:
            continue
        faces.append(
            {
                "index": index,
                "embedding": np.asarray(rep["embedding"], dtype=np.float32),
                "area": {"x": x, "y": y, "w": w, "h": h},
            }
        )
    return faces


def score_faces(faces, references):
    scores = []
    for face in faces:
        score = max(cosine_similarity(face["embedding"], ref) for ref in references)
        scores.append(score)
    return scores


def clamp_box(x, y, w, h, image_width, image_height):
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(image_width, int(round(x + w)))
    y2 = min(image_height, int(round(y + h)))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def expanded_face_box(area, image_size, scale):
    image_width, image_height = image_size
    x, y, w, h = area["x"], area["y"], area["w"], area["h"]
    cx = x + w / 2.0
    cy = y + h / 2.0
    new_w = w * scale
    new_h = h * scale
    return clamp_box(
        cx - new_w / 2.0,
        cy - new_h / 2.0,
        new_w,
        new_h,
        image_width,
        image_height,
    )


def crop_replacement_face(image: Image.Image, area, padding=1.18):
    box = expanded_face_box(area, image.size, padding)
    return image.crop(box)


def build_soft_oval_mask(size, feather_fraction):
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    margin_x = max(1, int(width * 0.06))
    margin_y = max(1, int(height * 0.025))
    draw.ellipse(
        (margin_x, margin_y, width - margin_x, height - margin_y),
        fill=255,
    )
    blur_radius = max(1, int(min(width, height) * feather_fraction))
    return mask.filter(ImageFilter.GaussianBlur(blur_radius))


def replace_face_region(scene, replacement_crop, area, scale, feather_fraction, opacity):
    output = scene.copy()
    target_box = expanded_face_box(area, output.size, scale)
    x1, y1, x2, y2 = target_box
    target_size = (x2 - x1, y2 - y1)

    fitted = ImageOps.fit(
        replacement_crop,
        target_size,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.48),
    ).convert("RGB")

    mask = build_soft_oval_mask(target_size, feather_fraction)
    if opacity < 1.0:
        mask = mask.point(lambda value: int(value * opacity))

    output.paste(fitted, (x1, y1), mask)
    return output


def annotate_matches(image, faces, scores, threshold):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for index, face in enumerate(faces):
        area = face["area"]
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]
        matched = scores[index] >= threshold
        label = f"Face {index}: {scores[index]:.3f}" + (" MATCH" if matched else "")
        draw.rectangle((x, y, x + w, y + h), width=4)
        draw.text((x + 4, max(0, y - 16)), label)
    return annotated


def image_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


st.header("1. Upload the target recognition profile")
profile_upload = st.file_uploader(
    "Target profile ZIP",
    type=["zip"],
    key="target_profile_zip",
)

metadata = None
references = None
profile_threshold = 0.70

if profile_upload:
    try:
        metadata, references, _ = load_profile(profile_upload)
        profile_threshold = float(metadata.get("similarity_threshold", 0.70))
        st.success(
            f"Loaded profile with {references.shape[0]} reference face(s) using "
            f"{metadata.get('model_name', 'Facenet512')}."
        )
    except Exception as exc:
        st.error(f"Could not load profile: {exc}")

st.header("2. Upload the photo to edit")
scene_upload = st.file_uploader(
    "Photo containing the enrolled person",
    type=SUPPORTED_TYPES,
    key="scene_image",
)

st.header("3. Upload the replacement face")
replacement_upload = st.file_uploader(
    "Replacement-face photo (a clear image with one main face works best)",
    type=SUPPORTED_TYPES,
    key="replacement_face",
)

st.header("4. Match and replace")
threshold = st.slider(
    "Recognition threshold",
    min_value=0.40,
    max_value=0.95,
    value=min(0.95, max(0.40, profile_threshold)),
    step=0.01,
    help="Defaults to the threshold stored in the uploaded target profile.",
)

col1, col2, col3 = st.columns(3)
with col1:
    replacement_scale = st.slider(
        "Replacement size",
        min_value=0.85,
        max_value=1.30,
        value=1.08,
        step=0.01,
    )
with col2:
    feather = st.slider(
        "Edge feathering",
        min_value=0.01,
        max_value=0.18,
        value=0.07,
        step=0.01,
    )
with col3:
    opacity = st.slider(
        "Blend opacity",
        min_value=0.50,
        max_value=1.00,
        value=0.95,
        step=0.05,
    )

replace_all = st.checkbox(
    "Replace every matching instance of the enrolled person",
    value=True,
)

ready = metadata is not None and references is not None and scene_upload and replacement_upload

if st.button("Detect target and replace face", type="primary", disabled=not ready):
    try:
        scene_bytes = scene_upload.getvalue()
        replacement_bytes = replacement_upload.getvalue()

        scene_image = Image.open(io.BytesIO(scene_bytes)).convert("RGB")
        replacement_image = Image.open(io.BytesIO(replacement_bytes)).convert("RGB")

        with st.spinner("Detecting and comparing faces..."):
            scene_faces = detect_faces(scene_bytes, scene_upload.name, metadata)
            scene_scores = score_faces(scene_faces, references)
            replacement_faces = detect_faces(replacement_bytes, replacement_upload.name, metadata)

        if not scene_faces:
            st.error("No faces were detected in the photo to edit.")
            st.stop()
        if not replacement_faces:
            st.error("No face was detected in the replacement-face photo.")
            st.stop()

        matching_indices = [
            index for index, score in enumerate(scene_scores) if score >= threshold
        ]
        if not matching_indices:
            st.warning(
                "Faces were detected, but none matched the uploaded target profile at this threshold."
            )
            st.image(
                annotate_matches(scene_image, scene_faces, scene_scores, threshold),
                caption="Detected faces and similarity scores",
                use_container_width=True,
            )
            st.stop()

        # If multiple faces exist in the replacement image, use the largest detected face.
        replacement_face = max(
            replacement_faces,
            key=lambda face: face["area"]["w"] * face["area"]["h"],
        )
        replacement_crop = crop_replacement_face(
            replacement_image,
            replacement_face["area"],
        )

        indices_to_replace = matching_indices
        if not replace_all:
            indices_to_replace = [max(matching_indices, key=lambda i: scene_scores[i])]

        edited = scene_image.copy()
        for index in indices_to_replace:
            edited = replace_face_region(
                edited,
                replacement_crop,
                scene_faces[index]["area"],
                replacement_scale,
                feather,
                opacity,
            )

        st.success(
            f"Replaced {len(indices_to_replace)} matching face(s). "
            f"Best target similarity: {max(scene_scores[i] for i in matching_indices):.3f}."
        )

        preview_left, preview_right = st.columns(2)
        with preview_left:
            st.image(scene_image, caption="Original", use_container_width=True)
        with preview_right:
            st.image(edited, caption="Edited", use_container_width=True)

        with st.expander("Show recognition boxes and scores"):
            st.image(
                annotate_matches(scene_image, scene_faces, scene_scores, threshold),
                use_container_width=True,
            )

        st.download_button(
            "Download edited image",
            data=image_to_png_bytes(edited),
            file_name=f"face_replaced_{Path(scene_upload.name).stem}.png",
            mime="image/png",
            type="primary",
        )

    except Exception as exc:
        st.exception(exc)

st.divider()
st.caption(
    "This page performs local face recognition plus image compositing. The target-profile ZIP contains biometric embeddings; "
    "keep it private and only use images you have permission to process."
)
