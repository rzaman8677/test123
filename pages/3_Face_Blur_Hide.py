import io, json, os, tempfile, zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFilter, ImageFont

TYPES = ["jpg", "jpeg", "png", "webp"]


def tmp_image(data: bytes, name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(data)
    f.flush()
    f.close()
    return f.name


def load_profile(uploaded):
    with zipfile.ZipFile(io.BytesIO(uploaded.getvalue()), "r") as zf:
        meta = json.loads(zf.read("metadata.json").decode("utf-8"))
        with np.load(io.BytesIO(zf.read("target_profile.npz")), allow_pickle=False) as data:
            refs = np.asarray(data["reference_embeddings"], dtype=np.float32)
    return meta, refs


def detect_faces(image_bytes: bytes, filename: str, meta):
    path = tmp_image(image_bytes, filename)
    try:
        reps = DeepFace.represent(
            img_path=path,
            model_name=meta.get("model_name", "Facenet512"),
            detector_backend=meta.get("detector_backend", "retinaface"),
            normalization=meta.get("normalization", "base"),
            align=bool(meta.get("align", True)),
            enforce_detection=True,
        )
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    out = []
    for rep in reps:
        a = rep.get("facial_area", {})
        w, h = int(a.get("w", 0)), int(a.get("h", 0))
        if w > 0 and h > 0:
            out.append({
                "embedding": np.asarray(rep["embedding"], dtype=np.float32),
                "area": {
                    "x": int(a.get("x", 0)),
                    "y": int(a.get("y", 0)),
                    "w": w,
                    "h": h,
                },
            })
    return out


def cosine(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return -1.0 if denom == 0 else float(np.dot(a, b) / denom)


def score_faces(faces, refs):
    return [max(cosine(face["embedding"], ref) for ref in refs) for face in faces]


def box(area, image_size, scale):
    iw, ih = image_size
    x, y, w, h = area["x"], area["y"], area["w"], area["h"]
    cx, cy = x + w / 2.0, y + h / 2.0
    nw, nh = w * scale, h * scale
    x1 = max(0, int(round(cx - nw / 2.0)))
    y1 = max(0, int(round(cy - nh / 2.0)))
    x2 = min(iw, int(round(cx + nw / 2.0)))
    y2 = min(ih, int(round(cy + nh / 2.0)))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def annotate(image, faces, scores, threshold):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for i, face in enumerate(faces):
        a = face["area"]
        draw.rectangle((a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]), width=4)
        txt = f"Face {i}: {scores[i]:.3f}" + (" MATCH" if scores[i] >= threshold else "")
        draw.text((a["x"] + 4, max(0, a["y"] - 16)), txt)
    return img


def get_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_centered_text(image, text, face_box, text_color, font_size):
    if not text.strip():
        return image

    draw = ImageDraw.Draw(image)
    font = get_font(font_size)
    x1, y1, x2, y2 = face_box
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=2)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x1 + ((x2 - x1) - tw) // 2
    ty = y1 + ((y2 - y1) - th) // 2

    outline = "black" if text_color.lower() != "black" else "white"
    draw.text(
        (tx, ty),
        text,
        fill=text_color,
        font=font,
        stroke_width=2,
        stroke_fill=outline,
    )
    return image


def png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


st.set_page_config(page_title="Target Face Blur / Hide", layout="wide")
st.title("Target Face Blur / Hide")
st.caption("Upload the target profile ZIP and a photo, then blur or hide matching faces.")

profile = st.file_uploader("1. Target profile ZIP", type=["zip"])
photo = st.file_uploader("2. Photo", type=TYPES)
method = st.selectbox("3. Hide method", ["Blur", "Pixelate", "Black box", "White box"])
threshold = st.slider("Recognition threshold", 0.40, 0.95, 0.70, 0.01)
scale = st.slider("Face area size", 1.00, 1.50, 1.12, 0.01)
strength = st.slider("Blur strength", 4, 40, 18, 1, disabled=(method != "Blur"))
pixel = st.slider("Pixel size", 4, 40, 12, 1, disabled=(method != "Pixelate"))
hide_all = st.checkbox("Hide every matching instance", value=True)

st.subheader("Optional text overlay")
overlay_text = st.text_input(
    "Text to put over the hidden face",
    value="",
    placeholder="Type anything here",
)
text_col1, text_col2 = st.columns(2)
with text_col1:
    text_color = st.selectbox("Text color", ["white", "black", "red", "yellow", "blue"])
with text_col2:
    font_size = st.slider("Text size", 10, 80, 30, 2)

if st.button("Detect target and hide face", type="primary", disabled=not (profile and photo)):
    try:
        meta, refs = load_profile(profile)
        stored_threshold = float(meta.get("similarity_threshold", 0.70))
        if threshold == 0.70:
            threshold = stored_threshold

        data = photo.getvalue()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        faces = detect_faces(data, photo.name, meta)

        if not faces:
            st.error("No faces were detected in the uploaded photo.")
            st.stop()

        scores = score_faces(faces, refs)
        matches = [i for i, s in enumerate(scores) if s >= threshold]

        if not matches:
            st.warning("No detected face matched the uploaded target profile at this threshold.")
            st.image(annotate(image, faces, scores, threshold), use_container_width=True)
            st.stop()

        if not hide_all:
            matches = [max(matches, key=lambda i: scores[i])]

        edited = image.copy()
        for i in matches:
            x1, y1, x2, y2 = box(faces[i]["area"], edited.size, scale)
            face_box = (x1, y1, x2, y2)

            if method == "Blur":
                region = edited.crop(face_box).filter(ImageFilter.GaussianBlur(radius=strength))
                edited.paste(region, (x1, y1))
            elif method == "Pixelate":
                region = edited.crop(face_box)
                w, h = region.size
                small = region.resize((max(1, w // pixel), max(1, h // pixel)), Image.Resampling.BILINEAR)
                edited.paste(small.resize((w, h), Image.Resampling.NEAREST), (x1, y1))
            else:
                draw = ImageDraw.Draw(edited)
                fill = (0, 0, 0) if method == "Black box" else (255, 255, 255)
                draw.rectangle(face_box, fill=fill)

            edited = draw_centered_text(
                edited,
                overlay_text,
                face_box,
                text_color,
                font_size,
            )

        st.success(f"Processed {len(matches)} matching face(s).")
        c1, c2 = st.columns(2)
        with c1:
            st.image(image, caption="Original", use_container_width=True)
        with c2:
            st.image(edited, caption="Edited", use_container_width=True)

        with st.expander("Show recognition boxes and scores"):
            st.image(annotate(image, faces, scores, threshold), use_container_width=True)

        st.download_button(
            "Download edited image",
            png_bytes(edited),
            f"face_hidden_{Path(photo.name).stem}.png",
            "image/png",
            type="primary",
        )
    except Exception as exc:
        st.exception(exc)
