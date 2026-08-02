# Specific Person Face Matcher

A small Streamlit app that lets you identify one specific person across a folder of mixed or group photos.

## How it works

1. Upload a few **reference photos** containing the target person. The photos can contain other people too.
2. The app detects every face and shows each detected face separately.
3. Select which detected faces belong to the target person.
4. Upload the mixed photos you want to search.
5. Each detected face is embedded with `Facenet512` through DeepFace and compared against the selected target reference embeddings using cosine similarity.
6. Matching photos can be downloaded as a ZIP together with a `results.csv` containing the similarity score for every detected face.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Your browser should open the local Streamlit interface.

## Using your photos

You do not need to rename or label the photos beforehand.

### Reference photos

Upload several photos where the target person's face is visible. If a reference image is a group photo, simply select the correct detected face in the app.

Using multiple reference faces with different angles and lighting generally works better than using only one image.

### Mixed photos

Upload the photos you want to search. They can contain:

- the target person
- multiple people
- only unrelated people
- group photos

The app checks every detected face independently.

## Similarity threshold

The default cosine similarity threshold is `0.70`.

This is intentionally adjustable. Similarity is **not a probability**, and the best cutoff depends on your specific photos. Start around 0.70 and inspect known examples:

- Raise the threshold if unrelated people are being marked as matches.
- Lower it slightly if clear photos of the target are being missed.

For a serious evaluation, create a small validation set containing known target and known non-target faces and choose the threshold from those results.

## Privacy

Face photos and generated biometric embeddings should be treated as sensitive data.

The repository `.gitignore` excludes common local image/data folders and embedding files. The Streamlit app processes uploaded photos in the local app session; it does not intentionally commit them to this repository.

Only use this tool with the subject's permission and in ways consistent with applicable privacy rules.
