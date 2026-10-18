# Malaysian Licence Plate Recognition & State Identification

Classical OpenCV plate localisation + EasyOCR (PyTorch) text recognition, with
state identification from the plate prefix. **No Haar Cascade, no TensorFlow.**

## Files

| File | Purpose |
|---|---|
| `plate_detector.py` | Classical localisation: CLAHE, bilateral filter, multi-scale Canny, morphology, adaptive threshold, HSV colour masks, contour geometry, character-blob check, NMS |
| `state_identifier.py` | OCR clean-up, position-aware character correction, plate-format check, prefix-to-state lookup |
| `lpr_service.py` | The full pipeline (single implementation, shared by CLI and GUI), batch processing, charts |
| `evaluate.py` | Ground-truth evaluation: precision, recall, F1, mean IoU; optional text accuracy and character error rate |
| `main_pipeline.py` | Command-line run over the dataset: summary, graphs, sample images, results CSV |
| `app.py` | Streamlit GUI: Single Image, Batch Analysis, Dataset Gallery |

## Setup

```bash
pip install -r requirements.txt
```

Put the dataset images in a `train/` folder next to these files (or `train/images/`
for a YOLO export). EasyOCR downloads its models on the first run.

If the first run fails with an SSL certificate error on macOS, run
`/Applications/Python 3.x/Install Certificates.command` once (the code no longer
switches certificate checking off).

## Run

```bash
python main_pipeline.py            # random sample of 200 images (seed 42)
python main_pipeline.py --all      # whole dataset
python main_pipeline.py --sample 50
streamlit run app.py               # GUI
```

Outputs: `evaluation_graphs/` (8 charts + `ground_truth_metrics.json`),
`output/sample_detections/` (green = plate read, orange = rejected region),
`output/results.csv`.

## Deploy on Streamlit Community Cloud

This repository is configured to run as a Streamlit app with `app.py` as its
entry point.

1. Push the project files to GitHub, including `app.py`, `lpr_service.py`,
   `plate_detector.py`, `state_identifier.py`, `evaluate.py`, and
   `requirements.txt`.
2. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with the
   GitHub account that can access the repository.
3. Select **New app**, choose
   `Ayaanwanii/Malaysian-License-Plate-Recognition-State-Identification-System`,
   select the branch containing the project, and set the main file to
   `app.py`.
4. Deploy the app and wait for Streamlit Community Cloud to install the
   dependencies from `requirements.txt`.

The `train/` directory is intentionally ignored because image datasets and
generated outputs can be large. To analyze a dataset in the deployed app,
either commit a suitably sized `train/` directory or upload images through the
Single Image and Batch Analysis tabs. EasyOCR downloads its model files during
the first app startup, so the first launch may take longer than subsequent
launches.

Streamlit Community Cloud does not provide a persistent local filesystem for
generated results. Download CSV results from the app if they need to be kept.

## Evaluation

If the dataset includes annotations, they are found automatically:
COCO (`train/_annotations.coco.json`), YOLO (`train/labels/*.txt`) or
Pascal VOC (`train/*.xml`). Predicted boxes are matched to ground truth at
IoU ≥ 0.5 to give precision, recall, F1 and mean IoU.

For recognition accuracy, add `train/plate_texts.csv`:

```
filename,text
175_jpg.rf.GlebX7lzblRdn08cP4pW.jpg,PQS1633
```

This adds exact-match accuracy and character error rate (CER).

### Metric definitions
- **Images with a plate read**: share of images where at least one region produced
  plate-shaped text. It is not accuracy; use precision/recall for that.
- **Regions rejected**: regions whose text was not plate-shaped (badges, signs, lights).

## How text becomes a plate

1. EasyOCR reads each candidate region (colour, grey, binary and inverted versions).
2. Every split into prefix (1–3 letters) + number (1–4 digits) + optional suffix
   letter is tried; characters are forced to the class their position needs
   (`5→S` in the prefix, `O→0` in the number). The split with the fewest changes
   wins, with at most 2 changes and a number made mostly of genuine digits.
3. The prefix must be a real Malaysian prefix, so words such as TOYOTA, HONDA or
   PROTON (without a number) are rejected.
4. The prefix gives the state (e.g. `J` Johor, `W`/`V` Kuala Lumpur, `KV` Langkawi,
   Kedah, `H` taxi, Kuala Lumpur).

## Known limitations
- Short badge text that happens to be plate-shaped (e.g. `MG5`) can still pass.
- Character confusions that swap one valid letter for another (`V`↔`N`) cannot be
  corrected from the text alone.
- EasyOCR on CPU takes roughly 2–4 s per image.
