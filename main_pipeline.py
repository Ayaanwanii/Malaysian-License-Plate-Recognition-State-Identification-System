"""
Main License Plate Recognition Pipeline (command line)
======================================================
- NO Haar Cascade, NO TensorFlow
- Uses: OpenCV morphology, Canny edges, adaptive thresholding, HSV segmentation,
  contour analysis, EasyOCR (PyTorch-based)
- The pipeline itself lives in lpr_service.py (shared with the Streamlit app);
  this script runs it over the dataset and writes graphs, sample images, a CSV
  of results and, if annotations exist, ground-truth metrics.

Usage:
    python main_pipeline.py              # random sample of 200 images
    python main_pipeline.py --all        # whole dataset
    python main_pipeline.py --sample 50
"""

import argparse
import csv
import json
import os

import cv2

from lpr_service import LPRService, TRAIN_DIR, empty_metrics
from state_identifier import identify_state
from evaluate import load_ground_truth, load_text_labels, evaluate_results

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
GRAPHS_DIR = os.path.join(BASE_DIR, 'evaluation_graphs')

_service = None


def get_service():
    """Create the service and load EasyOCR on first use (not at import time)."""
    global _service
    if _service is None:
        print("[INFO] Initialising EasyOCR (PyTorch-based reader)...")
        _service = LPRService()
        _service.init_reader()
        print("[INFO] EasyOCR ready.\n")
    return _service


def process_single_image(img_path, max_dim=1024):
    """Process one image file through the full pipeline."""
    image = cv2.imread(img_path)
    if image is None:
        return None
    result = get_service().process_image(image, max_dim=max_dim)
    result['image_path'] = img_path
    return result


def run_pipeline(sample_size=None):
    """Run the pipeline on the dataset. Returns (results, metrics)."""
    service = get_service()
    images = service.get_dataset_images()
    if not images:
        print(f"[ERROR] No images found in: {TRAIN_DIR}")
        print("[HINT] Put the dataset images in train/ (or train/images/).")
        return [], empty_metrics()

    print(f"[INFO] Found {len(images)} images in dataset")
    if sample_size and sample_size < len(images):
        print(f"[INFO] Processing random sample of {sample_size} images (seed 42)")

    def progress(i, total):
        if i == 0 or (i + 1) % 20 == 0 or i == total:
            print(f"[PROGRESS] {min(i + 1, total)}/{total}")

    results, metrics = service.run_batch(sample_size=sample_size, progress_cb=progress)
    print(f"\n[INFO] Processing complete! {len(results)} images processed.")

    gt, fmt = load_ground_truth()
    if gt:
        gt_metrics = evaluate_results(results, gt, load_text_labels())
        if gt_metrics:
            gt_metrics['annotation_format'] = fmt
            metrics['ground_truth'] = gt_metrics
    else:
        print("[INFO] No annotation file found - ground-truth metrics skipped.")
    return results, metrics


def plot_all_graphs(metrics):
    """Save all evaluation graphs to evaluation_graphs/."""
    print("\n[INFO] Generating evaluation graphs...")
    saved = get_service().save_all_charts(metrics, GRAPHS_DIR)
    for p in saved:
        print(f"  [OK] {os.path.basename(p)}")
    if metrics.get('ground_truth'):
        with open(os.path.join(GRAPHS_DIR, 'ground_truth_metrics.json'), 'w') as f:
            json.dump(metrics['ground_truth'], f, indent=2)
        print("  [OK] ground_truth_metrics.json")


def save_sample_detections(all_results, num_samples=20):
    """Save annotated sample images (green = plate read, orange = rejected region)."""
    sample_dir = os.path.join(OUTPUT_DIR, 'sample_detections')
    os.makedirs(sample_dir, exist_ok=True)
    samples = [r for r in all_results if r['num_plates'] > 0][:num_samples]
    for i, result in enumerate(samples):
        img = LPRService.annotate_image(result['image'], result['detections'], result['rejected'])
        fname = f"detection_{i + 1:03d}_{os.path.basename(result['image_path'])}"
        cv2.imwrite(os.path.join(sample_dir, fname), img)
    print(f"  [OK] {len(samples)} sample detections saved to {sample_dir}/")


def save_results_csv(all_results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = LPRService.results_to_rows(all_results)
    path = os.path.join(OUTPUT_DIR, 'results.csv')
    if rows:
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  [OK] Results table saved to {path}")


def print_summary(metrics):
    print("\n" + "=" * 70)
    print("           MODEL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Images processed            : {metrics['total_images']}")
    print(f"  Plates read (plate-shaped)  : {metrics['total_detections']}")
    print(f"  Regions rejected            : {metrics['total_rejected_regions']}")
    print(f"  Images with a plate read    : {metrics['plate_read_rate'] * 100:.1f}%")
    print(f"  Avg plates per image        : {metrics['avg_plates_per_image']:.2f}")
    print(f"  Avg confidence score        : {metrics['avg_confidence']:.4f}")
    print(f"  Avg processing time         : {metrics['avg_processing_time']:.4f}s")
    print(f"  States identified           : {len(metrics['states_distribution'])}")

    gt = metrics.get('ground_truth')
    if gt:
        print(f"\n  ── Against ground truth ({gt['annotation_format']}, IoU >= {gt['iou_threshold']}) ──")
        print(f"    Images evaluated : {gt['images_evaluated']}")
        print(f"    Precision        : {gt['precision']:.3f}")
        print(f"    Recall           : {gt['recall']:.3f}")
        print(f"    F1               : {gt['f1']:.3f}")
        print(f"    Mean IoU         : {gt['mean_iou']:.3f}")
        if 'exact_match_accuracy' in gt:
            print(f"    Text exact match : {gt['exact_match_accuracy']:.3f}")
            print(f"    Char error rate  : {gt['character_error_rate']:.3f}")

    print("\n  ── State Distribution (Top 10) ──")
    for state, count in metrics['states_distribution'].most_common(10):
        print(f"    {state:25s} : {count}")
    print("\n  ── Plate Type Distribution ──")
    for ptype, count in metrics['plate_types_dist'].items():
        print(f"    {ptype:25s} : {count}")
    print("\n  ── Sample Recognised Plates ──")
    for text in metrics['texts_found'][:15]:
        state, prefix, ptype = identify_state(text)
        print(f"    {text:15s} -> State: {state}, Prefix: {prefix}, Type: {ptype}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Malaysian licence plate recognition")
    parser.add_argument('--sample', type=int, default=200, help="random sample size (default 200)")
    parser.add_argument('--all', action='store_true', help="process the whole dataset")
    args = parser.parse_args()

    print("=" * 70)
    print("LICENSE PLATE RECOGNITION & STATE IDENTIFICATION SYSTEM")
    print("=" * 70)
    print("\n[CONFIG] Prohibited methods NOT used: Haar Cascade, TensorFlow")
    print("[CONFIG] Methods used: CLAHE, bilateral filter, multi-scale Canny, morphology,")
    print("         adaptive threshold, HSV segmentation, contour geometry, character-blob")
    print("         check, NMS, EasyOCR (PyTorch), plate-format check, prefix-to-state lookup\n")

    results, metrics = run_pipeline(sample_size=None if args.all else args.sample)
    if not results:
        return
    print_summary(metrics)
    plot_all_graphs(metrics)
    save_sample_detections(results)
    save_results_csv(results)
    print("\n[DONE] Pipeline complete!")
    print(f"  -> Evaluation graphs: {GRAPHS_DIR}/")
    print(f"  -> Sample detections: {OUTPUT_DIR}/sample_detections/")


if __name__ == '__main__':
    main()
