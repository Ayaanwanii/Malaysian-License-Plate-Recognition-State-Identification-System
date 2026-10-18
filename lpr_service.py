"""
LPR Service Layer
=================
Single implementation of the recognition pipeline, shared by:
  - main_pipeline.py  (command-line batch run, graphs, sample images)
  - app.py            (Streamlit GUI)

No side effects at import time: EasyOCR is only loaded when init_reader() is called.

Pipeline per image
  1. Resize to max_dim
  2. Classical plate localisation (plate_detector: CLAHE, Canny, morphology,
     adaptive threshold, HSV colour masks, contour geometry, character-blob check, NMS)
  3. EasyOCR full-image text pass to recover plates the morphology missed
     (only text that has plate structure is kept)
  4. OCR on each candidate crop (colour, grey, binary, inverted)
  5. Position-aware character correction + plate-format check
  6. State identification from the prefix
Regions whose text is not plate-shaped are returned separately as 'rejected'.
"""

import os
import time
import random
import warnings
from collections import Counter

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plate_detector import detect_plates, extract_plate_region, preprocess_image
from state_identifier import (identify_state, is_plausible_plate, format_plate_text,
                              clean_ocr_text)

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
OCR_ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def _iou_and_containment(a, b):
    """IoU and intersection-over-smaller-box for two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    xa, ya = max(ax, bx), max(ay, by)
    xb, yb = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = aw * ah + bw * bh - inter
    smaller = min(aw * ah, bw * bh)
    return (inter / union if union > 0 else 0.0,
            inter / smaller if smaller > 0 else 0.0)


def empty_metrics():
    return _aggregate([])


def _aggregate(all_results):
    """Build the metrics dict from a list of per-image results."""
    det_counts, rejected_counts, proc_times = [], [], []
    conf_scores, detect_times, ocr_times = [], [], []
    states, types_, texts_ = [], [], []

    for result in all_results:
        det_counts.append(result['num_plates'])
        rejected_counts.append(len(result.get('rejected', [])))
        proc_times.append(result['total_time'])
        for det in result['detections']:
            states.append(det['state'])
            types_.append(det['plate_type'])
            conf_scores.append(det['score'])
            texts_.append(det['plate_text'])
            detect_times.append(det['detect_time'])
            ocr_times.append(det['ocr_time'])

    n = max(len(det_counts), 1)
    return {
        'total_images': len(all_results),
        'total_detections': sum(det_counts),
        'total_rejected_regions': sum(rejected_counts),
        # share of images where at least one plate-shaped text was read
        'plate_read_rate': sum(1 for d in det_counts if d > 0) / n,
        'avg_plates_per_image': float(np.mean(det_counts)) if det_counts else 0.0,
        'avg_processing_time': float(np.mean(proc_times)) if proc_times else 0.0,
        'avg_confidence': float(np.mean(conf_scores)) if conf_scores else 0.0,
        'states_distribution': Counter(states),
        'plate_types_dist': Counter(types_),
        'detection_counts': det_counts,
        'rejected_counts': rejected_counts,
        'processing_times': proc_times,
        'confidence_scores': conf_scores,
        'texts_found': texts_,
        'detect_times': detect_times,
        'ocr_times': ocr_times,
        'states_found': states,
        'plate_types': types_,
    }


class LPRService:
    """License Plate Recognition service."""

    def __init__(self, reader=None):
        self.reader = reader

    def init_reader(self):
        """Load EasyOCR once (PyTorch-based). Downloads models on first ever run."""
        if self.reader is None:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        return self.reader

    # ─── OCR ────────────────────────────────────────────────

    def _recognize_text(self, plate_img):
        """Run EasyOCR on a plate crop. Returns (text, mean confidence)."""
        if plate_img is None or plate_img.size == 0:
            return "", 0.0
        try:
            results = self.reader.readtext(plate_img, detail=1, paragraph=False,
                                           allowlist=OCR_ALLOWLIST)
        except Exception:
            return "", 0.0
        kept = [r for r in results if r[2] > 0.2]
        if not kept:
            return "", 0.0
        # left-to-right, top-to-bottom (double-row plates: top row first)
        kept.sort(key=lambda r: (min(p[1] for p in r[0]) // 20, min(p[0] for p in r[0])))
        text = ' '.join(r[1] for r in kept).upper()
        return text, float(np.mean([r[2] for r in kept]))

    @staticmethod
    def _choose_text(options):
        """Pick the best (text, conf) reading: plate-shaped first, then confidence."""
        options = [(t, c) for t, c in options if t]
        if not options:
            return "", 0.0
        plausible = [(t, c) for t, c in options if is_plausible_plate(t)]
        pool = plausible if plausible else options
        return max(pool, key=lambda tc: (tc[1], len(tc[0])))

    # ─── Single image processing ────────────────────────────

    def process_image(self, image_bgr, max_dim=1024):
        """Process one BGR image.

        Returns dict: image (resized), scale (resized/original), detections
        (plate-shaped text only), rejected (other regions), total_time,
        detect_time, num_plates, preprocessing (gray, denoised).
        """
        if image_bgr is None:
            return None
        if self.reader is None:
            self.init_reader()

        start_time = time.time()

        image = image_bgr.copy()
        h, w = image.shape[:2]
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        # Classical localisation
        t0 = time.time()
        candidates = detect_plates(image)
        detect_time = time.time() - t0

        gray, denoised = preprocess_image(image)

        # EasyOCR full-image pass (catches plates the morphology missed)
        img_area = image.shape[0] * image.shape[1]
        try:
            full_ocr = self.reader.readtext(image, detail=1, paragraph=False,
                                            allowlist=OCR_ALLOWLIST)
        except Exception:
            full_ocr = []
        for bbox_pts, text, conf in full_ocr:
            if conf < 0.3 or not is_plausible_plate(text):
                continue
            ox, oy, ow, oh = cv2.boundingRect(np.array(bbox_pts, dtype=np.int32))
            if ow < 30 or oh < 8:
                continue
            aspect = ow / oh if oh > 0 else 0
            if not (1.0 <= aspect <= 8.0):
                continue
            if not (0.001 <= (ow * oh) / img_area <= 0.15):
                continue

            duplicate = None
            for c in candidates:
                iou, contain = _iou_and_containment((ox, oy, ow, oh), c['bbox'])
                if iou > 0.2 or contain > 0.6:
                    duplicate = c
                    break
            if duplicate is not None:
                # same plate already found: offer this reading as an extra option
                duplicate.setdefault('ocr_hints', []).append((text.upper(), float(conf)))
            else:
                candidates.append({
                    'bbox': (ox, oy, ow, oh),
                    'score': 0.5 + conf * 0.3,
                    'ocr_text': text.upper(),
                    'ocr_conf': float(conf),
                    'from_ocr': True,
                })

        detections, rejected = [], []
        for cand in candidates:
            extracted = extract_plate_region(image, cand)
            if extracted is None:
                continue
            plate_color, plate_resized, plate_binary, plate_inv = extracted

            t1 = time.time()
            if cand.get('from_ocr'):
                raw_text, ocr_conf = cand['ocr_text'], cand['ocr_conf']
            else:
                options = [self._recognize_text(v) for v in
                           (plate_color, plate_resized, plate_binary, plate_inv)]
                options += cand.get('ocr_hints', [])
                raw_text, ocr_conf = self._choose_text(options)
            ocr_time = time.time() - t1

            valid = is_plausible_plate(raw_text)
            plate_text = format_plate_text(raw_text) if valid else clean_ocr_text(raw_text)
            state, prefix, plate_type = identify_state(raw_text)

            score = (cand['score'] * 0.5 + ocr_conf * 0.3 +
                     (0.2 if state != 'Unknown' else 0.0))

            entry = {
                'bbox': cand['bbox'],
                'score': float(min(score, 1.0)),
                'plate_text': plate_text,
                'raw_text': raw_text,
                'ocr_conf': ocr_conf,
                'state': state,
                'prefix': prefix,
                'plate_type': plate_type,
                'detect_time': detect_time,
                'ocr_time': ocr_time,
                'plate_img': plate_color,
                'valid_plate': valid,
                'source': 'ocr' if cand.get('from_ocr') else 'morphology',
            }
            (detections if valid else rejected).append(entry)

        # One reading per plate: drop overlapping duplicates, keep the best score
        detections.sort(key=lambda r: r['score'], reverse=True)
        unique = []
        for d in detections:
            if all(max(_iou_and_containment(d['bbox'], u['bbox'])) <= 0.5 for u in unique):
                unique.append(d)

        return {
            'image': image,
            'scale': scale,
            'detections': unique,
            'rejected': rejected,
            'total_time': time.time() - start_time,
            'detect_time': detect_time,
            'num_plates': len(unique),
            'preprocessing': {'gray': gray, 'denoised': denoised},
        }

    # ─── Dataset helpers ────────────────────────────────────

    @staticmethod
    def get_image_dir(train_dir=TRAIN_DIR):
        """Images may sit directly in train/ (COCO/VOC export) or in train/images (YOLO)."""
        sub = os.path.join(train_dir, 'images')
        return sub if os.path.isdir(sub) else train_dir

    def get_dataset_images(self, train_dir=TRAIN_DIR):
        """Sorted list of image paths in the dataset folder (empty if missing)."""
        image_dir = self.get_image_dir(train_dir)
        if not os.path.isdir(image_dir):
            return []
        imgs = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        return sorted(imgs)

    # ─── Batch processing ───────────────────────────────────

    def run_batch(self, sample_size=None, progress_cb=None, max_dim=1024, image_paths=None):
        """Process dataset images (random sample, seed 42). Returns (results, metrics)."""
        all_images = image_paths if image_paths is not None else self.get_dataset_images()
        if sample_size and sample_size < len(all_images):
            random.seed(42)
            all_images = random.sample(all_images, sample_size)

        def load():
            for p in all_images:
                yield p, cv2.imread(p)
        return self._run(load(), len(all_images), progress_cb, max_dim)

    def run_batch_on_images(self, images, sample_size=None, progress_cb=None, max_dim=1024):
        """Process in-memory images: list of (name, image_bgr)."""
        if sample_size:
            images = images[:int(sample_size)]
        return self._run(iter(images), len(images), progress_cb, max_dim)

    def _run(self, items, total, progress_cb, max_dim):
        results = []
        for i, (name, img) in enumerate(items):
            if progress_cb:
                progress_cb(i, total)
            if img is None:
                continue
            result = self.process_image(img, max_dim=max_dim)
            if result is None:
                continue
            result['image_path'] = name
            results.append(result)
        if progress_cb:
            progress_cb(total, total)
        return results, _aggregate(results)

    @staticmethod
    def results_to_rows(results):
        """Flat rows (one per plate, or one per image with no plate) for CSV export."""
        rows = []
        for r in results:
            name = os.path.basename(str(r.get('image_path', '')))
            if not r['detections']:
                rows.append({'image': name, 'plate_text': '', 'state': '', 'prefix': '',
                             'plate_type': '', 'score': '', 'source': '',
                             'total_time_s': round(r['total_time'], 3)})
            for d in r['detections']:
                rows.append({'image': name, 'plate_text': d['plate_text'],
                             'state': d['state'], 'prefix': d['prefix'],
                             'plate_type': d['plate_type'], 'score': round(d['score'], 3),
                             'source': d['source'], 'total_time_s': round(r['total_time'], 3)})
        return rows

    # ─── Annotation ─────────────────────────────────────────

    @staticmethod
    def annotate_image(image, detections, rejected=None):
        """Draw plates (green) and optionally rejected regions (orange). Returns a copy."""
        img = image.copy()
        for det in rejected or []:
            x, y, w, h = det['bbox']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 1)
        for det in detections:
            x, y, w, h = det['bbox']
            color = (0, 200, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            label = f"{det['plate_text']} | {det['state']}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = max(y, th + 10)
            cv2.rectangle(img, (x, ty - th - 10), (x + tw + 4, ty), color, -1)
            cv2.putText(img, label, (x + 2, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return img

    # ─── Charts (return Figures; save_all_charts writes PNGs) ──

    def chart_detection_rate(self, metrics):
        fig, ax = plt.subplots(figsize=(8, 5))
        dc = metrics['detection_counts']
        labels = ['0 plates', '1 plate', '2 plates', '3+ plates']
        values = [dc.count(0), dc.count(1), dc.count(2), sum(1 for d in dc if d >= 3)]
        bars = ax.bar(labels, values, color=['#e74c3c', '#2ecc71', '#3498db', '#9b59b6'],
                      edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(val), ha='center', fontweight='bold', fontsize=12)
        ax.set_title('Plates Read per Image', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Images')
        plt.tight_layout()
        return fig

    def chart_state_distribution_pie(self, metrics):
        dist = metrics.get('states_distribution')
        if not dist:
            return None
        fig, ax = plt.subplots(figsize=(9, 6))
        items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        labels = [s[0] for s in items[:12]]
        sizes = [s[1] for s in items[:12]]
        if len(items) > 12:
            labels.append('Others')
            sizes.append(sum(s[1] for s in items[12:]))
        colors = [plt.cm.Set3(i / max(len(labels), 1)) for i in range(len(labels))]
        _, _, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors,
                                 startangle=140, pctdistance=0.85, textprops={'fontsize': 9})
        for t in autotexts:
            t.set_fontweight('bold')
        ax.set_title('State Distribution of Recognised Plates', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def chart_processing_times(self, metrics):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if metrics.get('processing_times'):
            pt = metrics['processing_times']
            axes[0].hist(pt, bins=30, color='#3498db', edgecolor='white', alpha=0.8)
            axes[0].axvline(np.mean(pt), color='red', linestyle='--', linewidth=2,
                            label=f"Mean: {np.mean(pt):.3f}s")
            axes[0].set_title('Total Processing Time per Image', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Time (seconds)')
            axes[0].set_ylabel('Frequency')
            axes[0].legend(fontsize=9)
        else:
            axes[0].axis('off')
        if metrics.get('detect_times'):
            axes[1].hist(metrics['detect_times'], bins=30, color='#2ecc71',
                         edgecolor='white', alpha=0.8, label='Detection')
            axes[1].hist(metrics['ocr_times'], bins=30, color='#e74c3c',
                         edgecolor='white', alpha=0.5, label='OCR')
            axes[1].set_title('Detection vs OCR Time', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Time (seconds)')
            axes[1].set_ylabel('Frequency')
            axes[1].legend(fontsize=9)
        else:
            axes[1].axis('off')
        plt.tight_layout()
        return fig

    def chart_confidence(self, metrics):
        scores = metrics['confidence_scores']
        if not scores:
            return None
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(scores, bins=25, color='#9b59b6', edgecolor='white', alpha=0.85)
        ax.axvline(np.mean(scores), color='red', linestyle='--', lw=2,
                   label=f'Mean: {np.mean(scores):.3f}')
        ax.axvline(np.median(scores), color='orange', linestyle='--', lw=2,
                   label=f'Median: {np.median(scores):.3f}')
        ax.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.tight_layout()
        return fig

    def chart_plate_types(self, metrics):
        dist = metrics['plate_types_dist']
        if not dist:
            return None
        fig, ax = plt.subplots(figsize=(8, 5))
        items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        names, vals = [i[0] for i in items], [i[1] for i in items]
        palette = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        bars = ax.barh(names[::-1], vals[::-1],
                       color=[palette[i % len(palette)] for i in range(len(names))][::-1],
                       edgecolor='white')
        for bar, val in zip(bars, vals[::-1]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(val), va='center', fontweight='bold')
        ax.set_title('Plate Type Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Count')
        plt.tight_layout()
        return fig

    def chart_text_lengths(self, metrics):
        texts = [t for t in metrics['texts_found'] if t]
        if not texts:
            return None
        fig, ax = plt.subplots(figsize=(8, 5))
        lengths = [len(t.replace(' ', '')) for t in texts]
        ax.hist(lengths, bins=range(0, max(lengths) + 2), color='#1abc9c',
                edgecolor='white', alpha=0.85, align='left')
        ax.axvline(np.mean(lengths), color='red', linestyle='--', lw=2,
                   label=f'Mean: {np.mean(lengths):.1f}')
        ax.set_title('Recognised Text Length Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Characters')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.tight_layout()
        return fig

    def chart_summary_dashboard(self, metrics):
        det_counts = metrics.get('detection_counts', [])
        states_dist = metrics.get('states_distribution', Counter())
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle('Model Evaluation Summary Dashboard', fontsize=16, fontweight='bold', y=1.02)

        ax = axes[0, 0]
        read = sum(1 for d in det_counts if d > 0)
        ax.bar(['Plate read', 'No plate read'], [read, len(det_counts) - read],
               color=['#2ecc71', '#e74c3c'], edgecolor='white', linewidth=2)
        ax.set_title(f"Images with a plate read: {metrics.get('plate_read_rate', 0) * 100:.1f}%",
                     fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Images')

        ax = axes[0, 1]
        top = sorted(states_dist.items(), key=lambda x: x[1], reverse=True)[:10]
        if top:
            ax.barh([s[0][:18] for s in top][::-1], [s[1] for s in top][::-1],
                    color='#3498db', edgecolor='white')
            ax.set_title('Top 10 States Identified', fontsize=11, fontweight='bold')
        else:
            ax.axis('off')

        ax = axes[1, 0]
        if metrics.get('processing_times'):
            bp = ax.boxplot(metrics['processing_times'], vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('#3498db')
            ax.set_title(f"Processing Time (Mean: {np.mean(metrics['processing_times']):.3f}s)",
                         fontsize=11, fontweight='bold')
            ax.set_ylabel('Seconds')
        else:
            ax.axis('off')

        ax = axes[1, 1]
        ax.axis('off')
        lines = [
            "=== KEY METRICS ===", "",
            f"Images processed:        {metrics.get('total_images', 0)}",
            f"Plates read:             {metrics.get('total_detections', 0)}",
            f"Regions rejected:        {metrics.get('total_rejected_regions', 0)}",
            f"Images with plate read:  {metrics.get('plate_read_rate', 0) * 100:.1f}%",
            f"Avg plates/image:        {metrics.get('avg_plates_per_image', 0):.2f}",
            f"Avg confidence:          {metrics.get('avg_confidence', 0):.3f}",
            f"Avg processing time:     {metrics.get('avg_processing_time', 0):.3f}s",
            f"States found:            {len(states_dist)}",
        ]
        gt = metrics.get('ground_truth')
        if gt:
            lines += ["", "--- vs ground truth (IoU>=0.5) ---",
                      f"Precision: {gt['precision']:.3f}  Recall: {gt['recall']:.3f}",
                      f"F1: {gt['f1']:.3f}  Mean IoU: {gt['mean_iou']:.3f}"]
        ax.text(0.02, 0.5, '\n'.join(lines), transform=ax.transAxes, fontsize=10.5,
                fontfamily='monospace', verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
        plt.tight_layout()
        return fig

    def chart_detection_trend(self, metrics):
        dc = metrics.get('detection_counts', [])
        if not dc:
            return None
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(range(len(dc)), dc, color='#3498db', alpha=0.7, linewidth=0.8)
        ax.fill_between(range(len(dc)), dc, alpha=0.3, color='#3498db')
        window = min(20, len(dc) // 5 + 1)
        if len(dc) > window:
            ma = np.convolve(dc, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(dc)), ma, color='red', linewidth=2,
                    label=f'Moving Avg (window={window})')
            ax.legend(fontsize=9)
        ax.set_title('Plates Read per Image (Over Dataset)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Plates')
        plt.tight_layout()
        return fig

    CHARTS = {
        'detection_rate': ('01_detection_rate.png', 'chart_detection_rate'),
        'state_pie': ('02_state_distribution.png', 'chart_state_distribution_pie'),
        'processing_times': ('03_processing_times.png', 'chart_processing_times'),
        'confidence': ('04_confidence_scores.png', 'chart_confidence'),
        'plate_types': ('05_plate_types.png', 'chart_plate_types'),
        'text_lengths': ('06_text_lengths.png', 'chart_text_lengths'),
        'summary_dashboard': ('07_summary_dashboard.png', 'chart_summary_dashboard'),
        'detection_trend': ('08_detection_trend.png', 'chart_detection_trend'),
    }

    def get_chart_figure(self, chart_type, metrics):
        """Return a matplotlib Figure for the requested chart type (or None)."""
        entry = self.CHARTS.get((chart_type or '').strip().lower())
        return getattr(self, entry[1])(metrics) if entry else None

    def save_all_charts(self, metrics, out_dir):
        """Write every chart as PNG. Returns the list of saved paths."""
        os.makedirs(out_dir, exist_ok=True)
        saved = []
        for key, (fname, _) in self.CHARTS.items():
            fig = self.get_chart_figure(key, metrics)
            if fig is None:
                continue
            path = os.path.join(out_dir, fname)
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)
        return saved
