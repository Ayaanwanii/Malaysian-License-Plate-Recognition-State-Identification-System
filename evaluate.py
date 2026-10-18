"""
Ground-truth evaluation
=======================
Compares predicted plate boxes with the dataset's annotations.

Supported annotation formats (Roboflow exports), detected automatically:
  - COCO JSON : train/_annotations.coco.json
  - YOLO TXT  : train/labels/<image>.txt  (images in train/images/)
  - Pascal VOC: train/<image>.xml         (next to each image)

Optional text labels for recognition accuracy:
  - train/plate_texts.csv  with columns: filename,text   (e.g. 175_jpg.rf.xxx.jpg,JSK3324)

Metrics
  - Detection: precision, recall, F1 at IoU >= 0.5, mean IoU of matched boxes
  - Recognition (only if plate_texts.csv exists): exact-match accuracy and
    character error rate (CER) of the best-matching plate per image
"""

import csv
import json
import os
import xml.etree.ElementTree as ET

import cv2

from lpr_service import TRAIN_DIR, LPRService

IOU_THRESHOLD = 0.5


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    xa, ya = max(ax, bx), max(ay, by)
    xb, yb = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(train_dir=TRAIN_DIR):
    """Return ({filename: [(x, y, w, h), ...]} in original pixel coords, format name) or ({}, None)."""
    # COCO
    coco_path = os.path.join(train_dir, '_annotations.coco.json')
    if os.path.isfile(coco_path):
        with open(coco_path) as f:
            coco = json.load(f)
        id_to_name = {img['id']: img['file_name'] for img in coco.get('images', [])}
        gt = {name: [] for name in id_to_name.values()}
        for ann in coco.get('annotations', []):
            name = id_to_name.get(ann['image_id'])
            if name is not None and ann.get('bbox'):
                x, y, w, h = ann['bbox']
                gt[name].append((int(x), int(y), int(w), int(h)))
        return gt, 'COCO'

    # YOLO
    label_dir = os.path.join(train_dir, 'labels')
    if os.path.isdir(label_dir):
        image_dir = LPRService.get_image_dir(train_dir)
        gt = {}
        for fname in os.listdir(image_dir):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            img = cv2.imread(os.path.join(image_dir, fname))
            if img is None:
                continue
            H, W = img.shape[:2]
            boxes = []
            label_path = os.path.join(label_dir, stem + '.txt')
            if os.path.isfile(label_path):
                with open(label_path) as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        coords = list(map(float, parts[1:]))
                        if len(coords) == 4:            # cx cy w h
                            cx, cy, bw, bh = coords
                            boxes.append((int((cx - bw / 2) * W), int((cy - bh / 2) * H),
                                          int(bw * W), int(bh * H)))
                        else:                           # polygon x1 y1 x2 y2 ...
                            xs, ys = coords[0::2], coords[1::2]
                            boxes.append((int(min(xs) * W), int(min(ys) * H),
                                          int((max(xs) - min(xs)) * W), int((max(ys) - min(ys)) * H)))
            gt[fname] = boxes
        return gt, 'YOLO'

    # Pascal VOC
    if os.path.isdir(train_dir):
        xmls = [f for f in os.listdir(train_dir) if f.lower().endswith('.xml')]
        if xmls:
            gt = {}
            for x in xmls:
                root = ET.parse(os.path.join(train_dir, x)).getroot()
                fname = root.findtext('filename') or os.path.splitext(x)[0] + '.jpg'
                boxes = []
                for obj in root.iter('object'):
                    bb = obj.find('bndbox')
                    x1, y1 = float(bb.findtext('xmin')), float(bb.findtext('ymin'))
                    x2, y2 = float(bb.findtext('xmax')), float(bb.findtext('ymax'))
                    boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
                gt[fname] = boxes
            return gt, 'VOC'

    return {}, None


def load_text_labels(train_dir=TRAIN_DIR):
    """Return {filename: text} from plate_texts.csv, or {}."""
    path = os.path.join(train_dir, 'plate_texts.csv')
    if not os.path.isfile(path):
        return {}
    with open(path, newline='') as f:
        return {row['filename']: row['text'].upper().replace(' ', '')
                for row in csv.DictReader(f) if row.get('filename')}


def _edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def evaluate_results(results, ground_truth, text_labels=None, iou_threshold=IOU_THRESHOLD):
    """Compare pipeline results with ground truth. Returns a metrics dict, or None if
    none of the processed images have annotations."""
    tp = fp = fn = 0
    ious = []
    exact, cer_sum, n_text = 0, 0.0, 0
    evaluated = 0

    for r in results:
        name = os.path.basename(str(r.get('image_path', '')))
        if name not in ground_truth:
            continue
        evaluated += 1
        scale = r.get('scale', 1.0) or 1.0
        # predictions are in resized coords -> back to original coords
        preds = [(tuple(int(v / scale) for v in d['bbox']), d) for d in r['detections']]
        gts = list(ground_truth[name])

        matched_gt = set()
        for box, _ in sorted(preds, key=lambda p: -p[1]['score']):
            best_j, best_iou = None, 0.0
            for j, g in enumerate(gts):
                if j in matched_gt:
                    continue
                iou = _iou(box, g)
                if iou > best_iou:
                    best_j, best_iou = j, iou
            if best_j is not None and best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_j)
                ious.append(best_iou)
            else:
                fp += 1
        fn += len(gts) - len(matched_gt)

        if text_labels and name in text_labels:
            truth = text_labels[name]
            read = [d['plate_text'].replace(' ', '') for _, d in preds]
            best = min(read, key=lambda t: _edit_distance(t, truth)) if read else ''
            n_text += 1
            exact += int(best == truth)
            cer_sum += _edit_distance(best, truth) / max(len(truth), 1)

    if evaluated == 0:
        return None
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {
        'images_evaluated': evaluated,
        'iou_threshold': iou_threshold,
        'true_positives': tp, 'false_positives': fp, 'false_negatives': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'mean_iou': sum(ious) / len(ious) if ious else 0.0,
    }
    if n_text:
        out.update({'text_images_evaluated': n_text,
                    'exact_match_accuracy': exact / n_text,
                    'character_error_rate': cer_sum / n_text})
    return out
