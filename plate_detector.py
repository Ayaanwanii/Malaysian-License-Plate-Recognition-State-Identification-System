"""
License Plate Detection Module - v3
Uses morphological operations + edge detection (NO Haar Cascade, NO TensorFlow, NO pattern matching)
Regions without character-shaped content are rejected outright (not just penalised).
"""

import cv2
import numpy as np


def preprocess_image(image):
    """Preprocessing pipeline with adaptive enhancement."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # CLAHE for adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(enhanced, 11, 17, 17)
    
    return gray, denoised


def detect_edges_multi_scale(image):
    """Multi-scale edge detection for robust plate boundary finding."""
    edges1 = cv2.Canny(image, 50, 200)
    
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges2 = cv2.Canny(blurred, 80, 200)
    
    combined = cv2.bitwise_or(edges1, edges2)
    
    # Morphological closing to connect nearby edges into plate-shaped blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    return opened


def count_character_blobs(region_gray):
    """Count connected components shaped like plate characters.

    Both polarities are tried (dark-on-light and light-on-dark plates) and the
    better count is returned. A thin border is trimmed first so the plate frame
    itself is not counted as a character.
    """
    h, w = region_gray.shape[:2]
    my, mx = max(1, int(h * 0.08)), max(1, int(w * 0.04))
    inner = region_gray[my:h - my, mx:w - mx]
    if inner.size == 0 or inner.shape[0] < 5 or inner.shape[1] < 10:
        return 0
    ih, iw = inner.shape[:2]
    _, binary = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best = 0
    for mask in (binary, cv2.bitwise_not(binary)):
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        count = 0
        for i in range(1, n_labels):
            bw, bh, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
            # character: 25-95% of plate height (double-row plates ~40%), narrow, not a solid block
            if not (0.25 * ih <= bh <= 0.95 * ih):
                continue
            if bw > 0.35 * iw or bw < 2:
                continue
            if area > 0.9 * bw * bh:      # filled rectangle = not a character
                continue
            count += 1
        best = max(best, count)
    return best


def has_text_like_content(gray_image, bbox):
    """Check whether a region contains text-like content (characters with strong edges).
    Filters out plain surfaces such as bumpers, lights, walls and grilles.
    Returns (has_text, text_score)."""
    x, y, w, h = bbox
    region = gray_image[y:y+h, x:x+w]

    if region.size == 0 or region.shape[0] < 5 or region.shape[1] < 10:
        return False, 0.0

    # 1. Edge density - characters produce many edges
    edges = cv2.Canny(region, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size

    # 2. Vertical edge strength - characters have vertical strokes
    sobelx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
    vert_density = np.mean(np.abs(sobelx)) / 255.0

    # 3. Contrast - plate text creates high local contrast
    local_std = np.std(region.astype(float))

    # 4. Character-shaped connected components
    n_chars = count_character_blobs(region)

    # Hard requirements: a plate must have several characters and real edges
    if not (3 <= n_chars <= 15) or edge_density < 0.04 or local_std < 25:
        return False, 0.0

    text_score = 0.0
    text_score += 0.25 if edge_density > 0.05 else 0.0
    text_score += 0.15 if edge_density > 0.10 else 0.0
    text_score += 0.15 if vert_density > 0.02 else 0.0
    text_score += 0.15 if local_std > 40 else 0.0
    text_score += 0.10 if local_std > 60 else 0.0
    text_score += 0.20 if 4 <= n_chars <= 10 else 0.10

    return text_score >= 0.4, text_score


def score_plate_candidate(cnt, img_shape, gray=None):
    """Score a contour on how likely it is to be a license plate."""
    h, w = img_shape[:2]
    img_area = h * w
    
    area = cv2.contourArea(cnt)
    
    # Area filter
    area_ratio = area / img_area
    if area_ratio < 0.002 or area_ratio > 0.15:
        return 0, None
    
    x, y, bw, bh = cv2.boundingRect(cnt)
    
    if bw < 40 or bh < 12:
        return 0, None
    
    aspect = bw / bh
    
    # Malaysian plates: single-row ~3.0-5.5, double-row ~1.3-2.5
    if aspect < 1.2 or aspect > 6.5:
        return 0, None
    
    # Rectangularity
    extent = area / (bw * bh)
    if extent < 0.45:
        return 0, None
    
    # Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    if solidity < 0.6:
        return 0, None
    
    # Polygon approximation
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    n_vertices = len(approx)
    
    # Text content validation (if gray image provided)
    text_score_val = 0.0
    if gray is not None:
        has_text, text_score_val = has_text_like_content(gray, (x, y, bw, bh))
        if not has_text:
            # No characters -> not a plate. Reject outright: the geometric scores
            # alone can reach ~0.67, so a mere penalty would still pass the 0.35 cut-off.
            return 0, None
    
    # Score components
    if 2.5 <= aspect <= 5.0:
        aspect_score = 1.0 - abs(aspect - 3.8) / 3.8
    elif 1.3 <= aspect <= 2.5:
        aspect_score = 0.7
    else:
        aspect_score = 0.3
    
    extent_score = extent
    solidity_score = solidity
    
    if 4 <= n_vertices <= 6:
        vertex_score = 1.0
    elif n_vertices < 4:
        vertex_score = 0.5
    else:
        vertex_score = max(0.2, 1.0 - (n_vertices - 6) * 0.1)
    
    size_score = min(1.0, area_ratio / 0.01)
    
    # Combined score WITH text density component
    score = (aspect_score * 0.20 +
             extent_score * 0.15 +
             solidity_score * 0.15 +
             vertex_score * 0.10 +
             size_score * 0.10 +
             text_score_val * 0.30)  # 30% weight on text content!
    
    info = {
        'contour': cnt,
        'bbox': (x, y, bw, bh),
        'aspect': aspect,
        'area': area,
        'extent': extent,
        'solidity': solidity,
        'n_vertices': n_vertices,
        'text_score': text_score_val,
        'score': score,
    }
    return score, info


def detect_plate_adaptive_threshold(gray, img_shape):
    """Detection using adaptive thresholding + morphology."""
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 19, 9)
    
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_h)
    
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    vertical = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, kernel_v)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    dilated = cv2.dilate(vertical, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for cnt in contours:
        score, info = score_plate_candidate(cnt, img_shape, gray)
        if score > 0.35:
            candidates.append(info)
    return candidates


def detect_plate_color_based(image, img_shape, gray):
    """Detect plates using color segmentation (white AND black plates)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    candidates = []
    
    # White plate regions
    lower_white = np.array([0, 0, 170])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Black plate regions (white text on black background)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 60, 80])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    for mask in [white_mask, black_mask]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            score, info = score_plate_candidate(cnt, img_shape, gray)
            if score > 0.35:
                candidates.append(info)
    
    return candidates


def non_max_suppression(candidates, overlap_thresh=0.3):
    """Remove overlapping plate candidates using NMS."""
    if len(candidates) == 0:
        return []
    
    candidates.sort(key=lambda c: c['score'], reverse=True)
    
    picked = []
    for i, cand in enumerate(candidates):
        x1, y1, w1, h1 = cand['bbox']
        keep = True
        for j in picked:
            x2, y2, w2, h2 = candidates[j]['bbox']
            xa = max(x1, x2)
            ya = max(y1, y2)
            xb = min(x1 + w1, x2 + w2)
            yb = min(y1 + h1, y2 + h2)
            
            inter = max(0, xb - xa) * max(0, yb - ya)
            union = w1 * h1 + w2 * h2 - inter
            iou = inter / union if union > 0 else 0
            # A box mostly inside another (e.g. one half of the plate) is a duplicate too
            containment = inter / min(w1 * h1, w2 * h2) if min(w1 * h1, w2 * h2) > 0 else 0

            if iou > overlap_thresh or containment > 0.6:
                keep = False
                break
        if keep:
            picked.append(i)
    
    return [candidates[i] for i in picked]


def detect_plates(image):
    """Main detection function combining multiple strategies."""
    gray, denoised = preprocess_image(image)
    
    all_candidates = []
    
    # Strategy 1: Edge-based detection
    edges = detect_edges_multi_scale(denoised)
    contours1, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours1:
        score, info = score_plate_candidate(cnt, image.shape, gray)
        if score > 0.35:
            all_candidates.append(info)
    
    # Strategy 2: Adaptive threshold-based detection
    candidates2 = detect_plate_adaptive_threshold(gray, image.shape)
    all_candidates.extend(candidates2)
    
    # Strategy 3: Color-based detection (white + black plates)
    candidates3 = detect_plate_color_based(image, image.shape, gray)
    all_candidates.extend(candidates3)
    
    # NMS to remove duplicates
    final = non_max_suppression(all_candidates, overlap_thresh=0.3)
    
    # Return top 2 candidates
    return final[:2]


def extract_plate_region(image, candidate, padding=10):
    """Extract and enhance the plate region for OCR."""
    x, y, w, h = candidate['bbox']
    
    img_h, img_w = image.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)
    
    plate_img = image[y1:y2, x1:x2]
    
    if plate_img.size == 0 or plate_img.shape[0] < 5 or plate_img.shape[1] < 10:
        return None
    
    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Resize to standard height
    target_h = 80
    scale = target_h / plate_gray.shape[0]
    target_w = int(plate_gray.shape[1] * scale)
    if target_w < 20:
        return None
    plate_resized = cv2.resize(plate_gray, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    plate_enhanced = clahe.apply(plate_resized)
    
    # Sharpen
    kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    plate_sharp = cv2.filter2D(plate_enhanced, -1, kernel_sharp)
    
    # Otsu threshold
    _, plate_binary = cv2.threshold(plate_sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Also try inverted binary (for white-on-black plates)
    plate_inv = cv2.bitwise_not(plate_binary)
    
    return plate_img, plate_resized, plate_binary, plate_inv
