"""
State Identification Module for Malaysian License Plates
=========================================================
1. Cleans OCR output and corrects common character confusions by position
   (prefix letters vs. number digits, e.g. 0->O in the prefix, O->0 in the number).
2. Checks that the text has the structure of a Malaysian plate
   (1-3 letter prefix, 1-4 digit number, optional single-letter suffix),
   so brand badges and signs ("TOYOTA", "HONDA") are not reported as plates.
3. Maps the prefix to a registration state.

Text is recognised elsewhere by EasyOCR (PyTorch-based, NOT TensorFlow).
"""

import re

# ─── Prefix → state ───
# Single letters cover every state; multi-letter entries are only needed where
# the meaning differs from the first letter (KV = Langkawi rather than mainland Kedah).
# One consistent name per state, so charts do not split a state into two bars.
MALAYSIAN_STATE_PREFIXES = {
    # Peninsular Malaysia
    'A': 'Perak', 'B': 'Selangor', 'C': 'Pahang', 'D': 'Kelantan',
    'J': 'Johor', 'K': 'Kedah', 'M': 'Malacca', 'N': 'Negeri Sembilan',
    'P': 'Penang', 'R': 'Perlis', 'T': 'Terengganu',
    # Federal Territories
    'W': 'Kuala Lumpur', 'V': 'Kuala Lumpur', 'F': 'Putrajaya', 'L': 'Labuan',
    # East Malaysia
    'S': 'Sabah', 'Q': 'Sarawak',
    # Special prefixes
    'H': 'Kuala Lumpur',   # taxis
    'Z': 'Military',
    'KV': 'Kedah',         # Langkawi
}

# Plate type by prefix (anything not listed is 'Standard')
PLATE_TYPES = {
    'H': 'Taxi',
    'Z': 'Military',
    'KV': 'Langkawi',
}

# Special-series plates identified by a keyword at the start of the plate
SPECIAL_SERIES = {
    'PUTRAJAYA': 'Putrajaya',
    'PROTON': 'Special Series - Proton',
    'PERODUA': 'Special Series - Perodua',
    'MALAYSIA': 'Special Series - Malaysia',
    'PATRIOT': 'Special Series - Patriot',
    'TENTERA': 'Military',
    'POLIS': 'Police',
    'DIPLOMATIC': 'Diplomatic',
}

# Characters EasyOCR commonly confuses
DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
LETTER_TO_DIGIT = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1',
                   'Z': '2', 'S': '5', 'G': '6', 'B': '8', 'T': '7'}

MAX_PREFIX, MAX_DIGITS, MAX_SUFFIX = 3, 4, 1
MAX_SUBSTITUTIONS = 2


def clean_ocr_text(text):
    """Upper-case, keep only letters/digits/spaces, collapse spaces."""
    if not text:
        return ""
    text = text.upper().strip()
    text = re.sub(r'[^A-Z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def _force(ch, want_letter):
    """Return ch as the wanted class (letter/digit), its substitution cost, or (None, None)."""
    if ch.isalpha() == want_letter:
        return ch, 0
    table = DIGIT_TO_LETTER if want_letter else LETTER_TO_DIGIT
    return (table[ch], 1) if ch in table else (None, None)


def correct_plate_text(text):
    """Position-aware correction of OCR confusions.

    Malaysian layout: PREFIX (1-3 letters) + NUMBER (1-4 digits) + optional SUFFIX (1 letter).
    Every possible split of the string into those three parts is tried; characters
    are forced to the class their position requires (e.g. '5'->'S' in the prefix,
    'O'->'0' in the number) and the split needing the fewest substitutions wins
    (ties: a known state prefix, then no suffix). At most MAX_SUBSTITUTIONS
    characters may change, and the number must be mostly genuine digits. If no split is possible the
    cleaned text is returned unchanged, and is_plausible_plate() will reject it.
    Returns the corrected compact string (no spaces).
    """
    compact = clean_ocr_text(text).replace(' ', '')
    if not compact:
        return ""
    n = len(compact)
    best = None
    for p in range(1, MAX_PREFIX + 1):
        for s in range(0, MAX_SUFFIX + 1):
            d = n - p - s
            if not (1 <= d <= MAX_DIGITS):
                continue
            out, cost, number_subs = [], 0, 0
            for i, ch in enumerate(compact):
                want_letter = i < p or i >= p + d
                fixed, c = _force(ch, want_letter)
                if fixed is None:
                    break
                out.append(fixed)
                cost += c
                if not want_letter:
                    number_subs += c
            else:
                # Guard against turning words into plates ("PROTON" -> "PRO70N"):
                # the number must be mostly genuine digits and few characters may change.
                if cost > MAX_SUBSTITUTIONS or number_subs > (d - 1) // 2:
                    continue
                candidate = ''.join(out)
                known = candidate[0] in MALAYSIAN_STATE_PREFIXES
                key = (cost, not known, s)
                if best is None or key < best[0]:
                    best = (key, candidate)
    return best[1] if best else compact


def split_plate(compact):
    """Split compact text into (prefix, number, suffix), or None if it is not plate-shaped."""
    i, n = 0, len(compact)
    while i < n and compact[i].isalpha():
        i += 1
    j = i
    while j < n and compact[j].isdigit():
        j += 1
    prefix, number, suffix = compact[:i], compact[i:j], compact[j:]
    if not (1 <= len(prefix) <= MAX_PREFIX):
        return None
    if not (1 <= len(number) <= MAX_DIGITS) or number.startswith('0'):
        return None
    if len(suffix) > MAX_SUFFIX or (suffix and not suffix.isalpha()):
        return None
    return prefix, number, suffix


def match_special_series(compact):
    """Return (keyword, label) if the plate is a special series such as 'PUTRAJAYA 1234'."""
    for keyword, label in SPECIAL_SERIES.items():
        if compact.startswith(keyword):
            rest = compact[len(keyword):]
            # keyword must be followed by a number (and optional suffix letter)
            if rest and split_plate('X' + rest) is not None:
                return keyword, label
    return None


def is_plausible_plate(text):
    """True if the (corrected) text has the structure of a Malaysian plate."""
    compact = correct_plate_text(text)
    if not compact:
        return False
    if match_special_series(compact) is not None:
        return True
    # Plate-shaped AND the prefix is a real Malaysian prefix (I, O, U, X, Y, E, G are not)
    return split_plate(compact) is not None and extract_prefix(compact) is not None


def extract_prefix(plate_text):
    """Return the state prefix of a plate-shaped text, or None."""
    parts = split_plate(correct_plate_text(plate_text))
    if parts is None:
        return None
    letters = parts[0]
    for length in (3, 2, 1):
        if len(letters) >= length and letters[:length] in MALAYSIAN_STATE_PREFIXES:
            return letters[:length]
    return None


def format_plate_text(text):
    """Corrected text in display form, e.g. 'JSK 3324' or 'WA 1234 B'."""
    compact = correct_plate_text(text)
    special = match_special_series(compact)
    if special:
        return f"{special[0]} {compact[len(special[0]):]}"
    parts = split_plate(compact)
    if parts is None:
        return clean_ocr_text(text)
    return ' '.join(p for p in parts if p)


def identify_state(plate_text):
    """Return (state, prefix, plate_type). Text that is not plate-shaped returns 'Unknown'."""
    compact = correct_plate_text(plate_text)
    if not compact:
        return 'Unknown', 'N/A', 'Unknown'

    special = match_special_series(compact)
    if special:
        return special[1], special[0], 'Special'

    prefix = extract_prefix(compact)
    if not prefix:
        return 'Unknown', 'N/A', 'Unknown'
    state = MALAYSIAN_STATE_PREFIXES[prefix]
    plate_type = PLATE_TYPES.get(prefix, PLATE_TYPES.get(prefix[0], 'Standard'))
    return state, prefix, plate_type


def format_plate_info(plate_text, state, prefix, plate_type):
    """Format plate information for display."""
    return {
        'raw_text': plate_text,
        'state': state,
        'prefix': prefix,
        'plate_type': plate_type,
    }
