import subprocess
import json
import cv2
import os
import re

# Tokens that are map chrome / OCR noise, not titles or legends
_NOISE_TOKENS = (
    "GrADS", "Highcharts", "CENAOS", "MET",
    "22N", "21N", "20N", "19N", "18N", "17N",
    "89°", "88°", "87°", "86°", "85°", "84°", "83°",
)

# Phrases that strongly signal legend / descriptive map text
_LEGEND_HINTS = (
    "pronostico", "pronóstico", "anomalia", "anomalía", "anomaly",
    "lluvia", "temperatura", "norma", "climatica", "climática",
    "nino", "niño", "enso", "sst", "region", "región",
    "models", "modelos", "legend", "leyenda",
    "acumulada", "mm", "iri",
)


def clean_title(text):
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 3]
    header_parts = []
    
    # Check for primary variable keywords and date subheaders
    for line in lines:
        if any(x in line for x in _NOISE_TOKENS):
            continue
        # Strip logo / seal text
        cleaned = re.sub(r'(@|\bGOBIERNO\b|\bREPÚBLICA\b|\bREPUBLICA\b|\bHONDURAS\b|\bCENAOS\b|\bCOPECO\b).*', '', line, flags=re.I).strip()
        cleaned = re.sub(r'[\"\'~_–—|;:{}()]+', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip(' ,.-')

        if len(cleaned) < 4:
            continue

        upper = cleaned.upper()
        # 1. Variable name candidate
        if any(w in upper for w in [
            'REGISTRO', 'PRONOSTICO', 'PRONÓSTICO', 'NORMA CLIMATICA', 'NORMA CLIMÁTICA',
            'ANOMALIA', 'ANOMALÍA', 'PRECIPITACION', 'PRECIPITACIÓN', 'TEMPERATURA',
            'LLUVIA', 'ENSO', 'SST', 'HURRICANE', 'SEASON'
        ]):
            if not any(cleaned in existing or existing in cleaned for existing in header_parts):
                header_parts.append(cleaned)
        # 2. Date or period candidate (e.g. '01 AL 12 DE JULIO DEL 2026', 'JULIO 2026', 'AGOSTO 2026')
        elif any(w in upper for w in [
            'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE',
            'ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 'DEL 2026', '2026'
        ]):
            # Try to extract clean date pattern like '01 AL 12 DE JULIO DEL 2026' or 'JULIO 2026'
            m_date = re.search(r'(\d{1,2}\s+AL\s+\d{1,2}\s+DE\s+[A-ZÁÉÍÓÚÜÑ]+\s+DEL\s+\d{4}|[A-ZÁÉÍÓÚÜÑ]+\s+\d{4})', upper)
            if m_date:
                d_str = m_date.group(1)
                if not any(d_str in existing for existing in header_parts):
                    header_parts.append(d_str)
            elif len(cleaned) >= 8:
                if not any(cleaned in existing or existing in cleaned for existing in header_parts):
                    header_parts.append(cleaned)

    if header_parts:
        return " - ".join(header_parts)

    clean_lines = []
    for line in lines:
        if any(x in line for x in _NOISE_TOKENS):
            continue
        cleaned = re.sub(r'(@|\bGOBIERNO\b|\bREPÚBLICA\b|\bREPUBLICA\b|\bHONDURAS\b|\bCENAOS\b|\bCOPECO\b).*', '', line, flags=re.I).strip()
        if len(cleaned) < 8:
            continue
        clean_lines.append(cleaned)

    if not clean_lines:
        return "Unknown Title"

    longest = max(clean_lines, key=len)
    return longest.replace('"', "").replace("|", "").strip()


def extract_legend_lines(text: str):
    """Filter OCR text for long legends and descriptive written labels.

    Keeps lines like "Pronostico SST en la region Nino 3.4, IRI" while
    dropping short noise, symbol-heavy garbage, and map chrome.
    """
    kept = []
    seen = set()

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue

        if any(tok in line for tok in _NOISE_TOKENS):
            continue

        letters = sum(c.isalpha() for c in line)
        if letters < 10:
            continue

        # Drop dashed axis ticks / OCR streak garbage (e.g. "-.-.-.--- eee nee...")
        if re.search(r"[-.]{4,}", line):
            continue
        if (line.count("-") + line.count(".")) > max(3, len(line) // 5):
            continue

        # Prefer mostly readable characters (letters, digits, spaces, light punctuation)
        punct = set(",;:%()°/_+'")
        readable = sum(c.isalnum() or c.isspace() or c in punct or c in ".-" for c in line)
        if readable / len(line) < 0.75:
            continue

        words = re.findall(r"[A-Za-zÁÉÍÓÚÜáéíóúüÑñ0-9]+", line)
        if len(words) < 2:
            continue

        # Require real-looking words (filters "eee nee enn nn nnn")
        real_words = [w for w in words if len(w) >= 3 and not re.fullmatch(r"(.)\1+", w)]
        if len(real_words) < 2:
            continue

        lower = line.lower()
        has_hint = any(h in lower for h in _LEGEND_HINTS)
        is_long_prose = len(line) >= 28 and len(real_words) >= 4
        is_mid_label = len(line) >= 18 and len(real_words) >= 3 and letters >= 14

        if not (is_long_prose or (is_mid_label and has_hint) or (has_hint and len(line) >= 22)):
            continue

        # Normalize light OCR prefix junk like "_. " or "__ "
        cleaned = re.sub(r"^[\W_]{1,4}\s*", "", line).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        kept.append(cleaned)

    return kept


def run_ocr(dataset_path: str, output_titles_path: str, raw_dir: str = "raw"):
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found. Run extract first.")
        return

    with open(dataset_path, "r") as f:
        data = json.load(f)["data"]

    sources = set(row["source"] for row in data)
    title_map = {}
    os.makedirs(raw_dir, exist_ok=True)
    fullpage_done = set()
    legend_accum = {}  # page_dir -> list of OCR text chunks

    for src in sorted(list(sources)):
        parts = src.split(".png_")
        if len(parts) != 2:
            continue
        page_stem = parts[0]  # e.g. full_page-02
        img_name = f"report/debug_{page_stem}_{parts[1]}.png"
        side = parts[1]

        img = cv2.imread(img_name)
        if img is None:
            print(f"Warning: Could not load {img_name}")
            continue

        h, w = img.shape[:2]
        # Expand crop region to capture full title text (top 30%, left edge included)
        crop = img[0:int(h * 0.3), 0:w]
        crop_path = "/tmp/temp_crop.png"
        cv2.imwrite(crop_path, crop)

        # Half-panel OCR archived page-by-page under raw/
        panel_res = subprocess.run(
            ["tesseract", img_name, "stdout"],
            capture_output=True,
            text=True,
        )
        try:
            page_num = int(page_stem.split("-")[1])
        except (IndexError, ValueError):
            page_num = None

        if page_num is not None:
            page_dir = os.path.join(raw_dir, f"page-{page_num:02d}")
        else:
            page_dir = os.path.join(raw_dir, page_stem)
        os.makedirs(page_dir, exist_ok=True)

        # Once per page: OCR full page into fullpage.txt
        if page_stem not in fullpage_done:
            full_page_img = f"report/{page_stem}.png"
            fullpage_text = ""
            if os.path.exists(full_page_img):
                fullpage_res = subprocess.run(
                    ["tesseract", full_page_img, "stdout"],
                    capture_output=True,
                    text=True,
                )
                fullpage_text = fullpage_res.stdout
                fullpage_path = os.path.join(page_dir, "fullpage.txt")
                with open(fullpage_path, "w", encoding="utf-8") as ff:
                    ff.write(fullpage_text)
                print(f"{page_stem} -> {fullpage_path}")
            else:
                print(f"Warning: Full page image not found: {full_page_img}")

            legend_accum[page_dir] = [fullpage_text]
            fullpage_done.add(page_stem)

        if page_dir in legend_accum:
            legend_accum[page_dir].append(panel_res.stdout)

        raw_path = os.path.join(page_dir, f"{side}.txt")
        with open(raw_path, "w", encoding="utf-8") as rf:
            rf.write(panel_res.stdout)

        # Title crop OCR feeds cleaned titles.json
        res = subprocess.run(
            ["tesseract", crop_path, "stdout"],
            capture_output=True,
            text=True,
        )
        title = clean_title(res.stdout)

        if title == "Unknown Title":
            if side == "left":
                title = "Pronostico de anomalias de Lluvia"
            else:
                title = "Pronostico de anomalias de Temperatura"

        if page_num is not None:
            title_map[src] = f"{title} (Page {page_num})"
        else:
            title_map[src] = title

        print(f"{src} -> {title_map[src]} [raw: {raw_path}]")

    # Write legend.txt per page from fullpage + panel OCR
    for page_dir, chunks in legend_accum.items():
        legend_lines = extract_legend_lines("\n".join(chunks))
        legend_path = os.path.join(page_dir, "legend.txt")
        with open(legend_path, "w", encoding="utf-8") as lf:
            lf.write("\n".join(legend_lines))
            if legend_lines:
                lf.write("\n")
        print(f"legend -> {legend_path} ({len(legend_lines)} lines)")

    os.makedirs(os.path.dirname(output_titles_path) or ".", exist_ok=True)
    with open(output_titles_path, "w") as f:
        json.dump(title_map, f, indent=2)
