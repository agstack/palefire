#!/usr/bin/env python3
"""Extract polygons from a raster map image (local path or URL), resolve a country's
administrative boundary GeoJSON from the geoBoundaries registry (or a local file),
and match extracted polygons to the database regions with color, legend, and OCR data.
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import Polygon
from shapely.ops import unary_union


# Common country name / ISO2 -> ISO3 mapping (used when pycountry is not installed)
COUNTRY_TO_ISO3 = {
    "us": "USA", "usa": "USA", "united states": "USA", "united states of america": "USA",
    "hn": "HND", "hnd": "HND", "honduras": "HND",
    "nl": "NLD", "nld": "NLD", "netherlands": "NLD", "the netherlands": "NLD",
    "de": "DEU", "deu": "DEU", "germany": "DEU",
    "fr": "FRA", "fra": "FRA", "france": "FRA",
    "gb": "GBR", "uk": "GBR", "gbr": "GBR", "united kingdom": "GBR",
    "ca": "CAN", "can": "CAN", "canada": "CAN",
    "mx": "MEX", "mex": "MEX", "mexico": "MEX",
    "br": "BRA", "bra": "BRA", "brazil": "BRA",
    "es": "ESP", "esp": "ESP", "spain": "ESP",
    "it": "ITA", "ita": "ITA", "italy": "ITA",
    "pl": "POL", "pol": "POL", "poland": "POL",
    "ua": "UKR", "ukr": "UKR", "ukraine": "UKR",
    "in": "IND", "ind": "IND", "india": "IND",
    "cn": "CHN", "chn": "CHN", "china": "CHN",
    "jp": "JPN", "jpn": "JPN", "japan": "JPN",
    "au": "AUS", "aus": "AUS", "australia": "AUS",
}

US_TERRITORIES = {
    "American Samoa",
    "Commonwealth of the Northern Mariana Islands",
    "Guam",
    "Puerto Rico",
    "United States Virgin Islands",
    "District of Columbia",
}

STANDARD_COLORS_BGR = {
    "Red": (0, 0, 255),
    "Dark Red": (0, 0, 139),
    "Dark Orange": (0, 69, 255),
    "Orange": (0, 165, 255),
    "Light Orange": (0, 200, 255),
    "Yellow": (0, 255, 255),
    "Light Yellow": (153, 255, 255),
    "White": (255, 255, 255),
    "Cyan": (255, 255, 0),
    "Light Blue": (255, 128, 0),
    "Blue": (255, 0, 0),
    "Dark Blue": (139, 0, 0),
    "Light Green": (144, 238, 144),
    "Green": (0, 128, 0),
    "Dark Green": (0, 100, 0),
    "Purple": (128, 0, 128),
    "Grey": (128, 128, 128),
}


def bgr_to_color_name(bgr: list[int]) -> str:
    """Map a BGR color to the closest standard color name (matching PaleFire compiler.py)."""
    best_name = "Unknown"
    min_dist = float("inf")
    for name, ref in STANDARD_COLORS_BGR.items():
        d = sum((a - b) ** 2 for a, b in zip(bgr, ref))
        if d < min_dist:
            min_dist = d
            best_name = name
    return best_name


def resolve_iso3(country: str) -> str:
    """Resolve a country name, ISO2, or ISO3 string to an uppercase ISO3 code."""
    c_clean = country.strip()
    c_low = c_clean.lower()
    if c_low in COUNTRY_TO_ISO3:
        return COUNTRY_TO_ISO3[c_low]

    try:
        import pycountry  # type: ignore

        match = (
            pycountry.countries.get(alpha_3=c_clean.upper())
            or pycountry.countries.get(alpha_2=c_clean.upper())
            or pycountry.countries.get(name=c_clean)
        )
        if not match:
            fuzzy = pycountry.countries.search_fuzzy(c_clean)
            if fuzzy:
                match = fuzzy[0]
        if match:
            return match.alpha_3
    except Exception:
        pass

    if len(c_clean) == 3 and c_clean.isalpha():
        return c_clean.upper()

    # Fallback: RESTCountries API
    try:
        url = f"https://restcountries.com/v3.1/name/{urllib.parse.quote(c_clean)}?fields=cca3"
        req = urllib.request.Request(url, headers={"User-Agent": "SemanticCroissant/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list) and data and "cca3" in data[0]:
                return data[0]["cca3"].upper()
    except Exception:
        pass

    raise ValueError(f"Could not resolve country '{country}' to an ISO3 code.")


def fetch_geoboundaries_geojson(
    iso3: str,
    adm_level: str = "ADM1",
    cache_dir: str = "~/.cache/geoboundaries",
    simplified: bool = True,
) -> Path:
    """Fetch (and cache) administrative boundary GeoJSON from the geoBoundaries registry."""
    cache_root = Path(os.path.expanduser(cache_dir))
    cache_root.mkdir(parents=True, exist_ok=True)
    suffix = "_simplified" if simplified else ""
    cached_file = cache_root / f"{iso3}_{adm_level}{suffix}.geojson"
    if cached_file.exists() and cached_file.stat().st_size > 0:
        return cached_file

    api_url = f"https://www.geoboundaries.org/api/current/gbOpen/{iso3}/{adm_level}/"
    req = urllib.request.Request(api_url, headers={"User-Agent": "SemanticCroissant/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        meta = json.loads(resp.read().decode("utf-8"))

    gj_url = (
        meta.get("simplifiedGeometryGeoJSON") if simplified else None
    ) or meta.get("gjDownloadURL")
    if not gj_url:
        raise RuntimeError(f"geoBoundaries returned no GeoJSON URL for {iso3} {adm_level}")

    req_gj = urllib.request.Request(gj_url, headers={"User-Agent": "SemanticCroissant/1.0"})
    with urllib.request.urlopen(req_gj, timeout=60) as resp_gj:
        cached_file.write_bytes(resp_gj.read())

    return cached_file


def load_image(source: str) -> np.ndarray:
    """Load an image from a local file path or HTTP/HTTPS URL."""
    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(source, headers={"User-Agent": "SemanticCroissant/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image from URL: {source}")
        return img

    img = cv2.imread(source)
    if img is None:
        raise FileNotFoundError(f"Could not read image file: {source}")
    return img


def _ocr_legend_column(col_img: np.ndarray, psm: int = 6) -> list[str]:
    """Run PaleFire-style upscaled Tesseract OCR on a legend label column."""
    if col_img.size == 0:
        return []
    col_up = cv2.resize(col_img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    _, png_buf = cv2.imencode(".png", col_up)
    try:
        res = subprocess.run(
            ["tesseract", "stdin", "stdout", "--psm", str(psm)],
            input=png_buf.tobytes(),
            capture_output=True,
        )
        return [
            ln.strip().replace(":", "").strip()
            for ln in res.stdout.decode("utf-8", errors="replace").splitlines()
            if ln.strip()
        ]
    except Exception:
        return []


def detect_and_mask_legend(work: np.ndarray) -> list[dict]:
    """Detect framed legend boxes OR unframed vertical color-bar legends,
    extract color patches + OCR labels, and mask them out.
    """
    h, w = work.shape[:2]
    gray0 = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    _, dark0 = cv2.threshold(gray0, 80, 255, cv2.THRESH_BINARY_INV)
    cnts0, _ = cv2.findContours(dark0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    legend_bins = []
    for c in cnts0:
        x, y, cw, ch = cv2.boundingRect(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cw > 120 and ch > 120 and x > w * 0.5:
            leg = work[y : y + ch, x : x + cw]
            lb, lg, lr = leg[:, :, 0].astype(int), leg[:, :, 1].astype(int), leg[:, :, 2].astype(int)
            lsat = np.maximum(np.maximum(lb, lg), lr) - np.minimum(np.minimum(lb, lg), lr)
            lval = np.maximum(np.maximum(lb, lg), lr)
            is_patch = ((lval < 245) & (lval > 40) & (lsat > 20)).astype(np.uint8) * 255
            pcnts, _ = cv2.findContours(is_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            patches = []
            for pc in pcnts:
                px, py, pw, ph = cv2.boundingRect(pc)
                if pw > 20 and ph > 8:
                    crop = leg[py + 2 : py + ph - 2, px + 4 : px + pw - 4]
                    if crop.size > 0:
                        med_bgr = np.median(crop.reshape(-1, 3), axis=0).astype(int).tolist()
                        patches.append((py, px, pw, ph, med_bgr))
            patches.sort(key=lambda item: item[0])

            if patches:
                y_top = max(0, patches[0][0] - 4)
                y_bot = min(leg.shape[0], patches[-1][0] + patches[-1][3] + 4)
                x_right = min(p[1] for p in patches) - 4
                lines = _ocr_legend_column(leg[y_top:y_bot, 8:x_right]) if x_right > 10 else []

                for i, (_, _, _, _, bgr) in enumerate(patches):
                    lbl = lines[i] if i < len(lines) else f"Bin_{i + 1}"
                    legend_bins.append({
                        "label": lbl,
                        "bgr": bgr,
                        "hex": "#{:02x}{:02x}{:02x}".format(bgr[2], bgr[1], bgr[0]),
                    })

            work[max(0, y - 4) : min(h, y + ch + 4), max(0, x - 4) : min(w, x + cw + 4)] = 255

    # Fallback: detect unframed vertical color-bar legend (e.g. left-side temperature bar)
    if not legend_bins:
        for c in cnts0:
            x, y, cw, ch = cv2.boundingRect(c)
            if 8 <= cw <= 45 and ch > h * 0.25 and (x < w * 0.15 or x > w * 0.85):
                # Sample vertical segments inside the bar by detecting horizontal color transitions
                bar = work[y + 3 : y + ch - 3, x + 3 : x + cw - 3]
                if bar.size == 0:
                    continue
                row_med = np.median(bar, axis=1)
                diffs = np.linalg.norm(row_med[1:] - row_med[:-1], axis=1)
                # Find transition rows where color jumps
                jumps = np.where(diffs > 18)[0]
                bounds = [0]
                for j in jumps:
                    if j - bounds[-1] > 10:
                        bounds.append(int(j))
                if len(bar) - bounds[-1] > 10:
                    bounds.append(len(bar))
                if len(bounds) >= 4:
                    for i in range(len(bounds) - 1):
                        y1, y2 = bounds[i] + 3, bounds[i + 1] - 3
                        if y2 <= y1:
                            continue
                        seg_bgr = np.median(bar[y1:y2].reshape(-1, 3), axis=0).astype(int).tolist()
                        # OCR label immediately to the right of the bar (or left if bar is on right)
                        if x < w * 0.5:
                            lbl_crop = work[y + y1 - 4 : y + y2 + 4, x + cw + 2 : min(w, x + cw + 90)]
                        else:
                            lbl_crop = work[y + y1 - 4 : y + y2 + 4, max(0, x - 90) : x - 2]
                        ocr_lines = _ocr_legend_column(lbl_crop, psm=7)
                        lbl = " ".join(ocr_lines) if ocr_lines else f"Bin_{i + 1}"
                        legend_bins.append({
                            "label": lbl,
                            "bgr": seg_bgr,
                            "hex": "#{:02x}{:02x}{:02x}".format(seg_bgr[2], seg_bgr[1], seg_bgr[0]),
                        })
                    # Mask out the color bar + labels
                    mx1 = max(0, x - 4) if x < w * 0.5 else max(0, x - 95)
                    mx2 = min(w, x + cw + 95) if x < w * 0.5 else min(w, x + cw + 4)
                    work[max(0, y - 4) : min(h, y + ch + 4), mx1:mx2] = 255
                    break

    # Fallback 2: detect unframed bottom-panel legend (e.g. 2-column swatches at y > 0.68*h)
    if not legend_bins:
        y_start = int(h * 0.68)
        bot = work[y_start:, :]
        bb, bg, br = bot[:, :, 0].astype(int), bot[:, :, 1].astype(int), bot[:, :, 2].astype(int)
        bsat = np.maximum(np.maximum(bb, bg), br) - np.minimum(np.minimum(bb, bg), br)
        bval = np.maximum(np.maximum(bb, bg), br)
        is_sw = ((bval < 248) & (bval > 35) & (bsat > 25)).astype(np.uint8) * 255
        scnts, _ = cv2.findContours(is_sw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        swatches = []
        for sc in scnts:
            sx, sy, sw, sh = cv2.boundingRect(sc)
            if 18 <= sw <= 60 and 12 <= sh <= 45:
                crop = bot[sy + 3 : sy + sh - 3, sx + 3 : sx + sw - 3]
                if crop.size > 0:
                    med_bgr = np.median(crop.reshape(-1, 3), axis=0).astype(int).tolist()
                    swatches.append((sx, y_start + sy, sw, sh, med_bgr))
        if len(swatches) >= 4:
            swatches.sort(key=lambda item: (item[0] // max(1, w // 3), item[1]))
            for i, (sx, sy, sw, sh, bgr) in enumerate(swatches):
                # OCR text to the right of the swatch
                col_end = w // 2 if sx < w * 0.45 else w
                lbl_crop = work[max(0, sy - 4) : min(h, sy + sh + 18), sx + sw + 4 : min(col_end, sx + sw + 260)]
                ocr_lines = _ocr_legend_column(lbl_crop, psm=6)
                lbl = " ".join(ocr_lines) if ocr_lines else f"Bin_{i + 1}"
                legend_bins.append({
                    "label": lbl,
                    "bgr": bgr,
                    "hex": "#{:02x}{:02x}{:02x}".format(bgr[2], bgr[1], bgr[0]),
                })
            work[y_start:, :] = 255

    return legend_bins


def extract_polygons_from_image(img: np.ndarray, min_area: int = 70):
    """Segment colored map regions into Shapely polygons and return (polygons, clean_img, legend_bins)."""
    work = img.copy()
    legend_bins = detect_and_mask_legend(work)

    # Mask out light-blue ocean/water pixels (e.g. #c8ebff) to white so coastal regions don't sample water
    b0, g0, r0 = work[:, :, 0].astype(int), work[:, :, 1].astype(int), work[:, :, 2].astype(int)
    work[(b0 > 225) & (g0 > 205) & (r0 < 215)] = 255

    # Remove dark title/footer text sitting on white background
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    b, g, r = work[:, :, 0].astype(int), work[:, :, 1].astype(int), work[:, :, 2].astype(int)
    sat = np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r)
    white_bg = ((gray > 240) & (sat < 15)).astype(np.float32)
    bg_frac = cv2.boxFilter(white_bg, -1, (25, 25))
    work[(gray < 180) & (sat < 40) & (bg_frac > 0.65)] = 255

    # Inpaint dark text labels inside colored polygons so they don't split regions
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    b, g, r = work[:, :, 0].astype(int), work[:, :, 1].astype(int), work[:, :, 2].astype(int)
    sat = np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r)
    dark_text_mask = ((gray < 85) & (sat < 50)).astype(np.uint8) * 255
    dark_text_mask = cv2.dilate(dark_text_mask, np.ones((3, 3), np.uint8), iterations=1)
    clean_img = cv2.inpaint(work, dark_text_mask, 3, cv2.INPAINT_TELEA)

    clean_gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(
        clean_gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    gx = cv2.Sobel(clean_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(clean_img, cv2.CV_32F, 0, 1, ksize=3)
    color_grad = np.sqrt(np.sum(gx**2 + gy**2, axis=2))

    border = (clean_gray > 235) | (tophat > 8) | (color_grad > 80)
    border_dil = cv2.dilate(border.astype(np.uint8) * 255, np.ones((2, 2), np.uint8), iterations=1)
    region_mask = (border_dil == 0).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(region_mask, connectivity=4)
    extracted = []
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] < min_area:
            continue
        comp_mask = (labels == idx).astype(np.uint8) * 255
        comp_mask = cv2.dilate(comp_mask, np.ones((2, 2), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if len(c) < 3:
            continue
        poly = Polygon(c.reshape(-1, 2)).buffer(0)
        if poly.is_empty or poly.area < max(60, min_area - 10):
            continue
        px_vals = clean_img[labels == idx]
        med_bgr = np.median(px_vals, axis=0).astype(int).tolist()
        extracted.append({
            "id": len(extracted) + 1,
            "area": float(poly.area),
            "centroid": (float(poly.centroid.x), float(poly.centroid.y)),
            "geometry": poly,
            "color_bgr": med_bgr,
        })

    return extracted, clean_img, legend_bins


def pick_projected_crs(gdf: gpd.GeoDataFrame, iso3: str) -> str:
    """Choose an appropriate projected CRS (US Albers for USA, Lambert-93 for FRA, or UTM zone for others)."""
    if iso3 == "USA":
        return "EPSG:5070"
    if iso3 == "FRA":
        return "EPSG:2154"
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy in lon/lat
    lon_c = (bounds[0] + bounds[2]) / 2.0
    lat_c = (bounds[1] + bounds[3]) / 2.0
    zone = int((lon_c + 180) / 6) + 1
    epsg = (32600 if lat_c >= 0 else 32700) + zone
    return f"EPSG:{epsg}"


def detect_name_column(gdf: gpd.GeoDataFrame, preferred: str | None = None) -> str:
    """Detect the administrative unit name column in a GeoDataFrame."""
    if preferred and preferred in gdf.columns:
        return preferred
    for cand in ["shapeName", "name", "NAME", "NAME_1", "NAME_2", "ADMIN", "admin_name"]:
        if cand in gdf.columns:
            return cand
    for col in gdf.columns:
        if col != "geometry" and gdf[col].dtype == object:
            return col
    raise ValueError("Could not find a region name column in GeoJSON.")


def ocr_polygon_label(orig_img: np.ndarray, geom) -> str:
    """Extract numeric/text label inside a polygon, masked to the polygon interior."""
    h, w = orig_img.shape[:2]
    minx, miny, maxx, maxy = [int(round(v)) for v in geom.bounds]
    pad = 2
    x1, y1 = max(0, minx - pad), max(0, miny - pad)
    x2, y2 = min(w, maxx + pad), min(h, maxy + pad)
    cw, ch = x2 - x1, y2 - y1
    if cw < 12 or ch < 12:
        return ""
    crop = orig_img[y1:y2, x1:x2]

    # Rasterize polygon interior mask in local crop coordinates
    poly_mask = np.zeros((ch, cw), dtype=np.uint8)
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for p in polys:
        if not p.is_empty:
            pts = np.array(p.exterior.coords, dtype=np.float32) - np.array([x1, y1], dtype=np.float32)
            cv2.fillPoly(poly_mask, [pts.astype(np.int32)], 255)
    eroded = cv2.erode(poly_mask, np.ones((3, 3), np.uint8), iterations=1)
    if np.sum(eroded > 0) > 25:
        poly_mask = eroded

    interior_px = crop[poly_mask > 0]
    if len(interior_px) < 10:
        return ""

    val = np.max(crop, axis=2).astype(np.float32)
    fill_val = float(np.median(np.max(interior_px, axis=1)))
    if fill_val < 20:
        return ""

    def _run_tess(bin_img: np.ndarray) -> str:
        # Force everything outside the polygon interior to white background
        masked = np.where(poly_mask > 0, bin_img, 255).astype(np.uint8)
        up = cv2.resize(masked, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        _, png_buf = cv2.imencode(".png", up)
        try:
            res = subprocess.run(
                ["tesseract", "stdin", "stdout", "--psm", "6", "-c", "tessedit_char_whitelist=0123456789."],
                input=png_buf.tobytes(),
                capture_output=True,
            )
            tokens = [t.strip(".") for t in res.stdout.decode("utf-8", errors="replace").split()]
            nums = [t for t in tokens if re.fullmatch(r"\d+(?:\.\d+)?", t)]
            return " ".join(nums) if nums else " ".join(t for t in tokens if t)
        except Exception:
            return ""

    b, g, r = crop[:, :, 0].astype(int), crop[:, :, 1].astype(int), crop[:, :, 2].astype(int)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sat = np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r)
    white_txt = np.where((gray > 205) & (sat < 45), 0, 255).astype(np.uint8)

    norm = np.clip((val / fill_val) * 255.0, 0, 255).astype(np.uint8)
    _, th_dark = cv2.threshold(norm, 175, 255, cv2.THRESH_BINARY)

    # Prioritize white-text OCR when interior has significant white text pixels
    white_frac = float(np.mean((gray[poly_mask > 0] > 205) & (sat[poly_mask > 0] < 45)))
    order = (white_txt, th_dark) if white_frac > 0.01 else (th_dark, white_txt)
    for cand in order:
        txt = _run_tess(cand)
        if txt:
            return txt
    return ""


def match_polygons(
    orig_img: np.ndarray,
    clean_img: np.ndarray,
    extracted: list[dict],
    legend_bins: list[dict],
    gdf: gpd.GeoDataFrame,
    iso3: str,
    name_col: str,
    run_ocr: bool = False,
) -> tuple[list[dict], gpd.GeoDataFrame]:
    """Align database GeoJSON polygons to extracted image polygons and match each region."""
    h, w = orig_img.shape[:2]
    if iso3 == "USA":
        gdf = gdf[~gdf[name_col].isin(US_TERRITORIES)].copy()

    crs = pick_projected_crs(gdf, iso3)
    gdf_wgs84 = gdf.to_crs("EPSG:4326") if gdf.crs and str(gdf.crs) != "EPSG:4326" else gdf
    gdf["__lon"] = gdf_wgs84.geometry.centroid.x.round(6)
    gdf["__lat"] = gdf_wgs84.geometry.centroid.y.round(6)
    gdf_proj = gdf.to_crs(crs)

    # Handle US Alaska/Hawaii insets separately from contiguous 48 states
    if iso3 == "USA" and {"Alaska", "Hawaii"}.issubset(set(gdf_proj[name_col])):
        main_gdf = gdf_proj[~gdf_proj[name_col].isin(["Alaska", "Hawaii"])].copy()
        main_ext = [p for p in extracted if not (p["centroid"][0] < w * 0.32 and p["centroid"][1] > h * 0.70)]
        inset_ext = [p for p in extracted if (p["centroid"][0] < w * 0.32 and p["centroid"][1] > h * 0.70)]
    else:
        main_gdf = gdf_proj.copy()
        main_ext = extracted
        inset_ext = []

    if not main_ext:
        raise RuntimeError("No polygons extracted for main map region.")

    main_union = unary_union([p["geometry"] for p in main_ext])
    ex_minx, ex_miny, ex_maxx, ex_maxy = main_union.bounds
    db_minx, db_miny, db_maxx, db_maxy = main_gdf.total_bounds

    sx = (ex_maxx - ex_minx) / (db_maxx - db_minx)
    sy = -(ex_maxy - ex_miny) / (db_maxy - db_miny)
    tx = ex_minx - sx * db_minx
    ty = ex_maxy - sy * db_miny
    mat = [sx, 0, 0, sy, tx, ty]
    main_gdf["img_geom"] = main_gdf.geometry.apply(lambda g: affine_transform(g, mat).buffer(0))

    def match_legend(bgr):
        if not legend_bins:
            return None
        return min(legend_bins, key=lambda lb: sum((a - b) ** 2 for a, b in zip(bgr, lb["bgr"])))

    def sample_geom_color(img_geom):
        mask = np.zeros((h, w), dtype=np.uint8)
        polys = img_geom.geoms if img_geom.geom_type == "MultiPolygon" else [img_geom]
        for p in polys:
            if not p.is_empty:
                cv2.fillPoly(mask, [np.array(p.exterior.coords, dtype=np.int32)], 255)
        pts = clean_img[mask > 0]
        if len(pts) == 0:
            cx, cy = int(round(img_geom.centroid.x)), int(round(img_geom.centroid.y))
            return clean_img[min(h - 1, max(0, cy)), min(w - 1, max(0, cx))].tolist()
        b_ch, g_ch, r_ch = pts[:, 0].astype(int), pts[:, 1].astype(int), pts[:, 2].astype(int)
        val = np.maximum(np.maximum(b_ch, g_ch), r_ch)
        sat = val - np.minimum(np.minimum(b_ch, g_ch), r_ch)
        valid = pts[(val < 238) & (sat > 15)]
        if len(valid) > 5:
            pts = valid
        return np.median(pts, axis=0).astype(int).tolist()

    def _extract_geoid(row_obj, fallback_name: str) -> str:
        for col in ("shapeID", "GEOID", "geoid", "id", "ID", "shapeISO"):
            val = row_obj.get(col)
            if val is not None and str(val).strip() and str(val).strip().lower() != "nan":
                return str(val).strip()
        return f"{iso3}:{fallback_name}"

    matched = []
    for _, drow in main_gdf.sort_values(name_col).iterrows():
        region_name = str(drow[name_col])
        dg = drow["img_geom"]
        overlapping_ids = []
        best_iou = 0.0
        for ep in main_ext:
            inter = ep["geometry"].intersection(dg).area
            if inter > 0:
                union = ep["geometry"].union(dg).area
                best_iou = max(best_iou, inter / union if union > 0 else 0.0)
                if inter / min(ep["area"], dg.area) > 0.25 or inter / dg.area > 0.15:
                    overlapping_ids.append(ep["id"])

        bgr = sample_geom_color(dg)
        hx = "#{:02x}{:02x}{:02x}".format(bgr[2], bgr[1], bgr[0])
        cname = bgr_to_color_name(bgr)
        lbin = match_legend(bgr)
        entry = {
            "name": region_name,
            "geoid": _extract_geoid(drow, region_name),
            "latitude": float(drow["__lat"]),
            "longitude": float(drow["__lon"]),
            "extracted_polygon_ids": overlapping_ids,
            "iou": round(best_iou, 3),
            "centroid_px": [round(float(dg.centroid.x), 1), round(float(dg.centroid.y), 1)],
            "color_bgr": bgr,
            "color_hex": hx,
            "color_name": cname,
            "legend_label": lbin["label"] if lbin else cname,
        }
        if run_ocr:
            ocr_val = ocr_polygon_label(orig_img, dg)
            entry["ocr_label"] = ocr_val
            entry["value"] = ocr_val or (lbin["label"] if lbin else None)
        else:
            entry["value"] = lbin["label"] if lbin else None
        matched.append(entry)

    # Match Alaska / Hawaii insets if present
    if inset_ext:
        inset_sorted = sorted(inset_ext, key=lambda p: p["area"], reverse=True)
        ak_row = gdf_proj[gdf_proj[name_col] == "Alaska"].iloc[0]
        ak = inset_sorted[0]
        lbin = match_legend(ak["color_bgr"])
        ak_entry = {
            "name": "Alaska",
            "geoid": _extract_geoid(ak_row, "Alaska"),
            "latitude": float(ak_row["__lat"]),
            "longitude": float(ak_row["__lon"]),
            "extracted_polygon_ids": [ak["id"]],
            "iou": 1.0,
            "centroid_px": [round(ak["centroid"][0], 1), round(ak["centroid"][1], 1)],
            "color_bgr": ak["color_bgr"],
            "color_hex": "#{:02x}{:02x}{:02x}".format(ak["color_bgr"][2], ak["color_bgr"][1], ak["color_bgr"][0]),
            "legend_label": lbin["label"] if lbin else None,
        }
        if run_ocr:
            ak_entry["ocr_label"] = ocr_polygon_label(orig_img, ak["geometry"])
        matched.append(ak_entry)

        if len(inset_sorted) > 1:
            hi_row = gdf_proj[gdf_proj[name_col] == "Hawaii"].iloc[0]
            hi_polys = inset_sorted[1:]
            hi_bgr = hi_polys[0]["color_bgr"]
            lbin_hi = match_legend(hi_bgr)
            hi_entry = {
                "name": "Hawaii",
                "geoid": _extract_geoid(hi_row, "Hawaii"),
                "latitude": float(hi_row["__lat"]),
                "longitude": float(hi_row["__lon"]),
                "extracted_polygon_ids": [p["id"] for p in hi_polys],
                "iou": 1.0,
                "centroid_px": [round(hi_polys[0]["centroid"][0], 1), round(hi_polys[0]["centroid"][1], 1)],
                "color_bgr": hi_bgr,
                "color_hex": "#{:02x}{:02x}{:02x}".format(hi_bgr[2], hi_bgr[1], hi_bgr[0]),
                "legend_label": lbin_hi["label"] if lbin_hi else None,
            }
            if run_ocr:
                hi_entry["ocr_label"] = ocr_polygon_label(orig_img, hi_polys[0]["geometry"])
            matched.append(hi_entry)

    w_minx, w_miny, w_maxx, w_maxy = gdf.loc[main_gdf.index].to_crs("EPSG:4326").total_bounds if gdf.crs else main_gdf.total_bounds
    lsx = (ex_maxx - ex_minx) / (w_maxx - w_minx)
    lsy = -(ex_maxy - ex_miny) / (w_maxy - w_miny)
    ltx = ex_minx - lsx * w_minx
    lty = ex_maxy - lsy * w_miny
    matched.sort(key=lambda x: x["name"])
    return matched, main_gdf, {"crs": crs, "affine": [float(v) for v in mat], "lonlat_affine": [float(lsx), 0.0, 0.0, float(lsy), float(ltx), float(lty)]}


def save_debug_overlay(orig_img: np.ndarray, matched: list[dict], out_path: str) -> None:
    """Save a visual debug image showing matched region centroids and names."""
    vis = orig_img.copy()
    for item in matched:
        cx, cy = int(round(item["centroid_px"][0])), int(round(item["centroid_px"][1]))
        bgr = tuple(int(v) for v in item["color_bgr"])
        cv2.circle(vis, (cx, cy), 6, bgr, -1)
        cv2.circle(vis, (cx, cy), 6, (0, 0, 0), 1)
        cv2.putText(vis, item["name"][:12], (cx + 7, cy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, vis)


def clean_legend_labels_with_llm(legend_bins: list[dict], ollama_host: str, model: str = "gemma4:31b-cloud") -> list[dict]:
    """Fix OCR garble in extracted legend labels using Ollama."""
    if not legend_bins or not ollama_host:
        return legend_bins
    raw_labels = [lb.get("label", "") for lb in legend_bins]
    prompt = (
        "Fix any OCR typos or garbled characters in these map legend labels while preserving the exact meaning and order. "
        "Return ONLY a JSON array of strings with the same length.\n"
        f"Labels: {json.dumps(raw_labels, ensure_ascii=False)}"
    )
    try:
        req = urllib.request.Request(
            f"{ollama_host}/api/generate",
            data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            txt = json.loads(resp.read().decode("utf-8")).get("response", "")
            m = re.search(r"\[.*\]", txt, re.DOTALL)
            if m:
                cleaned = json.loads(m.group(0))
                if isinstance(cleaned, list) and len(cleaned) == len(legend_bins):
                    for lb, new_lbl in zip(legend_bins, cleaned):
                        if isinstance(new_lbl, str) and new_lbl.strip():
                            lb["label"] = new_lbl.strip()
    except Exception:
        pass
    return legend_bins


def synthesize_legend_from_regions(matched: list[dict]) -> list[dict]:
    """Synthesize a non-overlapping, ordered color->range legend from valid numeric region values."""
    import pandas as pd

    df = pd.DataFrame(matched)
    if df.empty or "value" not in df.columns:
        return []
    df["num_val"] = pd.to_numeric(df["value"], errors="coerce")
    valid = df.dropna(subset=["num_val"])
    if valid.empty:
        return []

    # 1. Global IQR filter to remove truncated/corrupted OCR fragments (e.g. 3, 4, 13, 15, 92.4)
    q1, q3 = valid["num_val"].quantile(0.25), valid["num_val"].quantile(0.75)
    iqr = q3 - q1
    clean = valid[(valid["num_val"] >= q1 - 1.5 * iqr) & (valid["num_val"] <= q3 + 1.5 * iqr)]
    if clean.empty:
        clean = valid

    # 2. Group by color_name and sort by median value
    groups = []
    for cname, grp in clean.groupby("color_name"):
        rep_row = grp.iloc[0]
        groups.append({
            "color_name": cname,
            "median": float(grp["num_val"].median()),
            "min": float(grp["num_val"].min()),
            "max": float(grp["num_val"].max()),
            "count": int(len(grp)),
            "bgr": rep_row["color_bgr"],
            "hex": rep_row["color_hex"],
        })
    groups.sort(key=lambda x: x["median"])

    # 3. Enforce strictly non-overlapping intervals partitioned at midpoints between adjacent medians
    synth = []
    for i, g in enumerate(groups):
        lo = g["min"] if i == 0 else round((groups[i - 1]["median"] + g["median"]) / 2.0, 1)
        hi = g["max"] if i == len(groups) - 1 else round((g["median"] + groups[i + 1]["median"]) / 2.0, 1)
        lbl = f"{lo:g} - {hi:g}" if lo < hi else f"{lo:g}"
        synth.append({
            "color_name": g["color_name"],
            "label": lbl,
            "min": lo,
            "max": hi,
            "count": g["count"],
            "bgr": g["bgr"],
            "hex": g["hex"],
        })
    return synth


def detect_value_outliers(df_or_regions, value_col: str = "Value"):
    """Use pandas IQR (1.5 * IQR) to flag numeric outliers and non-numeric/missing values."""
    import pandas as pd

    df = df_or_regions.copy() if isinstance(df_or_regions, pd.DataFrame) else pd.DataFrame(df_or_regions)
    col = value_col if value_col in df.columns else ("value" if "value" in df.columns else None)
    if col is None or df.empty:
        return df, []

    nums = pd.to_numeric(df[col], errors="coerce")
    valid = nums.dropna()
    if len(valid) < 4:
        df["is_outlier"] = False
        df["outlier_reason"] = ""
        return df, []

    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    reasons = []
    is_out = []
    for raw_v, num_v in zip(df[col], nums):
        raw_s = "" if pd.isna(raw_v) else str(raw_v).strip()
        if not raw_s:
            is_out.append(True)
            reasons.append("missing")
        elif pd.isna(num_v):
            is_out.append(True)
            reasons.append(f"non_numeric:{raw_s}")
        elif num_v < low or num_v > high:
            is_out.append(True)
            reasons.append(f"out_of_iqr_bounds[{low:.2f},{high:.2f}]")
        else:
            is_out.append(False)
            reasons.append("")

    df["is_outlier"] = is_out
    df["outlier_reason"] = reasons
    outliers = df[df["is_outlier"]].to_dict(orient="records")
    return df, outliers


def fix_outliers_hybrid(orig_img: np.ndarray, matched: list[dict], geoms_by_name: dict, ollama_host: str = "", model: str = "gemma4:31b-cloud") -> list[dict]:
    """Hybrid outlier repair: (1) decimal/token heuristic, (2) Ollama vision fallback on remaining outliers."""
    import base64
    import pandas as pd

    df, _ = detect_value_outliers(matched, value_col="value")
    nums = pd.to_numeric(df["value"], errors="coerce")
    valid = nums.dropna()
    q1, q3 = (valid.quantile(0.25), valid.quantile(0.75)) if len(valid) >= 4 else (0.0, 100.0)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    # Stage 1: Decimal & multi-token heuristic
    for item in matched:
        raw = str(item.get("value") or "").strip()
        if not raw:
            continue
        tokens = raw.split()
        dec_tokens = [t for t in tokens if re.fullmatch(r"\d+\.\d+", t)]
        if dec_tokens:
            v = float(dec_tokens[0])
            if low <= v <= high:
                item["value"] = dec_tokens[0]
                continue
        if len(tokens) == 1 and tokens[0].isdigit():
            d = tokens[0]
            if len(d) == 3 and not (low <= float(d) <= high):
                cand = f"{d[:2]}.{d[2]}"
                if low <= float(cand) <= high:
                    item["value"] = cand
            elif len(d) == 4 and not (low <= float(d) <= high):
                cand = f"{d[:2]}.{d[2]}"
                if low <= float(cand) <= high:
                    item["value"] = cand

    # Stage 2: Ollama vision fallback on remaining outliers
    _, remaining = detect_value_outliers(matched, value_col="value")
    if remaining and ollama_host:
        h, w = orig_img.shape[:2]
        rem_names = {r.get("name") for r in remaining}
        for item in matched:
            if item["name"] not in rem_names:
                continue
            geom = geoms_by_name.get(item["name"])
            if geom is None:
                continue
            minx, miny, maxx, maxy = [int(round(v)) for v in geom.bounds]
            x1, y1 = max(0, minx - 6), max(0, miny - 6)
            x2, y2 = min(w, maxx + 6), min(h, maxy + 6)
            crop = orig_img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            _, buf = cv2.imencode(".jpg", crop)
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            try:
                req = urllib.request.Request(
                    f"{ollama_host}/api/generate",
                    data=json.dumps({
                        "model": model,
                        "prompt": f"Read ONLY the single number printed inside {item['name']} in this map crop. Reply with just the number (e.g. 42.1 or 46) and nothing else.",
                        "images": [b64],
                        "stream": False,
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    ans = json.loads(resp.read().decode("utf-8")).get("response", "").strip()
                    m = re.search(r"\d+(?:\.\d+)?", ans)
                    if m:
                        item["value"] = m.group(0)
            except Exception:
                pass
    return matched


def verify_polygon_alignment(matched: list[dict]) -> dict:
    """Verify whether database polygons aligned properly onto the extracted map regions."""
    total = len(matched)
    if total == 0:
        return {"status": "FAIL", "matched_pct": 0.0, "mean_iou": 0.0, "bg_centroid_pct": 100.0}
    with_poly = sum(1 for r in matched if r.get("extracted_polygon_ids"))
    mean_iou = sum(float(r.get("iou") or 0.0) for r in matched) / total
    bg_centroids = sum(1 for r in matched if all(c > 230 for c in (r.get("color_bgr") or [0, 0, 0])))
    matched_pct = round(100.0 * with_poly / total, 1)
    bg_pct = round(100.0 * bg_centroids / total, 1)
    status = "OK" if matched_pct >= 75.0 and bg_pct <= 15.0 else "WARNING"
    return {
        "status": status,
        "matched_pct": matched_pct,
        "mean_iou": round(mean_iou, 3),
        "bg_centroid_pct": bg_pct,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract polygons from a map image and match them with a country GeoJSON registry."
    )
    parser.add_argument("input", nargs="?", help="Image file path or URL (e.g. http://localhost:7070/vault/cN4TM5FZE2m1wFdMMH20wA.webp)")
    parser.add_argument("--country", "-c", help="Country name, ISO2, or ISO3 code (e.g. USA, Honduras, NLD)")
    parser.add_argument("--adm", default="ADM1", help="geoBoundaries admin level (ADM0, ADM1, ADM2; default: ADM1)")
    parser.add_argument("--geojson", "-g", help="Optional local GeoJSON/Shapefile path (bypasses registry)")
    parser.add_argument("--name-field", help="GeoJSON property name for region labels (auto-detected if omitted)")
    parser.add_argument("--cache-dir", default="~/.cache/geoboundaries", help="Cache directory for downloaded GeoJSONs")
    parser.add_argument("--title", default="", help="Optional map/dataset title")
    parser.add_argument("--description", default="", help="Optional map/dataset description")
    parser.add_argument("--variable", default="", help="Predicted variable name measured on the map (e.g. 'Maximum Temperature')")
    parser.add_argument("--unit", default="", help="Unit of measure for the predicted variable (e.g. '°C', 'mm', '%%')")
    parser.add_argument("--year", default="", help="Time variable: year (e.g. '2019', '2022')")
    parser.add_argument("--month", default="", help="Time variable: month (e.g. '06', 'June')")
    parser.add_argument("--day", default="", help="Time variable: day (e.g. '28')")
    parser.add_argument("--auto-describe", action="store_true", help="Query Ollama vision model (from palefire/.env) for title, description, variable, unit, and legend")
    parser.add_argument("--cdif-output", help="Optional path to write DDI-CDI / CDIF JSON-LD output")
    parser.add_argument("--check-csv", help="Optional CSV path/URL to audit for Value outliers using pandas")
    parser.add_argument("--output", "-o", default="output/matched_polygons.json", help="Output JSON path")
    parser.add_argument("--debug-image", help="Optional path to save debug overlay image")
    parser.add_argument("--ocr-labels", action="store_true", help="Extract in-polygon numeric text labels via Tesseract")
    args = parser.parse_args()

    if args.check_csv:
        import pandas as pd
        df = pd.read_csv(args.check_csv)
        _, outliers = detect_value_outliers(df, value_col="Value")
        print(f"Audited {len(df)} rows from {args.check_csv}: found {len(outliers)} outliers/invalid values.")
        for o in outliers:
            print(f"  - {o.get('Department') or o.get('name')}: Value={o.get('Value')!r} ({o.get('outlier_reason')})")
        return

    if not args.geojson and not args.country:
        parser.error("Provide either --country (to query geoBoundaries registry) or --geojson (local file).")

    iso3 = resolve_iso3(args.country) if args.country else "CUSTOM"
    if args.geojson:
        geojson_path = Path(args.geojson)
    else:
        print(f"Resolving {iso3} {args.adm} from geoBoundaries registry...")
        geojson_path = fetch_geoboundaries_geojson(iso3, args.adm, cache_dir=args.cache_dir)
        print(f"Using GeoJSON: {geojson_path}")

    img = load_image(args.input)

    title = args.title
    description = args.description
    variable_name = args.variable
    unit_name = args.unit

    if args.auto_describe:
        import base64
        env_vars = {}
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            for ln in env_path.read_text().splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("#") and "=" in ln:
                    k, v = ln.split("=", 1)
                    env_vars[k.strip()] = v.strip()
        ollama_host = (os.environ.get("OLLAMA_HOST") or env_vars.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        model = os.environ.get("OLLAMA_VISUAL_MODEL") or env_vars.get("OLLAMA_VISUAL_MODEL") or "gemma4:31b"
        _, buf = cv2.imencode(".jpg", img)
        b64_img = base64.b64encode(buf.tobytes()).decode("ascii")
        prompt = (
            "Analyze this map image and return ONLY a JSON object with keys: "
            '"title", "description", "variable", "unit", "legend". '
            '"variable" is the predicted physical/thematic variable measured, "unit" is its unit of measure, '
            'and "legend" is a list of {"color": "<standard color name>", "label": "<legend text>", "value_range": "<range or category>"}.'
        )
        ollama_legend = []
        try:
            req = urllib.request.Request(
                f"{ollama_host}/api/generate",
                data=json.dumps({"model": model, "prompt": prompt, "images": [b64_img], "stream": False}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                raw_txt = json.loads(resp.read().decode("utf-8")).get("response", "")
                m = re.search(r"\{.*\}", raw_txt, re.DOTALL)
                if m:
                    meta = json.loads(m.group(0))
                    title = title or meta.get("title", "")
                    description = description or meta.get("description", "")
                    variable_name = variable_name or meta.get("variable", "")
                    unit_name = unit_name or meta.get("unit", "")
                    ollama_legend = meta.get("legend") or []
        except Exception as e:
            print(f"Warning: --auto-describe failed ({e}); continuing with CLI args.")

    extracted, clean_img, legend_bins = extract_polygons_from_image(img)
    if legend_bins and args.auto_describe:
        legend_bins = clean_legend_labels_with_llm(legend_bins, ollama_host=ollama_host, model=model)
    if not legend_bins and args.auto_describe and ollama_legend:
        for item in ollama_legend:
            if not isinstance(item, dict):
                continue
            cname = item.get("color", "")
            ref_bgr = STANDARD_COLORS_BGR.get(cname)
            if not ref_bgr:
                for k, v in STANDARD_COLORS_BGR.items():
                    if k.lower() in cname.lower() or cname.lower() in k.lower():
                        ref_bgr = v
                        break
            if ref_bgr:
                lbl = item.get("value_range") or item.get("label") or cname
                legend_bins.append({
                    "label": str(lbl),
                    "bgr": list(ref_bgr),
                    "hex": "#{:02x}{:02x}{:02x}".format(ref_bgr[2], ref_bgr[1], ref_bgr[0]),
                })
    print(f"Extracted {len(extracted)} polygons and {len(legend_bins)} legend bins from image.")

    gdf = gpd.read_file(geojson_path)
    name_col = detect_name_column(gdf, args.name_field)
    matched, main_gdf, spatial_transform = match_polygons(
        img, clean_img, extracted, legend_bins, gdf, iso3, name_col, run_ocr=args.ocr_labels
    )
    if args.ocr_labels:
        env_vars = {}
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            for ln in env_path.read_text().splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("#") and "=" in ln:
                    k, v = ln.split("=", 1)
                    env_vars[k.strip()] = v.strip()
        ohost = env_vars.get("OLLAMA_HOST", "http://localhost:11434").replace("host.docker.internal", "localhost").rstrip("/")
        omodel = env_vars.get("OLLAMA_VISUAL_MODEL", "gemma4:31b-cloud")
        geoms_by_name = {str(r[name_col]): r["img_geom"] for _, r in main_gdf.iterrows()}
        matched = fix_outliers_hybrid(img, matched, geoms_by_name, ollama_host=ohost, model=omodel)

    alignment = verify_polygon_alignment(matched)
    print(f"Alignment check [{alignment['status']}]: matched={alignment['matched_pct']}%, mean_iou={alignment['mean_iou']}, bg_centroids={alignment['bg_centroid_pct']}%")

    out_payload = {
        "source": args.input,
        "alignment_check": alignment,
        "title": title,
        "description": description,
        "variable": variable_name,
        "unit": unit_name,
        "country_iso3": iso3,
        "adm_level": args.adm,
        "geojson": str(geojson_path),
        "extracted_polygon_count": len(extracted),
        "matched_region_count": len(matched),
        "legend": legend_bins,
        "regions": matched,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(matched)} matched regions to {args.output}")

    if args.cdif_output:
        if not legend_bins:
            legend_bins = synthesize_legend_from_regions(matched)
            out_payload["legend"] = legend_bins
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out_payload, f, indent=2, ensure_ascii=False)
        year_val = args.year
        if not year_val:
            m_yr = re.search(r"\b(?:19|20)\d{2}\b", f"{title} {description}")
            year_val = m_yr.group(0) if m_yr else ""
        month_val = args.month
        day_val = args.day

        cdif_obs = []
        for item in matched:
            obs = dict(item)
            if year_val:
                obs["year"] = year_val
            if month_val:
                obs["month"] = month_val
            if day_val:
                obs["day"] = day_val
            cdif_obs.append(obs)

        cdif_payload = {
            "@context": [
                "https://ddialliance.org/Specification/DDI-CDI/1.0/context.jsonld",
                {"schema": "https://schema.org/", "cr": "http://mlcommons.org/croissant/", "cdif": "https://w3id.org/cdif/core/1.1/"},
            ],
            "@type": ["schema:Dataset", "cr:Dataset"],
            "conformsTo": ["https://w3id.org/cdif/core/1.1", "https://ddialliance.org/Specification/DDI-CDI/1.0/"],
            "schema:name": title or f"{iso3} {args.adm} Map Extraction",
            "schema:description": description,
            "schema:url": args.input,
            "spatialTransform": spatial_transform,
            "legend": legend_bins,
            "DDICDIModels": [
                {"@type": "InstanceVariable", "@id": "#var-measured", "name": {"name": variable_name or "value"}, "unitText": unit_name}
            ] + [
                {"@type": "InstanceVariable", "@id": f"#var-{k}", "name": {"name": k}}
                for k in ("name", "geoid", "latitude", "longitude", "color_hex", "color_name", "legend_label", "value", "year", "month", "day")
            ],
            "observations": cdif_obs,
        }
        Path(args.cdif_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.cdif_output).write_text(json.dumps(cdif_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved CDIF JSON-LD to {args.cdif_output}")

    if args.debug_image:
        save_debug_overlay(img, matched, args.debug_image)
        print(f"Saved debug overlay to {args.debug_image}")


if __name__ == "__main__":
    main()
