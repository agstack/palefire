import cv2
import numpy as np
import geopandas as gpd
import json
from pathlib import Path
import os


def get_affine_transform(geo_bounds, pixel_bounds):
    min_lon, min_lat, max_lon, max_lat = geo_bounds
    x, y, w, h = pixel_bounds
    src_pts = np.float32([[min_lon, max_lat], [max_lon, min_lat], [min_lon, min_lat]])
    dst_pts = np.float32([[x, y], [x + w, y + h], [x, y + h]])
    return cv2.getAffineTransform(src_pts, dst_pts)


def geo_to_pixel_affine(lon, lat, matrix):
    pt = np.array([lon, lat, 1.0])
    res = np.dot(matrix, pt)
    return int(round(res[0])), int(round(res[1]))


def _group_centers(xs):
    if len(xs) == 0:
        return []
    groups = np.split(xs, np.where(np.diff(xs) > 15)[0] + 1)
    return [round(float(np.mean(g)), 1) for g in groups if len(g) <= 12]


def detect_panel_transform(im, side, full_img=None):
    """Automatically detect map frame and graticule ticks to compute an exact affine transform.
    Returns (map_kind, affine_matrix, sampling_img). If the panel has no geographic map,
    affine_matrix is None.
    """
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if 350 < cw < 1900 and 350 < ch < 1900 and y > 150:
            if cw * ch > best_area:
                best_area = cw * ch
                best = (x, y, cw, ch)

    # 1. Standard CENAOS Honduras map (frame width ~1750-1850, height ~1200-1280, graticule 89W..83W, 16N..13N)
    if best and 1700 < best[2] < 1880 and 1180 < best[3] < 1300:
        bx, by, bw, bh = best
        fy = by + int(np.argmin(np.mean(gray[by:by + 25, bx + 50:bx + bw - 50], axis=1)))
        fy2 = by + bh - 25 + int(np.argmin(np.mean(gray[by + bh - 25:by + bh + 5, bx + 50:bx + bw - 50], axis=1)))
        fx = bx + int(np.argmin(np.mean(gray[by + 50:by + bh - 50, bx:bx + 25], axis=0)))
        fx2 = bx + bw - 25 + int(np.argmin(np.mean(gray[by + 50:by + bh - 50, bx + bw - 25:bx + bw + 5], axis=0)))
        row_top = im[max(0, fy - 6), fx:fx2]
        top_dark = np.where((row_top[:, 0] < 80) & (row_top[:, 1] < 80) & (row_top[:, 2] < 80))[0] + fx
        tt = _group_centers(top_dark)
        col_left = im[fy:fy2, max(0, fx - 6)]
        left_dark = np.where((col_left[:, 0] < 80) & (col_left[:, 1] < 80) & (col_left[:, 2] < 80))[0] + fy
        lt = _group_centers(left_dark)
        if len(tt) >= 6 and len(lt) >= 4:
            x89, x_last = tt[0], tt[min(6, len(tt) - 1)]
            lon_right = -89.0 + min(6, len(tt) - 1)
            y16, y13 = lt[0], lt[3]
            src = np.float32([[-89.0, 16.0], [lon_right, 16.0], [-89.0, 13.0]])
            dst = np.float32([[x89, y16], [x_last, y16], [x89, y13]])
            return "cenaos_graticule", cv2.getAffineTransform(src, dst), im

    # 2. FAO GIEWS NDVI Honduras map (Page 11: frame width ~1600-1680, height ~780-830)
    if best and 1580 < best[2] < 1700 and 750 < best[3] < 860:
        crop = im[280:1260, 30:1750]
        sat = np.max(crop, axis=2).astype(int) - np.min(crop, axis=2).astype(int)
        ys, xs = np.where(sat > 25)
        if len(xs) > 100:
            xmin, xmax = float(np.min(xs) + 30), float(np.max(xs) + 30)
            ymin, ymax = float(np.min(ys) + 280), float(np.max(ys) + 280)
            src = np.float32([[-89.3568, 16.5158], [-83.1349, 16.5158], [-89.3568, 12.9808]])
            dst = np.float32([[xmin, ymin], [xmax, ymin], [xmin, ymax]])
            return "fao_giews", cv2.getAffineTransform(src, dst), im

    # 3. NMME Central America maps (Pages 4 & 5: frame width 800..1200, height 750..1120, 94W..75W, 22N..7N)
    if best and 780 < best[2] < 1220 and 750 < best[3] < 1120:
        bx, by, bw, bh = best
        src = np.float32([[-94.0, 22.0], [-75.0, 22.0], [-94.0, 7.0]])
        dst = np.float32([[bx + 6.0, by + 6.0], [bx + bw - 6.0, by + 6.0], [bx + 6.0, by + bh - 6.0]])
        return "nmme_subframe", cv2.getAffineTransform(src, dst), im

    # 4. ECMWF Seasonal Forecast maps (Page 6: frame width 1680..1750, height 900..960, 150W..60W, 30N..30S)
    if best and 1680 < best[2] < 1750 and 900 < best[3] < 960:
        bx, by, bw, bh = best
        src = np.float32([[-150.0, 30.0], [-60.0, 30.0], [-150.0, -30.0]])
        dst = np.float32([[bx + 150.0, by + 60.0], [bx + 1330.0, by + 60.0], [bx + 150.0, by + 860.0]])
        return "ecmwf_subframe", cv2.getAffineTransform(src, dst), im

    # 5. Full-page maps spanning across the left/right midpoint (Page 2 SST or Page 27 Drought Alert)
    if full_img is not None:
        fgray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
        _, fthresh = cv2.threshold(fgray, 80, 255, cv2.THRESH_BINARY_INV)
        fcontours, _ = cv2.findContours(fthresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        fbest = None
        fbest_area = 0
        for c in fcontours:
            x, y, cw, ch = cv2.boundingRect(c)
            if 2400 < cw < 3500 and 1200 < ch < 1850 and y > 200:
                if cw * ch > fbest_area:
                    fbest_area = cw * ch
                    fbest = (x, y, cw, ch)
        if fbest:
            bx, by, bw, bh = fbest
            if bw > 2900:
                # Page 2 SST Anomalies: Honduras (89W..83W) is in the right half
                if side == "right":
                    src = np.float32([[-120.0, 30.0], [-90.0, 30.0], [-120.0, 0.0]])
                    dst = np.float32([[2510.5 - 1920, 562.0], [3052.0 - 1920, 562.0], [2510.5 - 1920, 1216.0]])
                    return "sst_right", cv2.getAffineTransform(src, dst), im
                else:
                    return "no_map", None, im
            elif 2400 < bw <= 2900:
                # Page 27 Drought Alert Honduras map spanning x=101..2649
                if side == "left":
                    src = np.float32([[-89.0, 16.0], [-84.0, 16.0], [-89.0, 13.0]])
                    dst = np.float32([[300.5, 684.5], [2268.0, 684.5], [300.5, 1911.0]])
                    return "drought_full", cv2.getAffineTransform(src, dst), full_img
                else:
                    return "no_map", None, im

    return "no_map", None, im


def get_department_point(geom, loc_name):
    """Return a guaranteed interior point (lon, lat) for a department geometry.
    For Bay Islands, returns the interior point of Roatán (largest island) rather than open ocean.
    """
    if geom.geom_type == "MultiPolygon":
        # Filter out Swan Islands (>17N) when choosing the main island
        candidates = [g for g in geom.geoms if g.bounds[3] < 17.0] or list(geom.geoms)
        largest = max(candidates, key=lambda g: g.area)
        c = largest.centroid
        pt = c if largest.contains(c) else largest.representative_point()
        return float(pt.x), float(pt.y)
    c = geom.centroid
    pt = c if geom.contains(c) else geom.representative_point()
    return float(pt.x), float(pt.y)


def sample_department_color(sample_img, geom, loc_name, matrix, px, py):
    """Sample department color using full polygon rasterization + patch fallback,
    filtering out black outlines, text labels, and white background halos.
    """
    sh, sw = sample_img.shape[:2]
    mask = np.zeros((sh, sw), dtype=np.uint8)
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        if poly.bounds[3] > 17.0:
            continue
        coords = np.array(poly.exterior.coords)
        ones = np.ones((coords.shape[0], 1))
        pts = np.hstack([coords[:, :2], ones])
        px_pts = np.dot(pts, matrix.T).astype(np.int32)
        cv2.fillPoly(mask, [px_pts], 255)

    if loc_name != "Bay Islands":
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        if np.sum(eroded > 0) > 40:
            mask = eroded

    pixels = sample_img[mask > 0]
    if len(pixels) < 10:
        r = 12
        y1, y2 = max(0, py - r), min(sh, py + r + 1)
        x1, x2 = max(0, px - r), min(sw, px + r + 1)
        patch = sample_img[y1:y2, x1:x2]
        pixels = patch.reshape(-1, 3) if patch.size > 0 else np.array([[255, 255, 255]])

    b = pixels[:, 0].astype(int)
    g = pixels[:, 1].astype(int)
    r_ch = pixels[:, 2].astype(int)
    sat = np.maximum(np.maximum(b, g), r_ch) - np.minimum(np.minimum(b, g), r_ch)
    val = np.maximum(np.maximum(b, g), r_ch)

    # Prefer chromatic (colored) map pixels (excludes white ocean/background and black/grey text/outlines)
    sat_valid = pixels[(val <= 240) & (val >= 45) & (sat >= 16)]
    if len(sat_valid) >= 12:
        med = np.median(sat_valid, axis=0)
        return int(med[0]), int(med[1]), int(med[2])

    # Next, check non-white, non-black pixels (e.g. valid grey/tan map fill)
    mid_valid = pixels[(val <= 225) & (val >= 55)]
    if len(mid_valid) >= max(15, int(len(pixels) * 0.25)):
        med = np.median(mid_valid, axis=0)
        return int(med[0]), int(med[1]), int(med[2])

    med = np.median(pixels, axis=0)
    return int(med[0]), int(med[1]), int(med[2])


def extract_map(img, h, w, pixel_bounds, gdf, geo_bounds, source_name, matrix=None, sample_img=None):
    if sample_img is None:
        sample_img = img
    results = []

    for idx, row in gdf.iterrows():
        loc_name = row.get("shapeName", f"Department_{idx}")
        geom = row.geometry
        if geom is None:
            continue

        lon, lat = get_department_point(geom, loc_name)
        if matrix is None:
            # Non-map panel (title slide, chart, or text panel)
            results.append({
                "source": source_name,
                "location": loc_name,
                "lon": lon,
                "lat": lat,
                "pixel_x": -1,
                "pixel_y": -1,
                "color_bgr": [-1, -1, -1]
            })
            continue

        px, py = geo_to_pixel_affine(lon, lat, matrix)
        b, g, r_color = sample_department_color(sample_img, geom, loc_name, matrix, px, py)

        results.append({
            "source": source_name,
            "location": loc_name,
            "lon": lon,
            "lat": lat,
            "pixel_x": px,
            "pixel_y": py,
            "color_bgr": [b, g, r_color]
        })

        if 0 <= px < w and 0 <= py < h:
            cv2.circle(img, (px, py), 12, (b, g, r_color), -1)
            cv2.circle(img, (px, py), 12, (0, 0, 0), 2)

    return results


def process_image(image_path, gdf, geo_bounds, output_dir="report"):
    print(f"Processing {image_path}...")
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    h, w = img.shape[:2]
    all_results = []

    mid = w // 2
    left_img = img[:, :mid].copy()
    right_img = img[:, mid:].copy()

    pixel_bounds = (654, 604, 518, 446)

    if os.environ.get("SINGLE_MAP") == "1":
        pixel_bounds = (0, 0, w, h)
        _, mat, samp = detect_panel_transform(img, "left", img)
        if mat is None:
            mat = get_affine_transform(geo_bounds, pixel_bounds)
            samp = img
        res = extract_map(img, h, w, pixel_bounds, gdf, geo_bounds, Path(image_path).name, matrix=mat, sample_img=samp)
        all_results.extend(res)
        cv2.imwrite(f"{output_dir}/debug_{Path(image_path).name}", img)
    else:
        kind_l, mat_l, samp_l = detect_panel_transform(left_img, "left", img)
        print(f"  {Path(image_path).name}_left -> {kind_l}")
        res_left = extract_map(left_img, h, mid, pixel_bounds, gdf, geo_bounds, Path(image_path).name + "_left", matrix=mat_l, sample_img=samp_l)
        all_results.extend(res_left)
        cv2.imwrite(f"{output_dir}/debug_{Path(image_path).stem}_left.png", left_img)

        kind_r, mat_r, samp_r = detect_panel_transform(right_img, "right", img)
        print(f"  {Path(image_path).name}_right -> {kind_r}")
        res_right = extract_map(right_img, h, mid, pixel_bounds, gdf, geo_bounds, Path(image_path).name + "_right", matrix=mat_r, sample_img=samp_r)
        all_results.extend(res_right)
        cv2.imwrite(f"{output_dir}/debug_{Path(image_path).stem}_right.png", right_img)

    return all_results
