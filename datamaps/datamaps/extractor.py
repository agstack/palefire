import cv2
import numpy as np
import geopandas as gpd
import json
from pathlib import Path
import os

from shapely.geometry import Polygon, MultiPolygon

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


def extract_map(img, h, w, pixel_bounds, gdf, geo_bounds, source_name):
    matrix = get_affine_transform(geo_bounds, pixel_bounds)
    results = []

    for idx, row in gdf.iterrows():
        loc_name = row.get("shapeName", f"Department_{idx}")
        geom = row.geometry
        if geom is None: continue

        centroid = geom.centroid
        lon, lat = centroid.x, centroid.y
        px, py = geo_to_pixel_affine(lon, lat, matrix)

        # Multi-point polygon sampling:
        # Instead of only reading a small patch around the centroid (which can fall on text, lines, or near a boundary),
        # rasterize the entire department polygon to sample thousands of interior points.
        mask = np.zeros((h, w), dtype=np.uint8)
        polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
        for poly in polys:
            ext = np.array([geo_to_pixel_affine(c_lon, c_lat, matrix) for c_lon, c_lat in poly.exterior.coords])
            cv2.fillPoly(mask, [ext], 255)
            for interior in poly.interiors:
                int_pts = np.array([geo_to_pixel_affine(c_lon, c_lat, matrix) for c_lon, c_lat in interior.coords])
                cv2.fillPoly(mask, [int_pts], 0)

        # Apply morphological erosion to discard pixels within 3px of borders
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        dept_pixels = img[eroded_mask == 255]

        # For very small regions (or islands) where erosion may eliminate all pixels, fall back to un-eroded mask
        if len(dept_pixels) == 0:
            dept_pixels = img[mask == 255]

        # If polygon mask is outside image bounds or empty, fallback to local patch
        if len(dept_pixels) == 0:
            r = 15
            y1, y2 = max(0, py - r), min(h, py + r + 1)
            x1, x2 = max(0, px - r), min(w, px + r + 1)
            patch = img[y1:y2, x1:x2]
            if patch.size > 0:
                dept_pixels = patch.reshape(-1, 3)

        if len(dept_pixels) > 0:
            # Filter out pure white (background, text glow) and pure black (text, department borders)
            valid_pixels = []
            for p in dept_pixels:
                if (p[0] > 220 and p[1] > 220 and p[2] > 220) or (p[0] < 35 and p[1] < 35 and p[2] < 35):
                    continue
                valid_pixels.append(p)

            # Robust color calculation:
            # If we have valid interior pixels, take median of valid pixels.
            # Otherwise, fallback to the entire pixel sample.
            if len(valid_pixels) >= 10:
                median_color = np.median(valid_pixels, axis=0)
            elif valid_pixels:
                median_color = np.median(valid_pixels, axis=0)
            else:
                median_color = np.median(dept_pixels, axis=0)

            b, g, r_color = int(round(median_color[0])), int(round(median_color[1])), int(round(median_color[2]))

            results.append({
                "source": source_name,
                "location": loc_name,
                "lon": lon,
                "lat": lat,
                "pixel_x": px,
                "pixel_y": py,
                "color_bgr": [b, g, r_color],
                "sampled_points": len(valid_pixels) if valid_pixels else len(dept_pixels)
            })

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

    # Allow any image size; previously only 3840x2160 was processed
    mid = w // 2
    left_img = img[:, :mid].copy()
    right_img = img[:, mid:].copy()

    pixel_bounds = (654, 604, 518, 446)

    if os.environ.get("SINGLE_MAP") == "1":
        pixel_bounds = (0, 0, w, h)
        res = extract_map(img, h, w, pixel_bounds, gdf, geo_bounds, Path(image_path).name)
        all_results.extend(res)
        cv2.imwrite(f"{output_dir}/debug_{Path(image_path).name}", img)
    else:
        res_left = extract_map(left_img, h, mid, pixel_bounds, gdf, geo_bounds, Path(image_path).name + "_left")
        all_results.extend(res_left)
        cv2.imwrite(f"{output_dir}/debug_{Path(image_path).stem}_left.png", left_img)

        res_right = extract_map(right_img, h, mid, pixel_bounds, gdf, geo_bounds, Path(image_path).name + "_right")
        all_results.extend(res_right)
        cv2.imwrite(f"{output_dir}/debug_{Path(image_path).stem}_right.png", right_img)

    return all_results
