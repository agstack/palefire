import json
import urllib.parse
import os
import re
import csv
from typing import Dict, Any, List


COLOR_PALETTE: Dict[str, str] = {
    "Grey": "#87776e",
    "Tan": "#b6988b",
    "Light Tan": "#d2c39b",
    "Pale Yellow": "#f3f5c3",
    "Light Yellow-Green": "#dcfdac",
    "Yellow-Green": "#b1f57e",
    "Light Green": "#91e362",
    "Green": "#80bf71",
    "Medium Green": "#74c63c",
    "Dark Green": "#68a271",
    "Very Dark Green": "#3f871e",
    "Teal": "#6fa99d",
    "Light Cyan": "#95c9d0",
    "Very Light Blue": "#cdf0ff",
    "Pale Blue": "#afddfe",
    "Light Blue": "#8bd4ff",
    "Sky Blue": "#68a8f5",
    "Blue": "#5e8ae8",
    "Medium Blue": "#5477cf",
    "Dark Blue": "#526cad",
    "Navy Blue": "#4b5f98",
    "Indigo": "#4e4990",
    "Purple": "#68479a",
    "Dark Magenta": "#8a4ba3",
    "Magenta": "#b74cc9",
    "Pink-Magenta": "#c954b7",
    "Dark Brown": "#64471a",
    "Dark Red": "#a3125f",
    "Red": "#ea4043",
    "Dark Orange": "#e6be2e",
    "Orange": "#eba439",
    "Light Orange": "#dfb13c",
    "Yellow": "#e4d941",
    "Light Yellow": "#fff0b6",
    "Map Green": "#c6e887",
    "Map White": "#fdfeff",
    "White": "#ffffff",
}


def bgr_to_hex(bgr: List[int]) -> str:
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    if b < 0 or g < 0 or r < 0:
        return "N/A"
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_bgr(hex_str: str):
    h = hex_str.lstrip("#")
    if len(h) != 6:
        return (204, 204, 204)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def get_color_name(bgr):
    if bgr[0] < 0 or bgr[1] < 0 or bgr[2] < 0:
        return "No Map"
    def dist(c1, c2):
        return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

    color_map = {
        (204, 204, 204): "Grey",
        (110, 119, 135): "Grey",
        (139, 152, 182): "Tan",
        (155, 195, 210): "Light Tan",
        (195, 245, 243): "Pale Yellow",
        (172, 253, 220): "Light Yellow-Green",
        (126, 245, 177): "Yellow-Green",
        (201, 239, 219): "Light Green",
        (98, 227, 145): "Light Green",
        (158, 216, 179): "Green",
        (113, 191, 128): "Green",
        (120, 194, 145): "Dark Green",
        (113, 162, 104): "Dark Green",
        (66, 151, 93): "Dark Green",
        (157, 169, 111): "Teal",
        (208, 201, 149): "Light Cyan",
        (255, 240, 205): "Very Light Blue",
        (254, 221, 175): "Pale Blue",
        (255, 212, 139): "Light Blue",
        (245, 168, 104): "Sky Blue",
        (232, 138, 94): "Blue",
        (207, 119, 84): "Medium Blue",
        (173, 108, 82): "Dark Blue",
        (152, 95, 75): "Navy Blue",
        (144, 73, 78): "Indigo",
        (154, 71, 104): "Purple",
        (163, 75, 138): "Dark Magenta",
        (201, 76, 183): "Magenta",
        (183, 84, 201): "Pink-Magenta",
        (26, 71, 100): "Dark Brown",
        (182, 240, 255): "Light Yellow",
        (50, 220, 230): "Yellow",
        (55, 217, 227): "Yellow",
        (136, 253, 253): "Yellow",
        (139, 219, 255): "Light Orange",
        (181, 232, 253): "Light Orange",
        (47, 207, 216): "Orange",
        (57, 164, 235): "Orange",
        (144, 212, 253): "Orange",
        (46, 190, 230): "Dark Orange",
        (116, 197, 253): "Dark Orange",
        (96, 185, 253): "Red",
        (67, 64, 234): "Red",
        (95, 18, 163): "Dark Red",
        (135, 232, 198): "Map Green",
        (131, 228, 194): "Map Green",
        (234, 234, 234): "Map White",
        (253, 254, 255): "Map White",
    }

    best_dist = float("inf")
    best_name = "Unknown"

    for c_bgr, name in color_map.items():
        d = dist(bgr, c_bgr)
        if d < best_dist and d < 85:
            best_dist = d
            best_name = name

    return best_name


def match_legend_by_bgr(bgr: List[int], legend: List[Dict[str, Any]], max_dist: float = 95.0):
    """Find the closest legend item by Euclidean BGR distance using its hex color."""
    if not legend or bgr[0] < 0 or bgr[1] < 0 or bgr[2] < 0:
        return None, float("inf")

    def dist(c1, c2):
        return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

    best_item = None
    best_dist = float("inf")
    for item in legend:
        hx = item.get("hex")
        if not hx:
            continue
        item_bgr = hex_to_bgr(hx)
        d = dist(bgr, item_bgr)
        if d < best_dist:
            best_dist = d
            best_item = item

    if best_item is not None and best_dist <= max_dist:
        return best_item, best_dist
    return None, best_dist


def parse_source(src: str):
    """Parse full_page-02.png_left -> (page_num, side)."""
    m = re.match(r"full_page-(\d+)\.png(?:_(left|right))?$", src)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2) if m.group(2) else "single"


def load_raw_legend(raw_dir: str, page_num: int) -> List[str]:
    """Load filtered legend lines from raw/page-NN/legend.txt."""
    if page_num is None:
        return []
    path = os.path.join(raw_dir, f"page-{page_num:02d}", "legend.txt")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def safe_filename_part(text: str, max_len: int = 60) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in text).strip("_")
    safe = "_".join(filter(None, safe.split("_")))
    if len(safe) > max_len:
        safe = safe[:max_len].rstrip("_")
    return safe or "untitled"


def split_and_convert_to_jsonld(
    dataset_path: str,
    titles_path: str,
    output_dir: str,
    raw_dir: str = "raw",
):
    os.makedirs(output_dir, exist_ok=True)

    # Restructure: replace legacy title-only filenames with page_N_* reports
    for name in os.listdir(output_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(output_dir, name)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    titles = {}
    if os.path.exists(titles_path):
        with open(titles_path, "r", encoding="utf-8") as f:
            titles = json.load(f)

    legends = {}
    for leg_candidate in ("data/legends.json", "/app/data/legends.json"):
        if os.path.exists(leg_candidate):
            with open(leg_candidate, "r", encoding="utf-8") as f:
                legends = json.load(f)
            break

    # Cache raw/page-*/legend.txt by page number
    raw_legend_cache = {}

    grouped = {}
    for row in data:
        src = row["source"]
        loc = row["location"]
        bgr = row["color_bgr"]

        if src not in grouped:
            grouped[src] = {}
        grouped[src][loc] = {
            "bgr": bgr,
            "hex": bgr_to_hex(bgr),
            "color": get_color_name(bgr),
            "lon": row.get("lon"),
            "lat": row.get("lat"),
            "pixel_x": row.get("pixel_x"),
            "pixel_y": row.get("pixel_y"),
        }

    for src, points in grouped.items():
        page_num, side = parse_source(src)
        
        raw_title_data = titles.get(src, {})
        if isinstance(raw_title_data, str):
            raw_title = raw_title_data
            original_filename = src
            color_anomaly_mapping = {}
        else:
            raw_title = raw_title_data.get("title", f"Unknown ({src})")
            original_filename = raw_title_data.get("original_filename", src)
            color_anomaly_mapping = raw_title_data.get("color_anomaly_mapping", {})

        if page_num not in raw_legend_cache:
            raw_legend_cache[page_num] = load_raw_legend(raw_dir, page_num)
        page_legend_lines = raw_legend_cache[page_num]

        # Convert full_page-05.png_left to debug_full_page-05_left.png
        legend_key = "debug_" + src.replace(".png_left", "_left.png").replace(".png_right", "_right.png")
        raw_legend = legends.get(legend_key, [])
        if not raw_legend and page_num is not None:
            raw_legend = legends.get(f"debug_full_page-{page_num:02d}.png", [])

        # Enrich each legend item with hex code from COLOR_PALETTE if not already set
        legend = []
        page_color_palette = {}
        for item in raw_legend:
            item_copy = dict(item)
            c_key = item_copy.get("color", "")
            hx = item_copy.get("hex") or COLOR_PALETTE.get(c_key, "#cccccc")
            item_copy["hex"] = hx
            legend.append(item_copy)
            ar_or_vr = item_copy.get("absolute_range") if item_copy.get("absolute_range") not in (None, "N/A") else item_copy.get("value_range", "N/A")
            page_color_palette[hx] = {
                "color": c_key,
                "hex": hx,
                "value_range": item_copy.get("value_range", "N/A"),
                "absolute_range": item_copy.get("absolute_range", "N/A"),
                "label": item_copy.get("label", "N/A"),
                "range": ar_or_vr,
            }

        # Build a consolidated lookup for this map's color legend (merging multiple bins per color)
        legend_lookup = {}
        for item in legend:
            c_key = item.get("color")
            if not c_key:
                continue
            if c_key not in legend_lookup:
                legend_lookup[c_key] = {
                    "color": c_key,
                    "hex": item.get("hex", COLOR_PALETTE.get(c_key, "#cccccc")),
                    "hex_codes": [],
                    "value_ranges": [],
                    "absolute_ranges": [],
                    "labels": [],
                }
            hx = item.get("hex")
            vr = item.get("value_range", "N/A")
            ar = item.get("absolute_range", "N/A")
            lb = item.get("label", "N/A")
            if hx and hx not in legend_lookup[c_key]["hex_codes"]:
                legend_lookup[c_key]["hex_codes"].append(hx)
            if vr and vr != "N/A" and vr not in legend_lookup[c_key]["value_ranges"]:
                legend_lookup[c_key]["value_ranges"].append(vr)
            if ar and ar != "N/A" and ar not in legend_lookup[c_key]["absolute_ranges"]:
                legend_lookup[c_key]["absolute_ranges"].append(ar)
            if lb and lb != "N/A" and lb not in legend_lookup[c_key]["labels"]:
                legend_lookup[c_key]["labels"].append(lb)

        for c_key, agg in legend_lookup.items():
            agg["value_range"] = " / ".join(agg["value_ranges"]) if agg["value_ranges"] else "N/A"
            agg["absolute_range"] = " / ".join(agg["absolute_ranges"]) if agg["absolute_ranges"] else "N/A"
            agg["label"] = " / ".join(agg["labels"]) if agg["labels"] else "N/A"

        graph = []
        for loc_name, pt_info in points.items():
            bgr = pt_info["bgr"]
            obs_hex = pt_info["hex"]
            color_name = pt_info["color"]

            # Match observation BGR directly against this map's legend hex palette first
            matched_item, match_dist = match_legend_by_bgr(bgr, legend, max_dist=90.0)
            if matched_item is not None:
                color_name = matched_item.get("color", color_name)

            if color_name == "Unknown" and matched_item is None:
                continue

            safe_id = loc_name.replace(" ", "_")
            safe_id = (
                safe_id.replace("á", "%C3%A1")
                .replace("í", "%C3%AD")
                .replace("ó", "%C3%B3")
                .replace("é", "%C3%A9")
                .replace("ú", "%C3%BA")
                .replace("ñ", "%C3%B1")
            )

            obs = {
                "@type": "Observation",
                "location": f"http://maps2ai.org/locations/{safe_id}",
                "name": loc_name,
                "latitude": round(float(pt_info["lat"]), 5) if pt_info.get("lat") is not None else None,
                "longitude": round(float(pt_info["lon"]), 5) if pt_info.get("lon") is not None else None,
                "pixel_x": pt_info.get("pixel_x"),
                "pixel_y": pt_info.get("pixel_y"),
                "color": color_name,
                "hex": obs_hex,
            }
            if matched_item is not None and matched_item.get("hex"):
                obs["palette_hex"] = matched_item["hex"]

            # Map the color to the extracted anomaly using a fuzzy fallback
            anomaly = "Unknown"
            for k, v in color_anomaly_mapping.items():
                if k.lower() in color_name.lower() or color_name.lower() in k.lower():
                    anomaly = v
                    break
            obs["anomaly"] = anomaly

            if (color_name in ("Grey", "No Map") and matched_item is None) or (color_name == "Map White" and matched_item is None):
                obs["value_range"] = "No Map"
                obs["value"] = "No Map"
            else:
                mapping = {}
                
                # If we have an AI mapping, prioritize it!
                if color_anomaly_mapping:
                    if anomaly != "Unknown":
                        v_range = anomaly
                        label = "N/A"
                        absolute_val = "N/A"
                    else:
                        v_range = "Unknown"
                        label = "Unknown"
                        absolute_val = "N/A"
                elif matched_item is not None:
                    v_range = matched_item.get("value_range", "Unknown")
                    label = matched_item.get("label", "Unknown")
                    absolute_val = matched_item.get("absolute_range", "N/A")
                    if v_range in ("N/A", "Unknown", None) and absolute_val not in ("N/A", "Unknown", None):
                        v_range = absolute_val
                else:
                    mapping = legend_lookup.get(color_name)

                    # Equivalence fallback for Map Green / Green and Map White / White
                    if not mapping:
                        alt_map = {
                            "Map Green": ["Green", "Light Green", "Dark Green"],
                            "Map White": ["White", "Light Yellow"],
                        }
                        for alt_c in alt_map.get(color_name, []):
                            if alt_c in legend_lookup:
                                mapping = legend_lookup[alt_c]
                                break

                    # Fuzzy fallback if exact color is not found
                    if not mapping:
                        for k in legend_lookup.keys():
                            if color_name in k or k in color_name:
                                mapping = legend_lookup[k]
                                break

                    if not mapping:
                        mapping = {}

                    v_range = mapping.get("value_range", "Unknown")
                    label = mapping.get("label", "Unknown")
                    absolute_val = mapping.get("absolute_range", "N/A")
                    if v_range in ("N/A", "Unknown", None) and absolute_val not in ("N/A", "Unknown", None):
                        v_range = absolute_val

                # Determine if it's percentage or temperature based on the string
                if v_range is None or v_range == "N/A":
                    v_range = "Unknown"
                    
                if "%" in str(v_range):
                    obs["value_percentage"] = v_range
                elif "°C" in str(v_range) or "C" in str(v_range):
                    obs["value_temperature"] = v_range
                else:
                    obs["value_range"] = v_range

                if absolute_val != "N/A":
                    obs["absolute_range"] = absolute_val

                obs["label"] = label
                if absolute_val != "N/A" and absolute_val != v_range:
                    obs["value"] = f"{v_range} ({absolute_val}) ({label})" if label != "N/A" else f"{v_range} ({absolute_val})"
                else:
                    obs["value"] = f"{v_range} ({label})" if label not in ("N/A", "Unknown") else v_range
                if obs.get("anomaly") == "Unknown" and v_range != "Unknown":
                    obs["anomaly"] = f"{v_range} ({label})" if label not in ("N/A", "Unknown") else v_range

            graph.append(obs)

        output_data = {
            "@context": {
                "schema": "http://schema.org/",
                "Observation": "schema:Observation",
                "location": {"@id": "schema:location", "@type": "@id"},
                "latitude": "schema:latitude",
                "longitude": "schema:longitude",
                "pixel_x": "schema:position",
                "pixel_y": "schema:position",
                "value": "schema:value",
                "value_range": "schema:description",
                "value_percentage": "schema:description",
                "value_temperature": "schema:description",
                "absolute_range": "schema:value",
                "label": "schema:name",
                "hex": "schema:color",
                "palette_hex": "schema:color",
                "color_palette": "schema:hasPart",
                "variable": "schema:variableMeasured",
                "source": "schema:publisher",
                "legend": "schema:description",
                "legend_items": "schema:hasPart",
                "color_ranges": "schema:description",
                "page": "schema:position",
            },
            "source": src,
            "original_filename": original_filename,
            "page": page_num,
            "side": side,
            "variable_page_title": raw_title,
            "legend": page_legend_lines,
            "color_palette": page_color_palette if page_color_palette else COLOR_PALETTE,
            "legend_items": legend,
            "color_ranges": {
                c_k: {
                    "hex": c_v.get("hex", COLOR_PALETTE.get(c_k, "#cccccc")),
                    "hex_codes": c_v.get("hex_codes", []),
                    "value_range": c_v.get("value_range", "N/A"),
                    "absolute_range": c_v.get("absolute_range", "N/A"),
                    "label": c_v.get("label", "N/A"),
                }
                for c_k, c_v in legend_lookup.items()
            },
            "@graph": graph,
        }

        title_part = safe_filename_part(raw_title)
        if page_num is not None and side:
            filename = f"page_{page_num}_{side}_{title_part}.json"
        elif page_num is not None:
            filename = f"page_{page_num}_{title_part}.json"
        else:
            filename = f"page_unknown_{title_part}.json"

        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Wrote {filepath} (legend lines: {len(page_legend_lines)})")

    return len(grouped)


def compile_report(data_dir: str, output_csv: str):
    files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

    loaded_data = {}
    all_locations = set()

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = json.load(f)

        src = content["source"]
        title = content["variable_page_title"]

        points = {}
        for obs in content.get("@graph", []):
            loc_url = obs.get("location", "")
            loc_name = urllib.parse.unquote(loc_url.split("/")[-1]).replace("_", " ")
            v = obs.get("value", "No Map")
            vr = obs.get("value_range", "")
            points[loc_name] = {
                "value": vr if vr else v,
                "color": obs.get("color", "Unknown"),
                "hex": obs.get("hex", ""),
                "anomaly": obs.get("anomaly", "Unknown")
            }

        loaded_data[src] = {
            "title": title,
            "original_filename": content.get("original_filename", ""),
            "page": content.get("page"),
            "legend": content.get("legend", []),
            "points": points,
        }

        for loc in points.keys():
            all_locations.add(loc)

    def get_page_num(src_name):
        page = loaded_data[src_name].get("page")
        if page is not None:
            return int(page)
        m = re.search(r"page[_-](\d+)", src_name)
        if m:
            return int(m.group(1))
        return 999

    sorted_sources = sorted(loaded_data.keys(), key=lambda x: (get_page_num(x), x))
    sorted_locations = sorted(list(all_locations))

    headers = ["Location", "Page", "Side", "Variable", "Date", "Color", "Hex", "Anomaly", "Legend"]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        # Optional legend row summarizing raw/page-*/legend.txt per column
        legend_row = ["Legend"]
        for src in sorted_sources:
            lines = loaded_data[src].get("legend") or []
            legend_row.append(" | ".join(lines) if lines else "")
        writer.writerow(legend_row)

        for loc in sorted_locations:
            for src in sorted_sources:
                pt = loaded_data[src]["points"].get(loc, {})
                val = pt.get("value", "No Map")
                color = pt.get("color", "Unknown")
                hex_val = pt.get("hex", "")
                anomaly = pt.get("anomaly", "Unknown")
                page_val = loaded_data[src].get("page", "")
                _, side_val = parse_source(src)
                var_title = loaded_data[src].get("title", "")
                lines = loaded_data[src].get("legend") or []
                legend_str = " | ".join(lines[:3]) if lines else ""
                
                # Parse Date from original_filename (IMG-YYYYMMDD-...)
                orig_file = loaded_data[src].get("original_filename", "")
                date_str = "Unknown"
                m_date = re.search(r"IMG-(\d{4})(\d{2})(\d{2})", orig_file)
                if m_date:
                    date_str = f"{m_date.group(1)}-{m_date.group(2)}-{m_date.group(3)}"
                    
                row = [loc, page_val, side_val or "", var_title, date_str, color, hex_val, anomaly if anomaly != "Unknown" else val, legend_str]
                writer.writerow(row)
