import json
import urllib.parse
import os
import re
import csv
from typing import Dict, Any, List


def get_color_name(bgr):
    def dist(c1, c2):
        return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

    color_map = {
        (204, 204, 204): "Grey",
        (201, 239, 219): "Light Green",
        (158, 216, 179): "Green",
        (120, 194, 145): "Dark Green",
        (66, 151, 93): "Dark Green",
        (182, 240, 255): "Light Yellow",
        (50, 220, 230): "Yellow",
        (55, 217, 227): "Yellow",
        (139, 219, 255): "Light Orange",
        (181, 232, 253): "Light Orange",
        (47, 207, 216): "Orange",
        (144, 212, 253): "Orange",
        (46, 190, 230): "Dark Orange",
        (116, 197, 253): "Dark Orange",
        (96, 185, 253): "Red",
        (135, 232, 198): "Map Green",
        (131, 228, 194): "Map Green",
        (234, 234, 234): "Map White",
        (253, 254, 255): "Map White"
    }

    best_dist = float("inf")
    best_name = "Unknown"

    for c_bgr, name in color_map.items():
        d = dist(bgr, c_bgr)
        if d < best_dist and d < 80:
            best_dist = d
            best_name = name

    return best_name


def parse_source(src: str):
    """Parse full_page-02.png_left -> (page_num, side)."""
    m = re.match(r"full_page-(\d+)\.png(?:_(left|right))?$", src)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2) if m.group(2) else "single"


def load_raw_legend(raw_dir: str, page_num: int, side: str = None) -> List[str]:
    """Load filtered legend lines from raw/page-NN/side.txt or raw/page-NN/legend.txt."""
    if page_num is None:
        return []
    # If side-specific legend or txt exists, prefer it
    if side:
        side_legend = os.path.join(raw_dir, f"page-{page_num:02d}", f"{side}_legend.txt")
        if os.path.exists(side_legend):
            with open(side_legend, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                if lines:
                    return lines

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
    if os.path.exists("data/legends.json"):
        with open("data/legends.json", "r", encoding="utf-8") as f:
            legends = json.load(f)

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
            "color_name": get_color_name(bgr),
            "bgr": bgr
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

        # Convert full_page-05.png_left to debug_full_page-05_left.png
        legend_key = "debug_" + src.replace(".png_left", "_left.png").replace(".png_right", "_right.png")
        legend = legends.get(legend_key, [])

        # Build panel-specific legend: prefer structured color legend from this specific panel
        panel_legend_lines = []
        if legend:
            for item in legend:
                col = item.get("color", "")
                vr = item.get("value_range", "")
                ar = item.get("absolute_range", "")
                lb = item.get("label", "")
                parts = []
                if vr and vr != "N/A":
                    parts.append(str(vr))
                if ar and ar != "N/A":
                    parts.append(str(ar))
                if lb and lb != "N/A" and lb != "Unknown":
                    parts.append(str(lb))
                desc = " - ".join(parts) if parts else "N/A"
                panel_legend_lines.append(f"{col}: {desc}")

        # If no color legend was found for this panel, fallback to raw OCR legend text for the side/page
        if not panel_legend_lines:
            panel_legend_lines = load_raw_legend(raw_dir, page_num, side)

        # Build a fast lookup for this map's color legend (Gemini)
        legend_lookup = {item["color"]: item for item in legend}

        graph = []
        for loc_name, pt_info in points.items():
            color_name = pt_info["color_name"]
            pt_bgr = pt_info.get("bgr")

            if color_name == "Unknown":
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
                "color": color_name,
            }

            # Map the color to the extracted anomaly using a fuzzy fallback
            anomaly = "Unknown"
            for k, v in color_anomaly_mapping.items():
                if k.lower() in color_name.lower() or color_name.lower() in k.lower():
                    anomaly = v
                    break
            obs["anomaly"] = anomaly

            if color_name == "Grey" or color_name == "No Map":
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
                else:
                    # If legend entries have calibrated bgr values, match by closest BGR!
                    has_bgr_items = [it for it in legend if "bgr" in it]
                    if has_bgr_items and pt_bgr is not None:
                        def dist_bgr(c1, c2):
                            return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5
                        best_it = min(has_bgr_items, key=lambda it: dist_bgr(pt_bgr, it["bgr"]))
                        mapping = best_it
                        if "color" in best_it:
                            obs["color"] = best_it["color"]
                    else:
                        mapping = legend_lookup.get(color_name)

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

                # Determine if it's percentage or temperature based on the string
                if v_range is None:
                    v_range = "Unknown"

                # If v_range is N/A or Unknown but absolute_range exists (e.g. accumulated rainfall maps in mm), use absolute_range
                effective_val = v_range
                if (effective_val in ("N/A", "Unknown") or not effective_val) and absolute_val != "N/A":
                    effective_val = absolute_val
                    
                if "%" in str(effective_val):
                    obs["value_percentage"] = effective_val
                elif "°C" in str(effective_val) or "C" in str(effective_val):
                    obs["value_temperature"] = effective_val
                else:
                    obs["value_range"] = effective_val

                if absolute_val != "N/A":
                    obs["absolute_range"] = absolute_val

                obs["label"] = label
                parts = []
                if v_range not in ("N/A", "Unknown"):
                    parts.append(str(v_range))
                if absolute_val != "N/A":
                    parts.append(str(absolute_val))
                if label != "N/A" and label != "Unknown":
                    parts.append(str(label))

                if parts:
                    obs["value"] = " - ".join(parts)
                elif effective_val != "Unknown":
                    obs["value"] = effective_val
                else:
                    obs["value"] = "Unknown"

            graph.append(obs)

        output_data = {
            "@context": {
                "schema": "http://schema.org/",
                "Observation": "schema:Observation",
                "location": {"@id": "schema:location", "@type": "@id"},
                "value": "schema:value",
                "value_range": "schema:description",
                "value_percentage": "schema:description",
                "value_temperature": "schema:description",
                "absolute_range": "schema:value",
                "label": "schema:name",
                "variable": "schema:variableMeasured",
                "source": "schema:publisher",
                "legend": "schema:description",
                "page": "schema:position",
            },
            "source": src,
            "original_filename": original_filename,
            "page": page_num,
            "side": side,
            "variable_page_title": raw_title,
            "legend": panel_legend_lines,
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

        print(f"Wrote {filepath} (legend lines: {len(panel_legend_lines)})")

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
            ar = obs.get("absolute_range", "")
            
            # Prefer rich descriptive value, or value_range / absolute_range
            chosen_val = v
            if chosen_val in ("N/A", "Unknown", ""):
                if vr not in ("N/A", "Unknown", ""):
                    chosen_val = vr
                elif ar not in ("N/A", "Unknown", ""):
                    chosen_val = ar

            points[loc_name] = {
                "value": chosen_val,
                "color": obs.get("color", "Unknown"),
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

    # Determine headers for the flattened structure
    # With AI extraction, we process one map at a time, so there's usually just one source.
    first_src = sorted_sources[0] if sorted_sources else None
    map_title = loaded_data[first_src]["title"] if first_src else "Variable"
    
    headers = ["Location", map_title, "Date", "Color", "Anomaly"]

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
                anomaly = pt.get("anomaly", "Unknown")
                
                # Parse Date from original_filename (IMG-YYYYMMDD-...)
                orig_file = loaded_data[src].get("original_filename", "")
                date_str = "Unknown"
                m_date = re.search(r"IMG-(\d{4})(\d{2})(\d{2})", orig_file)
                if m_date:
                    date_str = f"{m_date.group(1)}-{m_date.group(2)}-{m_date.group(3)}"
                    
                row = [loc, val, date_str, color, anomaly]
                writer.writerow(row)
