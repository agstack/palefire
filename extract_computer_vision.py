import os
import sys
import re
import json
import base64
import argparse
import requests
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from rdflib import Graph, Namespace, URIRef, Literal, RDF

def get_odrl_log_target(policy_path):
    try:
        with open(policy_path, 'r', encoding='utf-8') as f:
            policy = json.load(f)
        for duty in policy.get('duty', []):
            if duty.get('action') == 'log' and duty.get('assignee') == 'system':
                target = duty.get('target', '')
                if target.startswith('file:'):
                    return target.split('file:')[1]
    except Exception:
        pass
    return None
    
def append_to_log(log_file, action, mode, detail):
    if not log_file:
        return
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().isoformat()}] ACTION: {action} | MODE: {mode}\n{detail}\n{'-'*40}\n")
    except Exception as e:
        print(f"Failed to write to log file {log_file}: {e}")

def preprocess_image(image_path, session_dir, mode='light'):
    """
    Uses ImageMagick to upscale and sharpen the image.
    The specific filter arguments are loaded dynamically from the ODRL policy file.
    """
    processed_dir = Path(session_dir) / "processed_images"
    processed_dir.mkdir(parents=True, exist_ok=True)
    enhanced_path = processed_dir / f"enhanced_{mode}_{Path(image_path).name}"
    
    # Load transformation args from ODRL
    policy_file = Path("policies/cv_extraction_policy.jsonld")
    magick_args = load_odrl_policy(policy_file, mode, "tool:imagemagick", action="preprocess", constraint_key="arguments")
    
    if not magick_args:
        print(f"Warning: No ODRL preprocessing arguments found for mode '{mode}'. Skipping ImageMagick.")
        return image_path
        
    # Basic security validation: restrict to known safe ImageMagick flags
    safe_flags = {"-resize", "-contrast-stretch", "-sharpen", "-colorspace", "-level", "-unsharp", "-normalize", "gray"}
    for arg in magick_args:
        if str(arg).startswith("-") and str(arg) not in safe_flags:
            print(f"Warning: Unsafe ImageMagick flag detected in ODRL policy: {arg}")
            return image_path

    
    try:
        # Try magick (ImageMagick v7) first
        cmd = ["magick", str(image_path)] + magick_args + [str(enhanced_path)]
        subprocess.run(cmd, check=True, capture_output=True)
        log_file = get_odrl_log_target(policy_file)
        append_to_log(log_file, "preprocess", mode, f"ImageMagick arguments applied: {magick_args}")
        return enhanced_path
    except FileNotFoundError:
        try:
            # Fallback to convert (ImageMagick v6)
            cmd = ["convert", str(image_path)] + magick_args + [str(enhanced_path)]
            subprocess.run(cmd, check=True, capture_output=True)
            log_file = get_odrl_log_target(policy_file)
            append_to_log(log_file, "preprocess", mode, f"ImageMagick (convert) arguments applied: {magick_args}")
            return enhanced_path
        except Exception as e2:
            print(f"Warning: ImageMagick (convert) preprocessing failed: {e2}")
            return image_path
    except Exception as e:
        print(f"Warning: ImageMagick preprocessing failed: {e}")
        return image_path

def load_odrl_policy(policy_path, target_mode, assignee_model, action="extract", constraint_key="prompt"):
    """
    Loads an ODRL policy and extracts the constraint for the given mode, assignee, and action.
    """
    try:
        with open(policy_path, 'r', encoding='utf-8') as f:
            policy = json.load(f)
            
        target = f"image:{target_mode}"
        
        for duty in policy.get('duty', []):
            targets = duty.get('target', [])
            if not isinstance(targets, list):
                targets = [targets]
                
            if action == duty.get('action') and target in targets and assignee_model == duty.get('assignee'):
                for constraint in duty.get('constraint', []):
                    if constraint.get('leftOperand') == constraint_key and constraint.get('operator') == 'eq':
                        return constraint.get('rightOperand')
                        
        print(f"Warning: No specific ODRL policy found for action={action}, target={target}, assignee={assignee_model}.")
    except Exception as e:
        print(f"Warning: Could not load ODRL policy from {policy_path}: {e}")
        
    return None

def analyze_image(image_path, session_dir, model="gemma4:latest"):
    """
    Analyzes an image using a local Ollama vision model.
    Implements a two-pass retry loop for empty calendar cells using heavy ImageMagick preprocessing.
    """
    executed_actions = []
    
    # Save the original image to the session folder
    original_dir = Path(session_dir) / "original_images"
    original_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    try:
        shutil.copy2(image_path, original_dir / Path(image_path).name)
    except Exception as e:
        print(f"Warning: Could not copy original image: {e}")
        
    def _run_pass(mode):
        import ollama
        enhanced_path = preprocess_image(image_path, session_dir, mode=mode)
        
        # Load the dynamic ODRL constraints/prompts from policy engine
        policy_file = Path("policies/cv_extraction_policy.jsonld")
        current_prompt = load_odrl_policy(policy_file, mode, f"model:{model}", action="extract", constraint_key="prompt")
        if not current_prompt and model != "gemma4:latest":
            # Fallback to gemma4:latest policy prompt if specific model not defined in ODRL policy
            current_prompt = load_odrl_policy(policy_file, mode, "model:gemma4:latest", action="extract", constraint_key="prompt")
        
        if not current_prompt:
            current_prompt = "Classify the contents of this image. Do not include extra text outside of the JSON block."
            
        print(f"Analyzing image: {image_path} (mode: {mode}) using local Ollama model: {model}...")
        try:
            response = ollama.generate(
                model=model,
                prompt=current_prompt,
                images=[str(Path(image_path).resolve()), str(Path(enhanced_path).resolve())],
                options={"num_ctx": 8192}
            )
            if enhanced_path != Path(image_path) and enhanced_path.exists():
                enhanced_path.unlink()
                
            result_text = response.get('response', '').strip()
            print(f"\n--- RAW MODEL OUTPUT ({mode.upper()}) ---\n{result_text}\n------------------------\n")
            
            json_match = re.search(r'```json\s*(\{.*\})\s*```', result_text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*\})', result_text, re.DOTALL)
                
            log_file = get_odrl_log_target(policy_file)
            append_to_log(log_file, "extract", mode, f"Model Output:\n{result_text}")
                
            if json_match:
                try:
                    return json.loads(json_match.group(1)), result_text
                except json.JSONDecodeError:
                    pass
            return None, result_text
        except Exception as e:
            print(f"Error during {mode} pass: {e}")
            if 'enhanced_path' in locals() and enhanced_path != Path(image_path) and enhanced_path.exists():
                enhanced_path.unlink()
            return None, str(e)

    try:
        # Pass 1: Light Mode
        needs_pass_2 = False
        extracted_data, raw_text = _run_pass('light')
        executed_actions.append({
            "odrl:action": "extract",
            "odrl:assignee": f"model:{model}",
            "mode": "light",
            "result": raw_text
        })
        
        if extracted_data:
            img_type = extracted_data.get('type')
            
            if img_type == 'precipitation_calendar':
                measurements = extracted_data.get('measurements', {})
                empty_count = sum(1 for d in range(1, 32) if not str(measurements.get(str(d), '')).strip())
                if empty_count > 0:
                    needs_pass_2 = True
            elif img_type == 'multi_month_calendar':
                data = extracted_data.get('data', {})
                for m, meas in data.items():
                    if any(not str(meas.get(str(d), '')).strip() for d in range(1, 32)):
                        needs_pass_2 = True
                        break
            
            # Pass 2: Heavy Mode Merge
            if needs_pass_2:
                print(f"Empty cells detected in {image_path}. Triggering heavy preprocessing retry loop...")
                heavy_data, heavy_raw = _run_pass('heavy')
                executed_actions.append({
                    "odrl:action": "extract",
                    "odrl:assignee": f"model:{model}",
                    "mode": "heavy",
                    "result": heavy_raw
                })
                
                if heavy_data and heavy_data.get('type') == img_type:
                    if img_type == 'precipitation_calendar':
                        final_measurements = extracted_data.setdefault('measurements', {})
                        heavy_measurements = heavy_data.get('measurements', {})
                        for day, heavy_val in heavy_measurements.items():
                            # If the heavy pass found a non-empty value, we prioritize it
                            if heavy_val and str(heavy_val).strip():
                                light_val = final_measurements.get(day, "")
                                if light_val != heavy_val:
                                    if not light_val:
                                        print(f"Merge: Recovered empty day {day} from heavy pass: {heavy_val}")
                                    else:
                                        print(f"Merge: Overwrote day {day} light value ({light_val}) with heavy value: {heavy_val}")
                                    final_measurements[day] = heavy_val
                    elif img_type == 'multi_month_calendar':
                        light_data = extracted_data.setdefault('data', {})
                        heavy_data_dict = heavy_data.get('data', {})
                        for m in set(list(light_data.keys()) + list(heavy_data_dict.keys())):
                            light_m = light_data.setdefault(m, {})
                            heavy_m = heavy_data_dict.get(m, {})
                            for d in range(1, 32):
                                ds = str(d)
                                if not str(light_m.get(ds, '')).strip() and str(heavy_m.get(ds, '')).strip():
                                    print(f"Merge: Recovered value for month {m} day {ds} from heavy pass: {heavy_m[ds]}")
                                    light_m[ds] = heavy_m[ds]
            
            # Format output
            if img_type == 'photo':
                result_text = f"Photo Description: {extracted_data.get('description', 'Unknown')}"
            elif img_type == 'other_log':
                result_text = f"Data Log Document: {extracted_data.get('description', 'Unknown')}"
            elif img_type == 'map':
                result_text = f"Map Document: {extracted_data.get('description', 'Unknown')}"
            elif img_type == 'chart':
                result_text = f"Chart Document: {extracted_data.get('variable_name', 'Unknown')}"
            elif img_type == 'multi_month_calendar':
                formatted_analysis = f"Ano: {extracted_data.get('year', 'Unknown')}\n"
                data = extracted_data.get('data', {})
                for month, measurements in data.items():
                    formatted_analysis += f"\nMes: {month}\n"
                    for day in range(1, 32):
                        val = measurements.get(str(day), '')
                        val_str = str(val).lower().replace('ml', '').replace('l', '').replace('mm', '').replace('m', '').strip()
                        if val_str:
                            formatted_analysis += f"{day}: {val_str}mm\n"
                        else:
                            formatted_analysis += f"{day}: \n"
                result_text = formatted_analysis.strip()
            else:
                formatted_analysis = (
                    f"Mes: {extracted_data.get('month', 'Unknown')}, "
                    f"Ano: {extracted_data.get('year', 'Unknown')}\n"
                )
                measurements = extracted_data.get('measurements', {})
                for day in range(1, 32):
                    val = measurements.get(str(day), '')
                    val_str = str(val).lower().replace('ml', '').replace('l', '').replace('mm', '').replace('m', '').strip()
                    if val_str:
                        formatted_analysis += f"{day}: {val_str}mm\n"
                    else:
                        formatted_analysis += f"{day}: \n"
                    
                result_text = formatted_analysis.strip()
        else:
            # Fallback to raw text if no JSON was found
            # Fallback to raw text if no JSON was found
            lines = raw_text.split('\n')
            clean_lines = [l for l in lines if not l.startswith('added image')]
            result_text = '\n'.join(clean_lines).strip()
            
        # Map Spatial Extraction Step
        if extracted_data and extracted_data.get('type') == 'map':
            map_count = int(extracted_data.get('map_count', 1)) if isinstance(extracted_data, dict) else 1
            split_grid_extracted = extracted_data.get('split_grid', '').strip() if isinstance(extracted_data, dict) else ''
            
            if (map_count > 1 or (split_grid_extracted != '' and split_grid_extracted != '1x1')) and not Path(image_path).name.startswith("split_"):
                # User explicitly requested: ALWAYS keep the split as 2x1 (vertical split into left and right halves) regardless of map count or LLM suggestion.
                split_grid = "2x1"
                
                print(f"\nMultiple maps detected ({map_count}, {split_grid_extracted}). Splitting image using grid {split_grid}...")
                processed_dir = Path(session_dir) / "processed_images"
                processed_dir.mkdir(parents=True, exist_ok=True)
                split_base = processed_dir / f"split_{Path(image_path).stem}_%d.jpg"
                
                try:
                    subprocess.run(['magick', str(image_path), '-crop', '50%x100%', '+repage', str(split_base)], check=True)
                except Exception as e:
                    try:
                        subprocess.run(["convert", str(image_path), "-crop", "50%x100%", "+repage", str(split_base)], check=True)
                    except Exception as e2:
                        print(f"Failed to split map: {e2}")
                        return None
                
                import glob
                split_episodes = []
                split_files = sorted(glob.glob(str(processed_dir / f"split_{Path(image_path).stem}_*.jpg")))
                for split_file in split_files:
                    print(f"Analyzing split part: {split_file}")
                    ep = analyze_image(Path(split_file), session_dir, model=model)
                    if ep:
                        if isinstance(ep, list):
                            split_episodes.extend(ep)
                        else:
                            split_episodes.append(ep)
                return split_episodes

            policy_file = Path("policies/cv_extraction_policy.jsonld")
            with open(policy_file, 'r', encoding='utf-8') as f:
                policy = json.load(f)
            
            spatial_allowed = False
            for duty in policy.get('duty', []):
                if duty.get('action') == 'extract_spatial' and duty.get('assignee') == 'tool:docker_datamaps':
                    spatial_allowed = True
                    break
            
            if spatial_allowed:
                print(f"\nRecognized as Map. Running spatial extraction via docker_datamaps...")
                datamaps_dir = Path("datamaps")
                
                # Cleanup previous state
                import shutil
                for d in ['report', 'data_points', 'raw']:
                    d_path = datamaps_dir / d
                    if d_path.exists():
                        shutil.rmtree(d_path)
                        
                # Just remove extracted images/pdfs from data without deleting legends.json
                data_path = datamaps_dir / 'data'
                for f in data_path.glob('*.*'):
                    if f.is_file() and f.name != 'legends.json' and f.name != 'pdf_full_dataset.json':
                        f.unlink()
                
                datamaps_report = datamaps_dir / "report"
                datamaps_report.mkdir(parents=True, exist_ok=True)
                
                png_target = datamaps_report / "full_page-01.png"
                try:
                    subprocess.run(['convert', str(image_path), str(png_target)], check=True)
                    
                    map_var_name = "Map"
                    if isinstance(extracted_data, dict):
                        map_var_name = extracted_data.get('variable_name', 'Map')
                        
                    titles_file = datamaps_report / 'titles.json'
                    
                    titles_data = {
                        "full_page-01.png": {
                            "title": map_var_name,
                            "original_filename": image_path.name,
                            "color_anomaly_mapping": extracted_data.get('color_anomaly_mapping', {}) if isinstance(extracted_data, dict) else {}
                        }
                    }
                    
                    # Add expected split suffixes so datamaps can find metadata for split parts
                    for suffix in ["_left", "_right", "_top", "_bottom", "_top_left", "_top_right", "_bottom_left", "_bottom_right"]:
                        titles_data[f"full_page-01.png{suffix}"] = titles_data["full_page-01.png"].copy()
                        titles_data[f"full_page-01{suffix}.png"] = titles_data["full_page-01.png"].copy()
                    
                    with open(titles_file, 'w', encoding='utf-8') as f:
                        json.dump(titles_data, f, indent=2)
                        
                    map_count = extracted_data.get('map_count', 1) if isinstance(extracted_data, dict) else 1
                    env_args = ['-e', 'SINGLE_MAP=1']
                    if str(map_count) == '2' or int(map_count) > 1:
                        env_args = ['-e', 'SINGLE_MAP=0']
                    ai_datamaps_dir = None
                    try:
                        import shutil
                        import os
                        compose_cmd = ['docker-compose'] if shutil.which('docker-compose') else ['docker', 'compose']
                        
                        subprocess.run(compose_cmd + ['run', '--rm'] + env_args + ['app', 'bash', '-c', 
                            'python -m datamaps extract && python -m datamaps split && python -m datamaps compile'], 
                            cwd=str(datamaps_dir), check=True)
                            
                        if hasattr(os, 'getuid'):
                            uid = os.getuid()
                            gid = os.getgid()
                            subprocess.run(compose_cmd + ['run', '--rm', 'app', 'bash', '-c', 
                                f'chown -R {uid}:{gid} /app/data /app/data_points /app/report'], 
                                cwd=str(datamaps_dir))
                            
                        ai_datamaps_dir = Path(session_dir) / f"{Path(image_path).stem}_datamaps"
                        ai_datamaps_dir.mkdir(parents=True, exist_ok=True)
                        
                        subprocess.run(['cp', '-r', str(datamaps_dir / 'data_points'), str(ai_datamaps_dir / 'data_points')], check=True)
                        subprocess.run(['cp', str(datamaps_report / 'final_heatmap_report.csv'), str(ai_datamaps_dir / 'final_heatmap_report.csv')], check=True)
                        
                        executed_actions.append({
                            "odrl:action": "extract_spatial",
                            "odrl:assignee": "tool:docker_datamaps",
                            "mode": "spatial",
                            "result": f"Successfully extracted spatial data to {ai_datamaps_dir}"
                        })
                    except Exception as inner_e:
                        print(f"Error in inner datamaps execution: {inner_e}")
                except Exception as e:
                    print(f"Error running datamaps: {e}")
                    executed_actions.append({
                        "odrl:action": "extract_spatial",
                        "odrl:assignee": "tool:docker_datamaps",
                        "mode": "spatial",
                        "error": str(e)
                    })
                    
            # NEW: Extract OCR Location Values using LLM
            loc_val_allowed = False
            loc_val_prompt = None
            for duty in policy.get('duty', []):
                if duty.get('action') == 'extract_ocr_location_values' and duty.get('assignee') == 'model:gemma4:latest':
                    loc_val_allowed = True
                    for constraint in duty.get('constraint', []):
                        if constraint.get('leftOperand') == 'prompt':
                            loc_val_prompt = constraint.get('rightOperand')
                    break
            
            if loc_val_allowed and loc_val_prompt:
                print(f"\nRecognized as Map. Extracting location values via OCR + LLM...")
                try:
                    import ollama
                    import cv2
                    import geopandas as gpd
                    import numpy as np
                    from unidecode import unidecode
                    
                    locations = extracted_data.get('extracted_locations', []) if isinstance(extracted_data, dict) else []
                    has_coords = extracted_data.get('has_coordinates', False) if isinstance(extracted_data, dict) else False
                    mapping = extracted_data.get('color_anomaly_mapping', {}) if isinstance(extracted_data, dict) else {}
                    
                    if (locations or has_coords) and mapping:
                        ocr_res = subprocess.run(["tesseract", str(Path(image_path).resolve()), "stdout", "--psm", "11", "tsv"], capture_output=True, text=True)
                        lines = ocr_res.stdout.strip().split('\n')
                        
                        img_annotated = cv2.imread(str(Path(image_path).resolve()))
                        marked_locations = []
                        
                        # Load GeoJSON and prep centroids
                        try:
                            gdf = gpd.read_file('datamaps/honduras_departments.geojson')
                            departments = {}
                            for _, row in gdf.iterrows():
                                if row.geometry is None: continue
                                name = unidecode(row.get("shapeName", "")).strip().lower()
                                centroid = row.geometry.centroid
                                departments[name] = (centroid.x, centroid.y)
                                departments[name + "_orig"] = row.get("shapeName", "")
                        except Exception as e:
                            print(f"Warning: Could not load GeoJSON for mapping: {e}")
                            departments = {}
                        
                        src_pts = []
                        dst_pts = []
                        fallback_pixels = []
                        fallback_names = []
                        
                        if len(lines) > 1:
                            header = lines[0].split('\t')
                            try:
                                text_idx = header.index("text")
                                left_idx = header.index("left")
                                top_idx = header.index("top")
                                width_idx = header.index("width")
                                height_idx = header.index("height")
                                
                                for line in lines[1:]:
                                    cols = line.split('\t')
                                    if len(cols) > text_idx:
                                        text = cols[text_idx].strip()
                                        if len(text) < 3:
                                            continue
                                        norm_text = unidecode(text).strip().lower()
                                        
                                        # Find bounding box center
                                        try:
                                            x = int(cols[left_idx])
                                            y = int(cols[top_idx])
                                            w = int(cols[width_idx])
                                            h = int(cols[height_idx])
                                            center_x = x + w // 2
                                            center_y = y + h + 15
                                        except ValueError:
                                            continue
                                            
                                        # Track for fallback
                                        for loc in locations:
                                            if unidecode(loc).strip().lower() in norm_text or norm_text in unidecode(loc).strip().lower():
                                                fallback_pixels.append((center_x, center_y))
                                                fallback_names.append(loc)
                                                break
                                                
                                        # Track for Affine matching against GeoJSON
                                        matched_dept = None
                                        for d_name in departments.keys():
                                            if not d_name.endswith("_orig") and (d_name in norm_text or norm_text in d_name):
                                                matched_dept = d_name
                                                break
                                                
                                        if matched_dept:
                                            geo_lon, geo_lat = departments[matched_dept]
                                            src_pts.append([geo_lon, geo_lat])
                                            dst_pts.append([center_x, center_y])
                                            
                            except ValueError:
                                pass
                                
                        # Affine Projection
                        matrix = None
                        if len(src_pts) >= 3:
                            src_pts_np = np.array(src_pts, dtype=np.float32)
                            dst_pts_np = np.array(dst_pts, dtype=np.float32)
                            matrix, inliers = cv2.estimateAffinePartial2D(src_pts_np, dst_pts_np)
                            
                        if matrix is None:
                            print(f"Could not compute Affine Transformation from OCR names (found {len(src_pts)} matching departments).")
                            
                            # NEW: Try to parse coordinate axes (lat/lon) from the OCR lines
                            lat_pts_axis = []
                            lon_pts_axis = []
                            for line in lines[1:]:
                                parts = line.split('\t')
                                if len(parts) < 12: continue
                                text_clean = parts[11].strip().upper().replace('O', '0').replace('{', '1')
                                if not text_clean: continue
                                x, y, w, h = map(int, parts[6:10])
                                cx, cy = x + w//2, y + h//2
                                
                                m_lat = re.search(r'^(\d+)(N|S|5)$', text_clean)
                                if m_lat:
                                    val = int(m_lat.group(1))
                                    if val in [10, 20, 30, 40, 50, 60, 70, 80]:
                                        if m_lat.group(2) in ['S', '5']: val = -val
                                        lat_pts_axis.append((val, cy)) # (geo, pixel)
                                        
                                m_lon = re.search(r'^(\d+)(E|W)$', text_clean)
                                if m_lon:
                                    val = int(m_lon.group(1))
                                    if val % 10 == 0:
                                        if m_lon.group(2) == 'W': val = -val
                                        lon_pts_axis.append((val, cx))
                                        
                            # Filter out axes from the bottom half of the image to prevent regression breaking on vertically stacked panels
                            h, w = img_annotated.shape[:2]
                            lat_pts_axis = [(val, cy) for val, cy in lat_pts_axis if cy < h * 0.6]
                            lon_pts_axis = [(val, cx) for val, cx in lon_pts_axis if cx < w] # Keep all valid lons for top map
                            
                            if len(lat_pts_axis) >= 2 and len(lon_pts_axis) >= 2:
                                print(f"Found {len(lat_pts_axis)} lat points and {len(lon_pts_axis)} lon points on axes. Using Axis Regression.")
                                
                                # Unwrap longitudes if map crosses dateline
                                has_positive_lon = any(v > 0 for v, _ in lon_pts_axis)
                                has_negative_lon = any(v < 0 for v, _ in lon_pts_axis)
                                if has_positive_lon and has_negative_lon:
                                    lon_pts_axis = [((v + 360) if v < 0 else v, p) for v, p in lon_pts_axis]
                                
                                m_y, b_y = np.polyfit([p[0] for p in lat_pts_axis], [p[1] for p in lat_pts_axis], 1)
                                m_x, b_x = np.polyfit([p[0] for p in lon_pts_axis], [p[1] for p in lon_pts_axis], 1)
                                
                                # We can construct a fallback_src and fallback_dst to compute a matrix, or just override the matrix logic later.
                                # Let's construct a matrix using 3 synthetic points.
                                syn_lons = [lon_pts_axis[0][0], lon_pts_axis[0][0], lon_pts_axis[1][0]]
                                syn_lats = [lat_pts_axis[0][0], lat_pts_axis[1][0], lat_pts_axis[0][0]]
                                
                                # If original longitudes were unwrapped, we need to make sure we unwrap the GeoJSON points later.
                                # To avoid changing later code, we'll just use the synthetic points as the matrix anchor.
                                # BUT we must remember to unwrap the target geo_lon if this map crosses dateline.
                                # A safer way is to just generate the transformation for the unwrapped space,
                                # and we will monkey-patch the projection loop if matrix has a special flag.
                                # We can just set matrix = "AXIS_REGRESSION" and store m_x, b_x, m_y, b_y
                                matrix = "AXIS_REGRESSION"
                                axis_transform = (m_x, b_x, m_y, b_y, has_positive_lon and has_negative_lon)
                            else:
                                print("Fallback to fixed bounding box mapping.")
                                h, img_w = img_annotated.shape[:2]
                                pixel_bounds = (img_w//4, h//4, img_w//2, h//2) # Rough centered bounding box
                            
                                min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
                                fallback_src = np.float32([
                                    [min_lon, max_lat],
                                    [max_lon, min_lat],
                                    [min_lon, min_lat]
                                ])
                                fallback_dst = np.float32([
                                    [pixel_bounds[0], pixel_bounds[1]],          
                                    [pixel_bounds[0] + pixel_bounds[2], pixel_bounds[1] + pixel_bounds[3]],  
                                    [pixel_bounds[0], pixel_bounds[1] + pixel_bounds[3]]       
                                ])
                                matrix = cv2.getAffineTransform(fallback_src, fallback_dst)
                            
                        # Keep original color image for sampling
                        img_color_original = img_annotated.copy()
                        
                        # Convert background map to grayscale
                        gray = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2GRAY)
                        img_annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        
                        if matrix is not None:
                            print(f"Calculated Affine Transformation. Projecting ALL GeoJSON points.")
                            
                            STANDARD_COLORS_BGR = {
                                "Red": (0, 0, 255),
                                "Dark Orange": (0, 69, 255),
                                "Orange": (0, 165, 255),
                                "Light Orange": (0, 200, 255),
                                "Yellow": (0, 255, 255),
                                "Light Yellow": (153, 255, 255),
                                "White": (255, 255, 255),
                                "Map White": (255, 255, 255),
                                "Cyan": (255, 255, 0),
                                "Light Blue": (255, 128, 0),
                                "Blue": (255, 0, 0),
                                "Dark Blue": (139, 0, 0),
                                "Light Green": (144, 238, 144),
                                "Green": (0, 128, 0),
                                "Dark Green": (0, 100, 0),
                                "Grey": (128, 128, 128)
                            }
                            calculated_colors = {}
                            
                            for d_name, val in departments.items():
                                if d_name.endswith("_orig"): continue
                                geo_lon, geo_lat = val
                                orig_name = departments.get(d_name + "_orig", d_name)
                                
                                if isinstance(matrix, str) and matrix == "AXIS_REGRESSION":
                                    m_x, b_x, m_y, b_y, wrap_lon = axis_transform
                                    geo_lon_cont = (geo_lon + 360) if (wrap_lon and geo_lon < 0) else geo_lon
                                    px = int(m_x * geo_lon_cont + b_x)
                                    py = int(m_y * geo_lat + b_y)
                                else:
                                    pt = np.array([geo_lon, geo_lat, 1.0])
                                    res = np.dot(matrix, pt)
                                    px, py = int(res[0]), int(res[1])
                                
                                # Bound check
                                h, w = img_annotated.shape[:2]
                                if 0 <= px < w and 0 <= py < h:
                                    # Sample color from original
                                    sampled_color = img_color_original[py, px]
                                    color = (int(sampled_color[0]), int(sampled_color[1]), int(sampled_color[2]))
                                else:
                                    color = (0, 0, 255) # Fallback red
                                    
                                min_dist = float('inf')
                                best_color = "Unknown"
                                for k, c_bgr in STANDARD_COLORS_BGR.items():
                                    dist = sum((a - b)**2 for a, b in zip(color, c_bgr))
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_color = k
                                calculated_colors[orig_name] = best_color
                                    
                                # Draw filled dot with sampled color
                                cv2.circle(img_annotated, (px, py), 12, color, -1) 
                                
                                marked_locations.append(orig_name)
                                
                            # Also plot any fallback OCR pixels that didn't match the GeoJSON (like country names)
                            for (px, py), name in zip(fallback_pixels, fallback_names):
                                h, w = img_annotated.shape[:2]
                                if 0 <= px < w and 0 <= py < h:
                                    sampled_color = img_color_original[py, px]
                                    color = (int(sampled_color[0]), int(sampled_color[1]), int(sampled_color[2]))
                                else:
                                    color = (0, 0, 255)
                                    
                                min_dist = float('inf')
                                best_color = "Unknown"
                                for k, c_bgr in STANDARD_COLORS_BGR.items():
                                    dist = sum((a - b)**2 for a, b in zip(color, c_bgr))
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_color = k
                                calculated_colors[name] = best_color
                                    
                                cv2.circle(img_annotated, (px, py), 12, color, -1)
                                marked_locations.append(name)
                                
                        if marked_locations:
                            annotated_path = str(Path(image_path).resolve()).replace(".jpg", "_annotated.jpg")
                            cv2.imwrite(annotated_path, img_annotated)
                            shutil.copy2(annotated_path, session_dir / Path(annotated_path).name)
                            
                            dynamic_prompt = f"{loc_val_prompt}\n\nLocations marked: {json.dumps(list(set(marked_locations)))}\nLegend mapping: {json.dumps(mapping)}"
                            
                            loc_resp = ollama.generate(
                                model=model,
                                prompt=dynamic_prompt,
                                images=[annotated_path],
                                options={"num_ctx": 8192}
                            )
                            loc_raw = loc_resp.get('response', '').strip()
                            
                            executed_actions.append({
                                "odrl:action": "extract_ocr_location_values",
                                "odrl:assignee": f"model:{model}",
                                "mode": "ocr_location_value_extraction",
                                "result": loc_raw
                            })
                            print(f"--- OCR LOCATION VALUE EXTRACTION OUTPUT ---\n{loc_raw}\n------------------------\n")
                            log_file = get_odrl_log_target(policy_file)
                            append_to_log(log_file, "extract_ocr_location_values", "ocr_location_value_extraction", f"Output:\n{loc_raw}")
                            result_text += f"\nOCR Location Values (LLM):\n{loc_raw}"
                            
                            # Merge LLM extracted values and calculated colors back into the CSV
                            try:
                                loc_data = json.loads(loc_raw.replace('```json', '').replace('```', ''))
                                if 'location_values' in loc_data and ai_datamaps_dir is not None:
                                    csv_path = ai_datamaps_dir / 'final_heatmap_report.csv'
                                    if csv_path.exists():
                                        import pandas as pd
                                        df = pd.read_csv(csv_path)
                                        update_dict = {item['location']: item['value'] for item in loc_data['location_values']}
                                        
                                        if 'Color' not in df.columns:
                                            df['Color'] = 'Unknown'
                                            
                                        if 'Anomaly' in df.columns and 'Location' in df.columns:
                                            df['Anomaly'] = df.apply(lambda row: update_dict.get(row['Location'], row['Anomaly']), axis=1)
                                            df['Color'] = df.apply(lambda row: calculated_colors.get(row['Location'], row['Color']), axis=1)
                                            
                                            # Also mathematically map color to Anomaly if it's in the legend mapping
                                            flat_mapping = {}
                                            if mapping:
                                                for k, v in mapping.items():
                                                    if isinstance(v, dict):
                                                        flat_mapping.update(v)
                                                    else:
                                                        flat_mapping[k] = v
                                            for idx, row in df.iterrows():
                                                c = row['Color']
                                                if c in flat_mapping:
                                                    df.at[idx, 'Anomaly'] = flat_mapping[c]
                                                    
                                            df.to_csv(csv_path, index=False)
                                            print(f"Merged OCR values and calculated colors into {csv_path}")
                            except Exception as e:
                                print(f"Failed to merge values into CSV: {e}")
                        else:
                            print("No extracted locations found by OCR.")
                    else:
                        print("Skipping OCR location value extraction: no locations or mappings found in primary extraction.")
                except Exception as e:
                    print(f"OCR Location value extraction failed: {e}")
                    executed_actions.append({
                        "odrl:action": "extract_ocr_location_values",
                        "odrl:assignee": f"model:{model}",
                        "mode": "ocr_location_value_extraction",
                        "error": str(e)
                    })

        # Chart Data Extraction Step
        elif extracted_data and extracted_data.get('type') == 'chart':
            chart_allowed = False
            chart_prompt = None
            policy_file = Path("policies/cv_extraction_policy.jsonld")
            with open(policy_file, 'r', encoding='utf-8') as f:
                policy = json.load(f)
            for duty in policy.get('duty', []):
                if duty.get('action') == 'extract_chart' and duty.get('assignee') == 'model:gemma4:latest':
                    chart_allowed = True
                    for constraint in duty.get('constraint', []):
                        if constraint.get('leftOperand') == 'prompt':
                            chart_prompt = constraint.get('rightOperand')
                    break
            
            if chart_allowed and chart_prompt:
                print(f"\nRecognized as Chart. Running chart data extraction via gemma4:latest...")
                try:
                    import ollama
                    chart_resp = ollama.generate(
                        model=model,
                        prompt=chart_prompt,
                        images=[str(Path(image_path).resolve())],
                        options={"num_ctx": 8192}
                    )
                    chart_raw = chart_resp.get('response', '').strip()
                    
                    executed_actions.append({
                        "odrl:action": "extract_chart",
                        "odrl:assignee": f"model:{model}",
                        "mode": "chart_extraction",
                        "result": chart_raw
                    })
                    print(f"--- CHART EXTRACTION OUTPUT ---\n{chart_raw}\n------------------------\n")
                    log_file = get_odrl_log_target(policy_file)
                    append_to_log(log_file, "extract_chart", "chart_extraction", f"Chart Output:\n{chart_raw}")
                    result_text += f"\nChart Extraction Data:\n{chart_raw}"
                except Exception as e:
                    print(f"Chart extraction failed: {e}")
                    executed_actions.append({
                        "odrl:action": "extract_chart",
                        "odrl:assignee": f"model:{model}",
                        "mode": "chart_extraction",
                        "error": str(e)
                    })

        # Optional Verification Step
        policy_file = Path("policies/cv_extraction_policy.jsonld")
        verify_prompt = load_odrl_policy(policy_file, "light", "model:gemma4:e2b", action="verify", constraint_key="prompt")
        if verify_prompt:
            # Inject raw extractions into the verification prompt
            aggregation_prompt = f"{verify_prompt}\n\n--- PREVIOUS EXTRACTION RESULTS ---\n"
            for action in executed_actions:
                if action.get("odrl:action") == "extract":
                    aggregation_prompt += f"{action.get('mode').title()} Pass Data:\n{action.get('result')}\n\n"
            aggregation_prompt += "-----------------------------------"
            
            print(f"\nRunning verification pass with gemma4:e2b...")
            try:
                import ollama
                # Pass both original and enhanced images (default to light if heavy wasn't run)
                enhanced_verify_path = preprocess_image(image_path, session_dir, mode='light') if not needs_pass_2 else preprocess_image(image_path, session_dir, mode='heavy')
                
                verify_resp = ollama.generate(
                    model="gemma4:e2b",
                    prompt=aggregation_prompt,
                    images=[str(image_path), str(enhanced_verify_path)],
                    options={"num_ctx": 4096}
                )
                
                # Cleanup verification enhanced image if it was recreated just for this step
                if enhanced_verify_path != Path(image_path) and enhanced_verify_path.exists():
                    enhanced_verify_path.unlink()
                verify_raw = verify_resp.get('response', '').strip()
                executed_actions.append({
                    "odrl:action": "verify",
                    "odrl:assignee": "model:gemma4:e2b",
                    "mode": "light",
                    "result": verify_raw
                })
                print(f"--- VERIFICATION OUTPUT ---\n{verify_raw}\n------------------------\n")
                log_file = get_odrl_log_target(policy_file)
                append_to_log(log_file, "verify", "light", f"Verification Model Output:\n{verify_raw}")
            except Exception as e:
                print(f"Verification failed: {e}")
                executed_actions.append({
                    "odrl:action": "verify",
                    "odrl:assignee": "model:gemma4:e2b",
                    "mode": "light",
                    "result": str(e)
                })
            
        # Save AI trace as JSON-LD
        ai_dir = Path(session_dir)
        ai_dir.mkdir(exist_ok=True)
        trace_file = ai_dir / f"{Path(image_path).stem}_trace.jsonld"
        
        full_trace = {
            "@context": "http://www.w3.org/ns/odrl.jsonld",
            "@type": "odrl:Log",
            "source_file": Path(image_path).name,
            "timestamp": datetime.now().isoformat(),
            "executed_actions": executed_actions
        }
        
        try:
            with open(trace_file, 'w', encoding='utf-8') as f:
                json.dump(full_trace, f, indent=4, ensure_ascii=False)
                
            txt_file = ai_dir / f"{Path(image_path).stem}_trace.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"Source File: {Path(image_path).name}\n")
                f.write(f"Timestamp: {full_trace['timestamp']}\n\n")
                for action in executed_actions:
                    f.write(f"{'='*80}\n")
                    f.write(f"ACTION: {action.get('odrl:action')}\n")
                    f.write(f"ASSIGNEE: {action.get('odrl:assignee')}\n")
                    f.write(f"MODE: {action.get('mode')}\n")
                    f.write(f"{'-'*80}\n")
                    f.write(f"{action.get('result', '')}\n\n")
                    
        except Exception as e:
            print(f"Failed to save AI trace: {e}")

        return {
            "content": {
                "source_file": Path(image_path).name,
                "analysis": result_text
            },
            "type": "json",
            "description": "image analysis"
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def parse_computer_vision_file(file_path):
    """
    Parses a single Computer Vision chat export text file.
    Supports multiple common Computer Vision export formats.
    """
    pattern1 = re.compile(r'^\[?(\d{1,2}[\/\.]\d{1,2}[\/\.]\d{2,4},? \d{1,2}:\d{2}(?::\d{2})?(?: [APM]{2})?)\]? [-:]? ([^:]+): (.*)$')
    pattern2 = re.compile(r'^(\d{1,2}[\/\.]\d{1,2}[\/\.]\d{2,4},? \d{1,2}:\d{2}(?::\d{2})?(?: [APM]{2})?) - ([^:]+): (.*)$')

    episodes = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Match Format 1
                match1 = pattern1.match(line)
                if match1:
                    timestamp, sender, message = match1.groups()
                    episodes.append({
                        "content": {
                            "timestamp": timestamp.strip(),
                            "sender": sender.strip(),
                            "message": message.strip(),
                            "source_file": Path(file_path).name
                        },
                        "type": "json",
                        "description": "computer_vision message"
                    })
                    continue
                    
                # Match Format 2
                match2 = pattern2.match(line)
                if match2:
                    timestamp, sender, message = match2.groups()
                    episodes.append({
                        "content": {
                            "timestamp": timestamp.strip(),
                            "sender": sender.strip(),
                            "message": message.strip(),
                            "source_file": Path(file_path).name
                        },
                        "type": "json",
                        "description": "computer_vision message"
                    })
                    continue
                    
                # Multi-line handling
                if episodes:
                    episodes[-1]["content"]["message"] += "\n" + line
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        
    return episodes

def main():
    parser = argparse.ArgumentParser(description="Extract Computer Vision chat logs and analyze images for Palefire ingestion.")
    parser.add_argument("--dir", type=str, default="./data/computer_vision", help="Directory containing Computer Vision .txt files")
    parser.add_argument("--image", type=str, help="Path to a single image file to analyze")
    parser.add_argument("--image-dir", type=str, help="Directory containing images to batch process")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the LLM extraction cache for the specified image before processing")
    parser.add_argument("--pdf", type=str, help="Path to a local PDF file to extract and process")
    parser.add_argument("--pdf-url", type=str, help="URL of a PDF file to download, extract, and process")
    parser.add_argument("--vision-model", type=str, default="gemma4:latest", help="Ollama vision model to use for image analysis (default: gemma4:latest)")
    parser.add_argument("--output", type=str, default="computer_vision_episodes.json", help="Output JSON file name")
    
    args = parser.parse_args()
    
    # Handle PDF downloading and extraction if requested
    if args.pdf_url or args.pdf:
        try:
            import fitz
        except ImportError:
            print("Error: PyMuPDF is required to process PDFs. Install it with: pip install pymupdf")
            sys.exit(1)
            
        import urllib.request
        from PIL import Image
        
        pdf_path = args.pdf
        if args.pdf_url:
            pdf_path = "/tmp/downloaded_temp.pdf"
            print(f"Downloading PDF from {args.pdf_url}...")
            urllib.request.urlretrieve(args.pdf_url, pdf_path)
            
        out_dir = Path("datamaps/data/pdf-extraction")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Extracting all pages from {pdf_path} into {out_dir}...")
        for f in out_dir.glob("*.jpg"):
            f.unlink()
            
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            out_path = out_dir / f"page_{page_num + 1:02d}.jpg"
            img.save(out_path, "JPEG", quality=90)
            
        print(f"Total pages extracted: {len(doc)}")
        
        # Override image_dir to process the newly extracted pages
        args.image_dir = str(out_dir)
    
    all_episodes = []
    
    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session_dir = Path("data") / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    PALEFIRE = Namespace("http://palefire.org/schema#")
    kg_path = Path("knowledge_graph.ttl")
    kg = Graph()
    if kg_path.exists():
        kg.parse(kg_path, format="turtle")
    else:
        kg.bind("palefire", PALEFIRE)
        
    def process_with_cache(img_path_obj, sess_dir):
        img_uri = URIRef(img_path_obj.absolute().as_uri())
        cached_ep_path = img_path_obj.parent / f"{img_path_obj.stem}_cached.json"
        
        # Clear cache for this image if requested
        if hasattr(args, 'clear_cache') and args.clear_cache:
            if (img_uri, PALEFIRE.status, Literal("processed")) in kg:
                kg.remove((img_uri, PALEFIRE.status, Literal("processed")))
                kg.serialize(kg_path, format="turtle")
                print(f"Cleared Knowledge Graph cache for {img_path_obj.name}")
            if cached_ep_path.exists():
                cached_ep_path.unlink()
        
        if (img_uri, PALEFIRE.status, Literal("processed")) in kg and cached_ep_path.exists():
            print(f"Skipping {img_path_obj.name} (found in Knowledge Graph cache)")
            try:
                with open(cached_ep_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache for {img_path_obj.name}: {e}")
                
        episode = analyze_image(img_path_obj, sess_dir, model=args.vision_model)
        
        if episode:
            with open(cached_ep_path, 'w', encoding='utf-8') as f:
                json.dump(episode, f, indent=4, ensure_ascii=False)
            kg.add((img_uri, PALEFIRE.status, Literal("processed")))
            kg.add((img_uri, PALEFIRE.hasCachedEpisode, URIRef(cached_ep_path.absolute().as_uri())))
            kg.serialize(destination=kg_path, format="turtle")
            
        return episode
    
    # Process single image if provided
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Image not found: {args.image}")
            return
            
        episode = process_with_cache(img_path, session_dir)
        if episode:
            if isinstance(episode, list):
                all_episodes.extend(episode)
                print(f"\nImage Analysis Result: {len(episode)} multi-map splits extracted.\n")
            else:
                all_episodes.append(episode)
                print(f"\nImage Analysis Result:\n{episode['content']['analysis']}\n")
    elif args.image_dir:
        # Process an entire directory of images
        target_dir = Path(args.image_dir)
        if not target_dir.exists() or not target_dir.is_dir():
            print(f"Image directory {target_dir} does not exist.")
            return
            
        img_files = list(target_dir.glob('*.jpg')) + list(target_dir.glob('*.jpeg')) + list(target_dir.glob('*.png'))
        if not img_files:
            print(f"No images found in {target_dir}.")
            return
            
        for img_path in img_files:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Batch processing {img_path.name}...")
            episode = process_with_cache(img_path, session_dir)
            if episode:
                if isinstance(episode, list):
                    all_episodes.extend(episode)
                else:
                    all_episodes.append(episode)
    else:
        # Process Computer Vision chat directories
        target_dir = Path(args.dir)
        
        if not target_dir.exists() or not target_dir.is_dir():
            print(f"Directory {target_dir} does not exist. Creating it...")
            target_dir.mkdir(parents=True, exist_ok=True)
            print("Please place Computer Vision chat files (.txt) into the directory and run this script again.")
            return
            
        txt_files = list(target_dir.glob('*.txt'))
        
        if not txt_files:
            print(f"No .txt files found in {target_dir}. Please add Computer Vision export files.")
            return

        for file_path in txt_files:
            print(f"Processing {file_path.name}...")
            episodes = parse_computer_vision_file(file_path)
            all_episodes.extend(episodes)
            print(f"Extracted {len(episodes)} messages from {file_path.name}")
            
    # Save the aggregated episodes
    if all_episodes:
        out_file = Path(args.output)
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(all_episodes, f, indent=4, ensure_ascii=False)
        print(f"\nSuccessfully extracted {len(all_episodes)} total episodes.")
        print(f"Saved episodes to {out_file}")
        print("You can now ingest them using the palefire framework command:")
        print(f"  python palefire-cli.py ingest {out_file} --ner")
    else:
        print("No valid data found to extract.")

if __name__ == '__main__':
    main()
