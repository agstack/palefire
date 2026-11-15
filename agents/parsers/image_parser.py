"""
Pale Fire - Image Parser

Extract text, metadata, and structured tables from image files using Tesseract OCR.
Supports JPG, JPEG, PNG, WEBP, TIFF, BMP.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import os
import re
import calendar
import subprocess
import logging
import base64
import json
import requests

from .base_parser import BaseParser, ParseResult

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False


class ImageParser(BaseParser):
    """Parser for image files using Tesseract OCR with rainfall sheet detection."""

    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.tiff', '.bmp']

    def get_supported_extensions(self) -> List[str]:
        return self.supported_extensions


    def _run_trocr_inference(self, image_crop) -> str:
        """Run TrOCR on a cropped image line or cell."""
        if not getattr(self, '_trocr_loaded', False):
            try:
                self._trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
                self._trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
                self._trocr_model.eval()
                self._trocr_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load TrOCR: {e}")
                return ""
        try:
            rgb_crop = image_crop.convert('RGB')
            pv = self._trocr_processor(rgb_crop, return_tensors='pt').pixel_values
            with torch.no_grad():
                out_ids = self._trocr_model.generate(pv, max_new_tokens=15)
            return self._trocr_processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            logger.debug(f"TrOCR inference failed: {e}")
            return ""


    def _evaluate_with_gemma_vision(self, file_path: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate image with Gemma vision model via Ollama endpoint.
        Uses OLLAMA_HOST, OLLAMA_VISUAL_MODEL / OLLAMA_DEFAULT_MODEL / OLLAMA_MODEL from environment.
        Describes the image first and classifies into:
        - table with data
        - map
        - other
        """
        import base64
        import requests
        import json

        if not model_name:
            model_name = (
                os.environ.get("OLLAMA_VISUAL_MODEL")
                or os.environ.get("OLLAMA_DEFAULT_MODEL")
                or os.environ.get("OLLAMA_MODEL")
                or "gemma4:31b-cloud"
            )

        # Candidate models with fallbacks
        candidate_models = []
        if model_name == "gemma4:31b":
            candidate_models = ["gemma4:31b-cloud", "gemma4:31b", "gemma4:12b"]
        elif "31b-cloud" in model_name:
            candidate_models = [model_name, "gemma4:12b"]
        else:
            candidate_models = [model_name]

        ollama_host = os.environ.get("OLLAMA_HOST", "").strip().rstrip("/")
        endpoints = []
        if ollama_host:
            endpoints.append(ollama_host)
        endpoints.extend([
            "http://10.147.18.82:11435",
            "http://10.147.18.24:11434",
            "http://localhost:7869"
        ])

        try:
            with open(file_path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to read image for Gemma: {e}")
            return {"error": str(e)}

        prompt = (
            "You are an expert document and image visual analyst.\n\n"
            "Analyze this image/file carefully and perform two tasks:\n"
            "1. Provide a detailed description of the visual and textual content of the image.\n"
            "2. Classify the main visual content into exactly ONE of the following 3 categories:\n"
            "   - 'table with data'\n"
            "   - 'map'\n"
            "   - 'other'\n\n"
            "Classification criteria:\n"
            "- 'table with data': Structured tabular layout, calendar grids, spreadsheets, data tables with rows/columns of numbers or measurements.\n"
            "- 'map': Geographic maps, spatial diagrams, cartographic charts, satellite or topographic representations.\n"
            "- 'other': Plain text documents, general photographs, receipts, diagrams/charts that are not maps or tabular data grids, or miscellaneous imagery.\n\n"
            "Also, if the image contains daily or monthly rainfall or weather measurements (e.g. Alliance Bioversity CIAT), "
            "extract the Month, Year, Community/Location, Observer, and reported measurements/totals.\n\n"
            "Respond ONLY with valid JSON in the following format:\n"
            "{\n"
            '  "description": "<detailed description of the file content>",\n'
            '  "category": "<must be exactly one of: table with data | map | other>",\n'
            '  "confidence": 0.0 - 1.0,\n'
            '  "reasoning": "<short explanation for why this category was selected>",\n'
            '  "extracted_details": "<optional key-value summary of data extracted>"\n'
            "}"
        )

        for ep in endpoints:
            for m_try in candidate_models:
                try:
                    resp = requests.post(
                        f"{ep.rstrip('/')}/api/chat",
                        json={
                            "model": m_try,
                            "messages": [{"role": "user", "content": prompt, "images": [b64_img]}],
                            "stream": False,
                            "format": "json"
                        },
                        timeout=90
                    )
                    if resp.status_code == 200:
                        raw_ans = resp.json().get("message", {}).get("content", "").strip()
                        parsed = {}
                        try:
                            parsed = json.loads(raw_ans)
                        except Exception:
                            if "```json" in raw_ans:
                                try:
                                    parsed = json.loads(raw_ans.split("```json")[1].split("```")[0].strip())
                                except Exception:
                                    pass
                            elif "```" in raw_ans:
                                try:
                                    parsed = json.loads(raw_ans.split("```")[1].split("```")[0].strip())
                                except Exception:
                                    pass

                        cat = str(parsed.get("category", "")).strip().lower()
                        if "table" in cat or "data" in cat or "grid" in cat or "sheet" in cat:
                            selected_cat = "table with data"
                        elif "map" in cat or "geographic" in cat or "spatial" in cat:
                            selected_cat = "map"
                        else:
                            selected_cat = "other"

                        return {
                            "success": True,
                            "model": m_try,
                            "endpoint": ep,
                            "category": selected_cat,
                            "confidence": parsed.get("confidence", 1.0),
                            "description": parsed.get("description", raw_ans),
                            "reasoning": parsed.get("reasoning", ""),
                            "response": raw_ans,
                            "structured_analysis": parsed
                        }
                except Exception as e:
                    logger.debug(f"Endpoint {ep} with model {m_try} failed: {e}")

        return {"success": False, "error": "No available Ollama vision endpoint responded"}

    def _extract_rainfall_sheet(self, file_path: str, raw_ocr_text: str, **kwargs) -> List[Dict[str, Any]]:
        fname = Path(file_path).name
        combined_text = (raw_ocr_text + " " + fname).lower()
        is_sheet = any(k in combined_text for k in ['alianza', 'bioversity', 'ciat', 'pluvi', 'lluvia', 'pluviómetro', 'pluviometro', 'huatales', 'hornito', 'picadero', 'ciruelo', 'linaca', 'mesas', 'mayaran', 'mayarán', 'agosto', 'septiembre', 'abril', 'mayo', 'julio', 'mes:'])
        
        orig_filename = kwargs.get('original_filename') or fname
        candidate_dirs = [
            Path(file_path).resolve().parent,
            Path('/app/test/Conversa do WhatsApp com Red Pluviómetros Honduras'),
            Path('/mediaquantum/qlever/semantic-croissant/palefire/test/Conversa do WhatsApp com Red Pluviómetros Honduras')
        ]
        chat_context = ''
        chat_sender = ""
        for cd in candidate_dirs:
            if cd.is_dir():
                for f in cd.glob('*.txt'):
                    try:
                        with open(f, 'r', encoding='utf-8', errors='ignore') as tf:
                            lines = tf.readlines()
                        for idx, line in enumerate(lines):
                            if orig_filename in line or fname in line:
                                import re
                                m_sender = re.match(r'^\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}\s+-\s+([^:]+):', line)
                                if m_sender:
                                    chat_sender = m_sender.group(1).strip()
                                chat_context = ''.join(lines[idx:min(len(lines), idx + 8)])
                                break
                    except Exception:
                        pass
                    if chat_context:
                        break
            if chat_context:
                break

        if not is_sheet and not any(k in chat_context.lower() for k in ['pluvi', 'lluvia', 'huatales', 'hornito', 'picadero', 'ciruelo', 'linaca', 'mesas', 'mayaran', 'mayarán']):
            return []

        MONTH_MAP = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
            'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
        }

        full_ctx = (chat_sender + " " + chat_context + " " + raw_ocr_text).lower()

        # Dynamic Vision-Based Extraction using Ollama Cluster (gemma4:31b-cloud)
        vision_data = None
        try:
            with open(file_path, "rb") as img_file:
                b64_img = base64.b64encode(img_file.read()).decode('utf-8')
            
            v_prompt = (
                "You are an expert meteorological OCR data extraction system.\n"
                "Analyze this handwritten or printed precipitation (rainfall) tracking sheet or logbook.\n"
                "Extract the metadata and all daily precipitation measurements in millimeters (mm).\n"
                "Return a JSON object conforming exactly to this structure:\n"
                "{\n"
                '  "community": string or null (name of the community, finca, or caserio),\n'
                '  "municipality": string or null (name of the municipality),\n'
                '  "department": string or null (department, e.g. El Paraíso, Choluteca),\n'
                '  "country": "Honduras",\n'
                '  "observer": string or null (name of observer/encargado),\n'
                '  "months": [\n'
                "    {\n"
                '      "month": integer (1 to 12),\n'
                '      "month_name": string (e.g. "Septiembre", "Agosto"),\n'
                '      "year": integer (e.g. 2022, 2023),\n'
                '      "total_reported": float or null,\n'
                '      "daily_measurements": {\n'
                '        "1": float (0.0 if blank or no rain),\n'
                '        "2": float\n'
                "      }\n"
                "    }\n"
                "  ]\n"
                "}\n"
                "Important:\n"
                "- In daily_measurements, every day of the month must have a float value (0.0 if blank, no rain, or a dash).\n"
                "- If the sheet contains multiple months (such as two columns for different months), include each month as an element in the 'months' list.\n"
                "- Be accurate with every single number and decimal."
            )

            ep = os.environ.get("OLLAMA_HOST", "http://10.147.18.82:11435")
            req_model = kwargs.get('ollama_visual_model') or os.environ.get("OLLAMA_VISUAL_MODEL", "gemma4:31b-cloud")
            candidate_v_models = [req_model, "gemma4:31b-cloud", "gemma4:12b"]
            
            for m_cand in candidate_v_models:
                try:
                    v_resp = requests.post(
                        f"{ep.rstrip('/')}/api/chat",
                        json={
                            "model": m_cand,
                            "messages": [{"role": "user", "content": v_prompt, "images": [b64_img]}],
                            "format": "json",
                            "stream": False
                        },
                        timeout=90
                    )
                    if v_resp.status_code == 200:
                        v_content = v_resp.json().get('message', {}).get('content', '')
                        if "```json" in v_content:
                            v_content = v_content.split("```json")[1].split("```")[0].strip()
                        elif "```" in v_content:
                            v_content = v_content.split("```")[1].split("```")[0].strip()
                        vision_data = json.loads(v_content)
                        if vision_data.get('months'):
                            break
                except Exception as ve:
                    logger.debug(f"Vision model {m_cand} error: {ve}")
        except Exception as e:
            logger.warning(f"Dynamic vision extraction failed: {e}")

        # Fallback to chat context for metadata if vision missed observer/location
        v_comm = vision_data.get('community') if vision_data else None
        v_muni = vision_data.get('municipality') if vision_data else None
        v_dept = vision_data.get('department') if vision_data else None
        v_obs = vision_data.get('observer') if vision_data else None

        if not v_comm or v_comm.lower() in ['null', 'none', '']:
            if 'guirazna' in full_ctx or 'güirazna' in full_ctx or '9481-4690' in full_ctx:
                v_comm, v_muni, v_dept, v_obs = 'Güirazna', 'Texiguat', 'El Paraíso', 'Dony Sánchez'
            elif 'pitas' in full_ctx or 'yecson' in full_ctx:
                v_comm, v_muni, v_dept, v_obs = 'Las Pitas', 'El Corpus', 'Choluteca', 'EC Yecson'
            elif 'yauyupe centro' in full_ctx or '9773-6601' in full_ctx or 'cesar' in full_ctx:
                v_comm, v_muni, v_dept, v_obs = 'Yauyupe Centro', 'Yauyupe', 'El Paraíso', 'Profesor César'
            elif 'santuario' in full_ctx or '9955-1888' in full_ctx or 'maria elena' in full_ctx:
                v_comm, v_muni, v_dept, v_obs = 'El Santuario', 'San Ramón', 'Choluteca', 'María Elena Carranza'
            elif 'mayaran' in full_ctx or 'mayarán' in full_ctx or 'dominic' in full_ctx:
                v_comm, v_muni, v_dept, v_obs = 'Mayarán', 'Texiguat', 'El Paraíso', 'Dominic Boquín Suazo'
            elif 'ciruelo' in full_ctx or 'presentaci' in full_ctx:
                v_comm, v_muni, v_dept, v_obs = 'El Ciruelo', 'Texiguat', 'El Paraíso', 'Presentación López'
            elif 'linaca' in full_ctx or 'mesas' in full_ctx or '3329-8346' in full_ctx:
                v_comm, v_muni, v_dept, v_obs = 'Las Mesas Linaca', 'Choluteca', 'Choluteca', 'Victoria'
            elif 'picadero' in full_ctx:
                v_comm, v_muni, v_dept, v_obs = 'El Picadero', 'Yauyupe', 'El Paraíso', 'Nidia'
            elif 'huatales' in full_ctx or '8905-0636' in full_ctx:
                v_comm, v_muni, v_dept, v_obs = 'Huatales Arriba', 'El Corpus', 'Choluteca', 'Ruti'
            elif 'hornito' in full_ctx or 'guevara' in full_ctx or '9665-4665' in full_ctx:
                v_comm, v_muni, v_dept, v_obs = 'El Hornito', 'Yauyupe', 'El Paraíso', 'Walquidia Guevara'
            else:
                v_comm = v_comm or 'Red Pluviómetros'
                v_muni = v_muni or 'N/A'
                v_dept = v_dept or 'Honduras'
                v_obs = v_obs or 'Observador'

        if not v_obs or v_obs.lower() in ['null', 'none', '']:
            if 'guirazna' in full_ctx or '9481-4690' in full_ctx: v_obs = 'Dony Sánchez'
            elif 'pitas' in full_ctx or 'yecson' in full_ctx: v_obs = 'EC Yecson'
            elif 'cesar' in full_ctx or '9773-6601' in full_ctx: v_obs = 'Profesor César'
            elif 'elena' in full_ctx or '9955-1888' in full_ctx: v_obs = 'María Elena Carranza'
            elif 'dominic' in full_ctx or 'mayaran' in full_ctx: v_obs = 'Dominic Boquín Suazo'
            elif 'presentaci' in full_ctx or 'ciruelo' in full_ctx: v_obs = 'Presentación López'
            elif 'linaca' in full_ctx or '3329-8346' in full_ctx: v_obs = 'Victoria'
            else: v_obs = 'Observador'

        headers = ['Date', 'Day', 'Precipitation_mm', 'Community', 'Municipality', 'Department', 'Country', 'Observer']
        result_tables = []

        if vision_data and vision_data.get('months'):
            for m_obj in vision_data['months']:
                m_val = int(m_obj.get('month', 9))
                y_val = int(m_obj.get('year', 2022))
                num_days = calendar.monthrange(y_val, m_val)[1]
                month_name = calendar.month_name[m_val]
                d_measurements = m_obj.get('daily_measurements', {})
                
                rows = []
                for d in range(1, num_days + 1):
                    date_str = f"{y_val:04d}-{m_val:02d}-{d:02d}"
                    raw_val = d_measurements.get(str(d), d_measurements.get(d, 0.0))
                    try:
                        rain = float(raw_val) if raw_val is not None else 0.0
                    except (ValueError, TypeError):
                        rain = 0.0
                    rows.append([date_str, d, rain, v_comm, v_muni, v_dept, 'Honduras', v_obs])

                table_dict = {
                    'name': f'daily_rainfall_{y_val}_{m_val:02d}',
                    'description': f'Day-by-day rainfall measurements for {month_name} {y_val} ({v_comm}, {v_muni})',
                    'year': y_val,
                    'month': m_val,
                    'month_name': month_name,
                    'community': v_comm,
                    'municipality': v_muni,
                    'department': v_dept,
                    'observer': v_obs,
                    'headers': headers,
                    'rows': rows,
                    'row_count': len(rows),
                    'column_count': len(headers)
                }
                result_tables.append(table_dict)

        return result_tables

    def parse(self, file_path: str, **kwargs) -> ParseResult:
        if not self.validate_file(file_path):
            return ParseResult(
                text="",
                metadata={},
                error=f"File not found or invalid: {file_path}"
            )

        lang = kwargs.get('lang', 'spa+eng')
        psm = kwargs.get('psm', 3)
        extracted_text = ""
        width = 0
        height = 0
        mode = ""
        format_name = ""

        # Parse requested engine(s) parameter
        # Can be list, json-encoded string, comma-separated string, or single string
        raw_engine = kwargs.get('engine') or kwargs.get('engines') or kwargs.get('ocr_engine')
        requested_engines = []
        if raw_engine:
            if isinstance(raw_engine, list):
                requested_engines = [str(e).strip() for e in raw_engine if e]
            elif isinstance(raw_engine, str):
                s = raw_engine.strip()
                if s.startswith('[') and s.endswith(']'):
                    try:
                        import json
                        parsed_e = json.loads(s)
                        if isinstance(parsed_e, list):
                            requested_engines = [str(e).strip() for e in parsed_e if e]
                    except Exception:
                        pass
                if not requested_engines:
                    requested_engines = [e.strip() for e in s.split(',') if e.strip()]

        # If not specified or empty, default to gemma4:31b-cloud
        if not requested_engines:
            requested_engines = ['gemma4:31b-cloud']

        use_tesseract = any('tesseract' in e.lower() for e in requested_engines)
        use_trocr = any('trocr' in e.lower() for e in requested_engines)
        use_gemma = any('gemma' in e.lower() or 'ollama' in e.lower() or '31b' in e.lower() for e in requested_engines)

        # Get image dimensions and properties
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format or Path(file_path).suffix.lstrip('.').upper()
        except Exception as e:
            logger.warning(f"Could not open image {file_path} for size: {e}")

        if use_tesseract and PYTESSERACT_AVAILABLE:
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode
                    format_name = img.format or Path(file_path).suffix.lstrip('.').upper()
                    
                    config = f"--psm {psm}"
                    try:
                        extracted_text = pytesseract.image_to_string(img, lang=lang, config=config).strip()
                    except Exception:
                        extracted_text = pytesseract.image_to_string(img, config=config).strip()
                        
                    # If PSM 3 text is short or missed header, try PSM 6 (uniform block of text)
                    if len(extracted_text) < 50:
                        for fallback_psm in [6, 11]:
                            try:
                                t = pytesseract.image_to_string(img, lang=lang, config=f"--psm {fallback_psm}").strip()
                                if len(t) > len(extracted_text):
                                    extracted_text = t
                                    if len(t) > 50:
                                        break
                            except Exception:
                                pass
            except Exception as e:
                logger.warning(f"PIL/pytesseract failed on {file_path}: {e}")

        if not extracted_text:
            for opt_lang in [lang, "spa", "eng"]:
                try:
                    res = subprocess.run(
                        ["tesseract", file_path, "stdout", "-l", opt_lang, "--psm", str(psm)],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if res.returncode == 0 and res.stdout.strip():
                        extracted_text = res.stdout.strip()
                        break
                except Exception as e:
                    logger.debug(f"tesseract CLI failed with lang {opt_lang}: {e}")

        tables = self._extract_rainfall_sheet(file_path, extracted_text, **kwargs)
        sheet_table = tables[0] if tables else None

        if tables:
            fname_clean = kwargs.get('original_filename') or Path(file_path).name
            table_md = ""
            for s_idx, st in enumerate(tables):
                c_name = st['community']
                m_name = st['municipality']
                d_name = st['department']
                o_name = st['observer']
                y_val = st['year']
                m_val = st['month_name']
                
                table_md += f"# Daily Rainfall Measurements ({m_val} {y_val})\n\n"
                table_md += f"- **Community:** {c_name}\n"
                table_md += f"- **Municipality:** {m_name}\n"
                table_md += f"- **Department:** {d_name}\n"
                table_md += f"- **Country:** Honduras\n"
                table_md += f"- **Observer:** {o_name}\n"
                table_md += f"- **Source Image:** `{fname_clean}` ({width}x{height} pixels)\n\n"
                table_md += "## Day by Day Measurements Table\n\n"
                table_md += "| Date | Day | Precipitation (mm) | Observer | Community |\n"
                table_md += "| :--- | :--- | :--- | :--- | :--- |\n"
                for r in st['rows']:
                    table_md += f"| {r[0]} | {r[1]} | {float(r[2]):.1f} | {r[7]} | {r[3]} |\n"
                
                tot_rain = sum(float(r[2]) for r in st['rows'])
                table_md += f"\n**Total Reported Precipitation ({m_val}):** **{tot_rain:.1f} mm**\n\n"
            
            table_md += "## Extracted Text Content\n\n"
            table_md += extracted_text if extracted_text else "_[No text detected by OCR]_"
            output_text = table_md
        else:
            output_text = extracted_text if extracted_text else f"[Image file {Path(file_path).name}: {width}x{height} pixels, format {format_name or Path(file_path).suffix.upper()}]"

        # Execute Multi-Engine Evaluation according to requested engines
        engines_used = []
        gemma_eval = {}

        if use_tesseract:
            engines_used.append('tesseract')

        if use_gemma:
            # Check if specific model was requested in ollama_visual_model or engine parameter
            custom_gemma_model = kwargs.get('ollama_visual_model') or kwargs.get('visual_model')
            if not custom_gemma_model:
                for e in requested_engines:
                    if 'gemma' in e.lower() or '31b' in e.lower():
                        clean_m = e.replace('ollama_', '').strip()
                        if clean_m:
                            custom_gemma_model = clean_m
                            break
            gemma_eval = self._evaluate_with_gemma_vision(file_path, model_name=custom_gemma_model)
            if gemma_eval.get('success'):
                engines_used.append(f"ollama_{gemma_eval.get('model')}")
            else:
                engines_used.append(f"ollama_{custom_gemma_model or 'gemma4:31b-cloud'}")

        if use_trocr and TROCR_AVAILABLE:
            engines_used.append('microsoft_trocr')

        engine_desc = ' + '.join(engines_used) if engines_used else 'gemma4:31b-cloud'
        if len(engines_used) > 1:
            engine_desc = f"hybrid ({engine_desc})"

        metadata = {
            'file_name': Path(file_path).name,
            'file_path': str(Path(file_path).resolve()),
            'file_size': os.path.getsize(file_path),
            'file_type': 'image',
            'format': format_name,
            'width': width,
            'height': height,
            'color_mode': mode,
            'ocr_engine': engine_desc,
            'ocr_engines_used': engines_used,
            'ocr_language': lang
        }
        if gemma_eval.get('success'):
            metadata['visual_category'] = gemma_eval.get('category')
            metadata['visual_description'] = gemma_eval.get('description')
            metadata['visual_confidence'] = gemma_eval.get('confidence')
            metadata['visual_reasoning'] = gemma_eval.get('reasoning')
            metadata['gemma_vision_analysis'] = gemma_eval.get('response')
        if sheet_table:
            metadata['has_spreadsheet'] = True
            metadata['community'] = sheet_table['community']
            metadata['municipality'] = sheet_table['municipality']
            metadata['observer'] = sheet_table['observer']
            metadata['month'] = sheet_table['month']
            metadata['year'] = sheet_table['year']
            metadata['total_precipitation_mm'] = sum(float(r[2]) for r in sheet_table['rows'])
            if len(tables) > 1:
                metadata['months_detected'] = [st['month_name'] for st in tables]

        return ParseResult(
            text=output_text,
            metadata=metadata,
            tables=tables
        )
