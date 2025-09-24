# app.py
# Flask webapp: subes una imagen (orden ML) -> devuelve tabla (SKU, Título, Unidades, Marketplace, Envío)

import io
import re
from typing import Dict, Any, List, Tuple

from flask import Flask, request, render_template_string, send_file
from PIL import Image, ImageFilter, ImageOps
import pytesseract
import pandas as pd

app = Flask(__name__)

# --- Catálogo básico (mapeo por palabras clave -> (SKU, Título canónico)) ---
CATALOGO = [
    # (palabras clave en minúsculas, SKU, título canónico)
    (["lechelak", "cabra", "340"], "7501468144501", "LecheLak - Leche de Cabra en Polvo 340gr"),
    (["crecelac", "0-12", "800"], "7501468140442", "Crecelac 0-12 M 800 GR"),
    (["crecelac", "0-12", "400"], "7501468145508", "Crecelac 0-12 M 400 GR"),
    (["crecelac", "firstep", "1-3", "360"], "7501468148103", "Crecelac FIRSTEP 1-3 AÑOS 360 GR"),
    (["crecelac", "firstep", "1-3", "800"], "7501468148301", "Crecelac FIRSTEP 1-3 AÑOS 800 GR"),
    (["crecelac", "0-12", "1.5"], "7501468141043", "Crecelac 0-12 M 1.5 KG"),
    (["crecelac", "firstep", "1.5"], "7501468140947", "Crecelac FIRSTEP 1-3 AÑOS 1.5 KG"),
]

MARKETPLACE_CONST = "Mercadolibre"

HTML = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>OCR Orden → Tabla</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 32px;}
    .card{max-width:980px;margin:auto;padding:24px;border:1px solid #e5e7eb;border-radius:16px;box-shadow:0 8px 24px rgba(0,0,0,.06)}
    h1{margin:0 0 16px;font-size:22px}
    .hint{color:#6b7280;margin-bottom:16px}
    .drop{border:2px dashed #cbd5e1;padding:24px;border-radius:12px;text-align:center}
    table{border-collapse:collapse;width:100%;margin-top:16px}
    th,td{border:1px solid #e5e7eb;padding:10px 12px;text-align:left}
    th{background:#f8fafc}
    .btn{display:inline-block;background:#2563eb;color:#fff;padding:10px 16px;border-radius:10px;text-decoration:none;border:none;cursor:pointer}
    .row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
    .pill{background:#eef2ff;color:#3730a3;padding:4px 10px;border-radius:999px;font-size:12px}
    img{max-width:100%;height:auto;border-radius:12px;margin-top:12px}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:8px}
    @media (max-width: 800px){.grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <div class="card">
    <h1>Extraer datos de Orden (Imagen → Tabla)</h1>
    <p class="hint">Sube una captura de la orden (por ejemplo, Mercado Libre). El sistema hará OCR y te devolverá la tabla.</p>
    <form class="drop" action="/" method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" required>
      <br><br>
      <button class="btn" type="submit">Procesar</button>
    </form>

    {% if preview %}
    <div class="grid">
      <div>
        <h3>Vista previa</h3>
        <img src="data:image/png;base64,{{preview}}">
      </div>
      <div>
        <h3>Tabla detectada</h3>
        {{ table_html|safe }}
        <div style="margin-top:12px" class="row">
          <form method="post" action="/download">
            <input type="hidden" name="csv" value="{{csv_b64}}">
            <button class="btn" type="submit">Descargar CSV</button>
          </form>
          <span class="pill">Filas: {{rows}}</span>
        </div>
        <details style="margin-top:14px">
          <summary>Texto OCR (depuración)</summary>
          <pre style="white-space:pre-wrap">{{ ocr_text }}</pre>
        </details>
      </div>
    </div>
    {% endif %}
  </div>
</body>
</html>
"""

def preprocess(img: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.MedianFilter(size=3))
    w, h = g.size
    if max(w, h) < 1600:
        scale = 1600 / max(w, h)
        g = g.resize((int(w * scale), int(h * scale)))
    return g

def ocr_text_from_image(pil_img: Image.Image) -> Tuple[str, List[Dict[str, Any]]]:
    data = pytesseract.image_to_data(pil_img, lang="spa", output_type=pytesseract.Output.DICT)
    words = [w for w, conf in zip(data["text"], data["conf"]) if w and str(conf).isdigit() and int(conf) >= 40]
    text = " ".join(words)
    return text, data

EAN13_RE = re.compile(r"\b(7\d{12})\b")
UNITS_RE = re.compile(r"(\d+)\s*(?:unid(?:ad|ades)?)\b", re.IGNORECASE)
ENVIO_LINE_RE = re.compile(r"(env[ií]o[^\n]{0,60})", re.IGNORECASE)

def score_match(title_lower: str, keywords: List[str]) -> int:
    return sum(1 for k in keywords if k in title_lower)

def guess_from_catalog(title: str) -> Tuple[str, str]:
    tl = title.lower()
    best = ("", "")
    best_score = 0
    for keys, sku, canonical in CATALOGO:
        sc = score_match(tl, keys)
        if sc > best_score:
            best_score = sc
            best = (sku, canonical)
    return best

def find_title(text: str) -> str:
    cand = re.findall(r"([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚñáéíóú0-9\s\-\(\)\/]{15,})", text)
    brand_words = ("crecelac", "lechelak", "fórmula", "formula", "cabra")
    best = ""
    for c in cand:
        lower = c.lower()
        if any(b in lower for b in brand_words) and len(c) > len(best):
            best = c.strip()
    if not best and cand:
        best = max(cand, key=len).strip()
    return best or "Título no detectado"

def extract_fields(full_text: str) -> Dict[str, Any]:
    m_ean = EAN13_RE.search(full_text)
    sku = m_ean.group(1) if m_ean else ""
    title = find_title(full_text)
    m_u = UNITS_RE.search(full_text)
    unidades = int(m_u.group(1)) if m_u else 1
    m_env = ENVIO_LINE_RE.search(full_text.replace("  ", " "))
    envio = m_env.group(1).strip() if m_env else "Mercado Envíos"
    if not sku and title:
        sku_guess, canonical = guess_from_catalog(title)
        if sku_guess:
            sku = sku_guess
            if score_match(title.lower(), [w for w in canonical.lower().split()]) < 3:
                title = canonical
    return {
        "SKU": sku or "No detectado",
        "Título de la publicación": title,
        "Unidades": unidades,
        "Marketplace": MARKETPLACE_CONST,
        "Envío": envio,
    }

def img_to_base64(pil_img: Image.Image) -> str:
    import base64
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def df_to_b64_csv(df: pd.DataFrame) -> str:
    import base64
    csv = df.to_csv(index=False, encoding="utf-8-sig")
    return base64.b64encode(csv.encode("utf-8-sig")).decode("utf-8")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(HTML, preview=None)
    file = request.files.get("image")
    if not file:
        return render_template_string(HTML, preview=None)
    img = Image.open(file.stream).convert("RGB")
    pre = preprocess(img)
    text, _ = ocr_text_from_image(pre)
    row = extract_fields(text)
    df = pd.DataFrame([row])
    table_html = df.to_html(index=False)
    preview_b64 = img_to_base64(img)
    csv_b64 = df_to_b64_csv(df)
    return render_template_string(
        HTML,
        preview=preview_b64,
        table_html=table_html,
        ocr_text=text,
        rows=len(df),
        csv_b64=csv_b64,
    )

@app.route("/download", methods=["POST"])
def download_csv():
    import base64
    csv_b64 = request.form.get("csv", "")
    if not csv_b64:
        return "No hay datos", 400
    raw = base64.b64decode(csv_b64)
    return send_file(
        io.BytesIO(raw),
        mimetype="text/csv; charset=utf-8",
        as_attachment=True,
        download_name="orden_ocr.csv",
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
