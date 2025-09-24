# app.py
# Flask webapp: subes una imagen (orden ML) -> devuelve tabla (SKU, Título, Unidades, Marketplace, Envío)
import io, re, unicodedata
from typing import Dict, Any, List, Tuple

from flask import Flask, request, render_template_string, send_file
from PIL import Image, ImageFilter, ImageOps
import pytesseract
import pandas as pd

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024  # 6 MB

# ====== CATÁLOGO (SKU -> Descripción Producto) ======
CATALOGO_SKU: Dict[str, str] = {
    "7501468140442": "CRECELAC 0-12 M 800 GR",
    "7501468145508": "CRECELAC 0-12 M 400 GR",
    "7501468148103": "CRECELAC FIRSTEP 1-3 AÑOS 360 GR",
    "7501468148301": "CRECELAC FIRSTEP 1-3 AÑOS 800 GR",
    "7501468141043": "CRECELAC 0-12 M 1.5 KG",
    "7501468140947": "CRECELAC FIRSTEP 1-3 AÑOS 1.5 KG",
    "7501468144501": "LECHELAK LECHE DE CABRA 340 G",
}

# Precomputo tokens de descripciones para poder hacer match si no hay EAN
def _normalize(txt: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", txt.lower())
        if c.isalnum() or c.isspace()
    )

DESC_TOKENS = {
    sku: set(_normalize(desc).split())
    for sku, desc in CATALOGO_SKU.items()
}

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
    if max(w, h) > 1800:
        scale = 1800 / max(w, h)
        g = g.resize((int(w * scale), int(h * scale)))
    return g

def ocr_text_from_image(pil_img: Image.Image) -> str:
    # OCR más robusto en español
    return pytesseract.image_to_string(
        pil_img, lang="spa", config="--oem 1 --psm 6"
    )

EAN13_RE = re.compile(r"\b(7\d{12})\b")
UNITS_RE = re.compile(r"(\d+)\s*(?:unid(?:ad|ades)?)\b", re.IGNORECASE)
ENVIO_LINE_RE = re.compile(r"(env[ií]o[^\n]{0,60})", re.IGNORECASE)

def best_catalog_match(ocr_text: str) -> Tuple[str, str]:
    """
    Cuando NO hay EAN: calcula intersección de tokens del OCR contra
    los tokens de cada descripción de catálogo y devuelve el mejor.
    """
    tokens_text = set(_normalize(ocr_text).split())
    best_sku, best_desc, best_score = "", "", 0
    for sku, desc_tokens in DESC_TOKENS.items():
        score = len(tokens_text & desc_tokens)
        if score > best_score:
            best_score = score
            best_sku, best_desc = sku, CATALOGO_SKU[sku]
    return best_sku, best_desc

def extract_fields(full_text: str) -> Dict[str, Any]:
    # Unidades
    m_u = UNITS_RE.search(full_text)
    unidades = int(m_u.group(1)) if m_u else 1

    # Envío
    m_env = ENVIO_LINE_RE.search(full_text.replace("  ", " "))
    envio = m_env.group(1).strip() if m_env else "Mercado Envíos"

    # SKU por EAN
    m_ean = EAN13_RE.search(full_text)
    if m_ean:
        sku = m_ean.group(1)
        titulo = CATALOGO_SKU.get(sku, "Título no encontrado en catálogo")
    else:
        # Sin EAN → buscar mejor coincidencia por descripción
        sku, titulo = best_catalog_match(full_text)
        if not sku:
            sku, titulo = "No detectado", "Título no detectado"

    return {
        "SKU": sku,
        "Título de la publicación": titulo,  # SIEMPRE tomado del catálogo
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
    text = ocr_text_from_image(pre)

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
