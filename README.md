# Extractor universal

Aplicación **Streamlit** para extraer texto de sitios (Playwright, Trafilatura, apoyo PDF con pypdf). Los JSON y PDFs se guardan bajo `./almacen` y `./almacen/pdfs/`.

## Puesta en marcha

En la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium
streamlit run app.py
```

Streamlit queda en `http://localhost:8501` (el puerto exacto se muestra en consola).

## Buscador Jurídico (SCJN) — flujo y cambios recientes

### `scjn_tesis/buscador_juridico.py`

- **Detalle (`_fetch_bj_detail`)**  
  - Siempre se carga la página, se toma el texto de `body` (con el recorte de pie habitual) y **no** se abandona el texto aunque exista PDF.  
  - **Orden:** primero extracción del texto del DOM; en segundo lugar enlaces a PDF en la página y, por último, comprobación por `GET` por si la URL del detalle responde como binario PDF.  
  - Ya **no** hay `return` inmediato al detectar/descargar un PDF: el texto del cuerpo es la base y los PDFs son acción secundaria.  
  - Si se guardó al menos un PDF, al final del campo de texto añadimos la nota fija: `Archivo PDF descargado en almacen/pdfs`. (Las rutas concretas siguen en `extra.pdfs_descargados` de cada `TesisRecord`.)

- **`scrape_buscador_juridico`**  
  - Cada tarjeta parte del resumen de listado; si falla el detalle, se conserva **al menos** rubro y resumen de listado en el registro.  
  - Callback opcional **`on_record`**: se invoca justo después de añadir cada `TesisRecord` a la lista, para guardado incremental u otra lógica. Si el callback falla, se registra en el log y el scrape continúa.

### `app.py` (descarga masiva BJ)

- Tras cada tesis procesada, **`guardar_registro_bj_almacen`** se invoca vía `on_record`, no al acabar todas las páginas.  
- Cada guardado va en un **`try`/`except`**: si un registro no se puede escribir, se deja traza en el log y se sigue con la siguiente.

## CLI

`python -m scjn_tesis` mantiene los registros en memoria y un único `save_json` al final (no usa `on_record` salvo que se amplíe la CLI).
