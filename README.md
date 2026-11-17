# ForestMap Doñana V3

Inventario forestal individual con LiDAR para el sabinar de Doñana (*Juniperus phoenicea* subsp. *turbinata*).

## Instalación (Ubuntu)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt   # incluye CSF (clasificación de suelo)
streamlit run app.py
```
En Windows: `py -m venv .venv`, `.venv\Scripts\activate`, y el mismo `pip install` y `streamlit run app.py`.

## Versión web (Streamlit Community Cloud)
`requirements.txt` no incluye CSF para que la instalación en la nube no falle; en ese caso la app usa
automáticamente el clasificador de suelo morfológico. Para la demo online use recortes pequeños de nube (la RAM es limitada).
Variable opcional `FORESTMAP_DEMO=1` (en *Secrets*: `FORESTMAP_DEMO = "1"`) muestra el botón de escena sintética.

## Flujo (7 pasos)
1. Configuración (épocas, EPSG) · 2. Carga LAS/LAZ + parcela (`parcela.gpkg`) · 3. DEM/DSM/CHM + perfil vertical ·
4. Antena, vallado y sensores (tabla editable) · 5. Árboles, capas de altura (0,5 m), volúmenes, auditoría de ≥ 8 m ·
6. Cambios entre épocas (ΔH, mortalidad/reclutamiento) · 7. Informe HTML + paquete QGIS (ZIP).

## Módulos
`procesado` (DEM/DSM/CHM) · `copas` (ápices, copas, capas, volúmenes) · `vectores` (ráster→polígono exacto) ·
`estructuras` (antena/vallado/instrumentos) · `roi` · `pipeline` · `figuras` (informe) · `graficos` (Plotly) ·
`informe` · `exportar` · `morfologia` (forma de las copas) · `sintetico` (escena de prueba) · `pruebas/`.

## Pruebas
`bash ejecutar_pruebas.sh` (con datos sintéticos) y `python pruebas/stub_streamlit.py` (humo de la interfaz sin Streamlit).

## Novedades V3.1 (datos reales del Ojillo)
- Lector LAS propio (funciona sin laspy con .las sin comprimir; para .laz: `pip install 'laspy[lazrs]'`) y uso opcional de la clase 2 del archivo.
- Parcela desde `parcela.gpkg` sin geopandas (sqlite3). Por debajo de 1,3 m = suelo. Pino = ápice > 5 m, un pino = un individuo.
- Antena con brazos/sensores; lo dudoso (confianza baja) no se excluye.
- Dos épocas: emparejamiento de ápices a ≤ 1,2 m (protocolo ICTS) o copas solapadas → persisten / crecieron / estables / perdieron altura / mortalidad probable / nuevos, por especie.
- Validación con inventario de campo (xlsx): estima el desplazamiento y cuenta fichas emparejadas y tapadas.
- Informe con algoritmos, vistas 3D de la nube y comparación con campo.

## Novedades V3.2 (forma de las copas)
La tabla de árboles y la capa `Copas_<época>` llevan once columnas más con la morfometría de
cada polígono de copa, calculadas con las mismas fórmulas que la herramienta **Max Line Inside
Polygon** de [qgis-research-tools](https://github.com/Digdgeo/qgis-research-tools), de modo que
los valores coinciden con los que daría QGIS sobre la capa exportada:

| campo | qué es |
|---|---|
| `perimeter` | perímetro del polígono (m) |
| `major_len` · `major_azim` | diagonal interior más larga (eje mayor) y su azimut, 0–180° |
| `exter_len` · `exter_azim` | diagonal **secante**: la más larga que se sale por un entrante, y su azimut. Vacía en copas convexas |
| `orient` | orientación del lado largo del rectángulo mínimo rotado, 0–180° |
| `mid_width` · `max_width` | anchura perpendicular al eje mayor: en su punto medio y la máxima |
| `elongation` | `major_len / max_width` |
| `compact` · `rectang` · `convex` · `shape_idx` | índices de forma (Polsby-Popper, rectangularidad, convexidad, crenulación) |

Además se exporta la capa de líneas **`Copas_lineas_<época>`**, con las cuatro que dibuja la
herramienta de QGIS —`eje`, `secante`, `anch_med` y `anch_max`— y sus campos `id`, `linea`,
`largo_m` y `azimut`. En QGIS se estiliza por categorías sobre `linea` y se ve encima de los
polígonos de copa de dónde sale cada número.

Se activa con la casilla **«Parámetros de forma de la copa»** del paso 5 (por defecto sí) y
cuesta unos 4,5 ms por copa. Si falta `shapely`, las columnas salen vacías y el resto sigue igual.

**Antes de usarlas**, léase el encabezado de `morfologia.py`: sobre copas rasterizadas a 0,5 m,
`orient` sale cuantizada a la malla de píxeles y los índices que dividen por el perímetro
arrastran el sesgo de la escalera de píxeles. Las que se portan son las de longitud
(`major_len`, `max_width`, `elongation`).

## Notas
- Sin límite superior de altura: los individuos ≥ umbral (8 m) se marcan para auditoría, no se descartan.
- La antena y estructuras se excluyen del recuento; revise la tabla del paso 4.
- Con copas muy solapadas el CHM no permite separar individuos: el recuento es cota inferior.
- Sin `rasterio`, los ráster se exportan como `.asc` + `.prj`; sin `geopandas`, solo GeoJSON.
- `Informe_ForestMap.html` se imprime a PDF desde el navegador.