"""
morfologia.py — Parámetros morfométricos de los polígonos de copa.

Port de las fórmulas de `mlip_qgis.py` (Max Line Inside Polygon, del repositorio
qgis-research-tools), para que los valores que salgan aquí sean **los mismos** que
daría esa herramienta sobre la capa de copas exportada a QGIS.

Se conservan los nombres de campo del original (todos de ≤10 caracteres, que es el
límite del DBF si alguien exporta a Shapefile):

    perimeter   perímetro del polígono (m)
    major_len   diagonal interior más larga: eje mayor (m)
    major_azim  azimut de esa diagonal, 0–180° (0 = N–S, 90 = E–O)
    orient      orientación dominante, del lado largo del rectángulo mínimo, 0–180°
    mid_width   anchura perpendicular al eje mayor, en su punto medio (m)
    max_width   anchura máxima: la paralela más larga dentro del polígono (m)
    elongation  major_len / max_width (1 = redonda, crece con lo alargada)
    compact     Polsby-Popper 4·π·A / P² (1 = círculo)
    rectang     A / área del rectángulo mínimo rotado (≈1 = rectángulo)
    convex      A / área de la envolvente convexa (<1 si hay entrantes)
    shape_idx   P / (2·√(π·A)) (1 = círculo, crece con la irregularidad del borde)

Dos avisos que valen para cualquier uso forestal de estos números:

1. El borde de las copas va por el filo del píxel. El perímetro de una escalera de
   píxeles está inflado hasta un ~27 % en los tramos diagonales, así que `compact` y
   `shape_idx`, que dividen por el perímetro, dependen del tamaño de píxel tanto como
   de la planta. Se calculan sobre el polígono ya simplificado, que es lo que atenúa
   el efecto, pero no lo quita. Para comparar copas entre sí, bien; como valor
   absoluto, no.
2. A 0,5 m de resolución una copa de 2 m son cuatro píxeles de ancho: los ejes salen
   cuantizados y la elongación de las copas pequeñas es sobre todo ruido. Filtra por
   `area_m2` antes de sacar conclusiones de forma.

Nada de esto falla hacia arriba: si shapely no está o un polígono da guerra, las
columnas salen NaN y el resto del análisis sigue igual.

Medido sobre la escena sintética (126 copas, píxel de 0,5 m), por si ahorra el disgusto
de sacar conclusiones de estas columnas sin mirarlas antes:

* `orient` sale **cuantizada a la malla de píxeles**: 61 copas a 0°, 28 a 90°, 5 a 45°.
  El rectángulo mínimo de un polígono en escalera se alinea con los ejes del ráster, así
  que esta columna mide sobre todo el ráster. No hagáis con ella una rosa de
  orientaciones de copas; para lagunas, que es para lo que nació, es otra historia.
* **La escena sintética no sirve para validar la forma**: `sintetico._arbol` genera las
  copas de pino y de sabina como **círculos** (radio R, ángulo uniforme); solo cambian el
  perfil vertical y el tronco. La elongación de aquí es la escalera de un círculo, no la
  planta de un árbol. Quien quiera saber si la forma separa especies necesita datos reales.
* `convex` **no delató las copas fundidas** en esta escena: 0,906 de media con un árbol
  real dentro, 0,908 con dos y 0,862 con tres. Lo que sí escala es el área (25,5 → 53,9 →
  86,2 m²). Tiene sentido: el borde de una copa fundida lo dibuja la línea de watershed,
  que es recta, no el entrante entre las dos copas.
* Coste: 4,5 ms por copa (0,6 s las 126), dominado por el barrido de perpendiculares.
"""
import math

import numpy as np
import pandas as pd

try:
    from shapely.geometry import LineString, Polygon
    HAY_SHAPELY = True
except Exception:                                                   # pragma: no cover
    HAY_SHAPELY = False

CAMPOS = ['perimeter', 'major_len', 'major_azim', 'orient', 'mid_width',
          'max_width', 'elongation', 'compact', 'rectang', 'convex', 'shape_idx']

N_PASOS = 50        # paralelas a cada lado del eje mayor; el original usa 50


# ------------------------------------------------------------------ geometría
def _poligono(ent):
    """Polígono shapely de una entidad de `vectores.py`. De un MultiPolygon se
    queda con la parte mayor, igual que hace la herramienta de QGIS."""
    if ent['tipo'] == "Polygon":
        anillos = ent['coords']
        return Polygon(anillos[0], anillos[1:])
    partes = [Polygon(a[0], a[1:]) for a in ent['coords']]
    return max(partes, key=lambda p: p.area)


def _azimut(x1, y1, x2, y2):
    """Azimut del segmento (desde el norte, horario), plegado a 0–180°."""
    return math.degrees(math.atan2(x2 - x1, y2 - y1)) % 180.0


def _diagonal_interior(poly):
    """Segmento más largo entre vértices del anillo exterior que queda enteramente
    dentro del polígono. Mismo resultado que el bucle O(n²) del original, pero
    probando los pares de mayor a menor y parando en el primero válido."""
    v = np.asarray(poly.exterior.coords)[:-1]
    n = len(v)
    if n < 2:
        return None
    d = np.hypot(v[:, 0][:, None] - v[:, 0][None, :], v[:, 1][:, None] - v[:, 1][None, :])
    iu = np.triu_indices(n, 1)
    # Orden estable: entre diagonales de la misma longitud (las copas rasterizadas son
    # muy simetricas y los empates son frecuentes) gana la del primer par de vertices,
    # que es la que elegiria el bucle del original.
    orden = np.argsort(-d[iu], kind='stable')
    for k in orden:
        i, j = iu[0][k], iu[1][k]
        linea = LineString([tuple(v[i]), tuple(v[j])])
        if linea.within(poly):
            return linea
    return None


def _orientacion_mrr(poly):
    """Orientación dominante (0–180°): lado más largo del rectángulo mínimo rotado.
    Es robusta, no depende de un vértice suelto como la diagonal."""
    try:
        c = list(poly.minimum_rotated_rectangle.exterior.coords)
    except Exception:
        return np.nan
    mejor, largo = None, -1.0
    for a, b in zip(c[:-1], c[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L > largo:
            largo, mejor = L, (a[0], a[1], b[0], b[1])
    return _azimut(*mejor) if mejor else np.nan


def _anchuras(poly, eje):
    """Anchura en el punto medio del eje mayor y anchura máxima. La perpendicular se
    desliza a lo largo del eje; para una recta, desplazarla equivale al
    `parallel_offset` del original, y evita su geometría de buffer."""
    (x1, y1), (x2, y2) = list(eje.coords)
    L = eje.length
    if L <= 0:
        return np.nan, np.nan
    ux, uy = (x2 - x1) / L, (y2 - y1) / L         # unitario del eje
    px, py = -uy, ux                              # su perpendicular
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def corte(cx, cy):
        rec = LineString([(cx - px * L, cy - py * L), (cx + px * L, cy + py * L)])
        return rec.intersection(poly).length

    media = corte(mx, my)
    paso = L / (2.0 * N_PASOS)
    maxima = max(corte(mx + ux * paso * i, my + uy * paso * i)
                 for i in range(-N_PASOS, N_PASOS + 1))
    return (media if media > 0 else np.nan), (maxima if maxima > 0 else np.nan)


# ------------------------------------------------------------------- métricas
def _de_un_poligono(poly):
    area, perim = poly.area, poly.length
    m = dict.fromkeys(CAMPOS, np.nan)
    m['perimeter'] = perim
    m['orient'] = _orientacion_mrr(poly)
    if perim > 0:
        m['compact'] = 4.0 * math.pi * area / (perim ** 2)
    if area > 0:
        m['shape_idx'] = perim / (2.0 * math.sqrt(math.pi * area))
        try:
            ah = poly.convex_hull.area
            if ah > 0:
                m['convex'] = area / ah
            ar = poly.minimum_rotated_rectangle.area
            if ar > 0:
                m['rectang'] = area / ar
        except Exception:
            pass
    eje = _diagonal_interior(poly)
    if eje is not None:
        c = list(eje.coords)
        m['major_len'] = eje.length
        m['major_azim'] = _azimut(c[0][0], c[0][1], c[1][0], c[1][1])
        m['mid_width'], m['max_width'] = _anchuras(poly, eje)
        ref = m['max_width'] if np.isfinite(m['max_width']) else m['mid_width']
        if np.isfinite(ref) and ref > 0:
            m['elongation'] = m['major_len'] / ref
    return m


def metricas(entidades, decimales=3):
    """DataFrame con `id` y los campos morfométricos de cada polígono de copa.
    Si shapely no está disponible devuelve la tabla con todo a NaN."""
    ids = [int(e['props']['id']) for e in entidades]
    if not HAY_SHAPELY or not ids:
        return pd.DataFrame({'id': ids, **{c: np.nan for c in CAMPOS}})
    filas = []
    for e in entidades:
        try:
            filas.append(_de_un_poligono(_poligono(e)))
        except Exception:
            filas.append(dict.fromkeys(CAMPOS, np.nan))
    df = pd.DataFrame(filas, columns=CAMPOS).round(decimales)
    df.insert(0, 'id', ids)
    return df


def anotar(entidades, df):
    """Copia las métricas a las `props` de cada entidad, emparejando por id."""
    por_id = {int(r['id']): r for _, r in df.iterrows()}
    for e in entidades:
        r = por_id.get(int(e['props']['id']))
        if r is None:
            continue
        for c in CAMPOS:
            v = r[c]
            e['props'][c] = None if (v is None or not np.isfinite(v)) else float(v)
    return entidades
