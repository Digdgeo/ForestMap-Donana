"""Pruebas de morfologia.py: figuras de geometria conocida y comparacion contra el
bucle original de mlip_qgis.py (qgis-research-tools)."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

import morfologia as mo

if not mo.HAY_SHAPELY:
    print("sin shapely: morfologia devuelve NaN, nada que comprobar")
    sys.exit(0)

from shapely.geometry import LineString, Point, Polygon


def ent(exterior, huecos=(), i=1):
    return dict(tipo="Polygon", coords=[np.asarray(exterior)] + [np.asarray(h) for h in huecos],
                props=dict(id=i))


# 1) cuadrado de 10 m: diagonal 10√2, anchura maxima 10, compacidad pi/4, rectangularidad 1
cuad = ent([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
m = mo.metricas([cuad]).iloc[0]
assert abs(m['perimeter'] - 40) < 1e-6, m['perimeter']
assert abs(m['major_len'] - 10 * math.sqrt(2)) < 1e-3, m['major_len']
# ojo: la anchura es la cuerda mas larga PERPENDICULAR al eje mayor, y en un cuadrado
# el eje mayor es una diagonal, asi que su perpendicular es la otra diagonal, no el lado.
assert abs(m['max_width'] - 10 * math.sqrt(2)) < 1e-2, m['max_width']
assert abs(m['elongation'] - 1.0) < 1e-2, m['elongation']
assert abs(m['compact'] - math.pi / 4) < 1e-3, m['compact']
assert abs(m['rectang'] - 1.0) < 1e-6, m['rectang']
assert abs(m['convex'] - 1.0) < 1e-6, m['convex']
print("cuadrado OK: eje %.3f, anchura %.3f, compact %.3f" % (m['major_len'], m['max_width'], m['compact']))

# 2) rectangulo 20x5 tumbado E-O: orientacion 90, elongacion ~ 20.6/5
rect = ent([(0, 0), (20, 0), (20, 5), (0, 5), (0, 0)])
m = mo.metricas([rect]).iloc[0]
assert abs(m['orient'] - 90.0) < 1e-6, m['orient']
# perpendicular a la diagonal (20,5): la cuerda mas larga en esa direccion mide
# 5 * sqrt(425) / 20 = 5.154 m, limitada por los lados largos
assert abs(m['max_width'] - 5 * math.sqrt(425) / 20) < 1e-2, m['max_width']
assert abs(m['elongation'] - 4.0) < 0.05, m['elongation']
print("rectangulo OK: orient %.1f, elongacion %.3f" % (m['orient'], m['elongation']))

# 3) circulo: compacidad -> 1, shape_idx -> 1, elongacion -> 1
a = np.linspace(0, 2 * np.pi, 361)
circ = ent(np.c_[5 * np.cos(a), 5 * np.sin(a)])
m = mo.metricas([circ]).iloc[0]
assert abs(m['compact'] - 1.0) < 0.01, m['compact']
assert abs(m['shape_idx'] - 1.0) < 0.01, m['shape_idx']
assert abs(m['elongation'] - 1.0) < 0.01, m['elongation']
print("circulo OK: compact %.3f, shape_idx %.3f, elongacion %.3f" % (m['compact'], m['shape_idx'], m['elongation']))

# 4) forma en C: la envolvente convexa delata el entrante, el eje mayor NO sale de ella
c = ent([(0, 0), (10, 0), (10, 3), (3, 3), (3, 7), (10, 7), (10, 10), (0, 10), (0, 0)])
m, ln = mo.metricas([c], con_lineas=True)
m = m.iloc[0]
assert m['convex'] < 0.75, m['convex']
poly = Polygon(c['coords'][0])
assert not LineString([(0, 0), (10, 10)]).within(poly)
# la secante existe (hay entrante) y es mas larga que el eje interior
assert np.isfinite(m['exter_len']), m['exter_len']
assert m['exter_len'] > m['major_len'], (m['exter_len'], m['major_len'])
assert abs(m['exter_len'] - 10 * math.sqrt(2)) < 1e-3, m['exter_len']   # redondeo a 3 decimales
tipos = sorted(e['props']['linea'] for e in ln)
assert tipos == ['anch_max', 'anch_med', 'eje', 'secante'], tipos
assert all(e['tipo'] == "LineString" for e in ln)
assert all(int(e['props']['id']) == 1 for e in ln)
por = {e['props']['linea']: e for e in ln}
assert abs(por['eje']['props']['largo_m'] - m['major_len']) < 1e-3
assert abs(por['secante']['props']['largo_m'] - m['exter_len']) < 1e-3
print("forma en C OK: convex %.3f, eje %.3f, secante %.3f, 4 lineas"
      % (m['convex'], m['major_len'], m['exter_len']))

# 4 bis) poligono convexo: no hay secante posible, la columna va vacia
m2, ln2 = mo.metricas([ent([(0, 0), (6, 0), (6, 4), (0, 4), (0, 0)], i=2)], con_lineas=True)
assert not np.isfinite(m2.iloc[0]['exter_len']), m2.iloc[0]['exter_len']
assert sorted(e['props']['linea'] for e in ln2) == ['anch_max', 'anch_med', 'eje']
print("convexo OK: sin secante, 3 lineas")

# 5) poligono con hueco: el area baja, la convexidad tambien
anillo = ent([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
             huecos=[[(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]])
m = mo.metricas([anillo]).iloc[0]
assert m['convex'] < 1.0, m['convex']
print("con hueco OK: convex %.3f" % m['convex'])


# 6) mismos numeros que el bucle original de mlip_qgis.py, sobre formas irregulares
def diag_original(poly):
    vs = [Point(cc) for cc in poly.exterior.coords]
    m_in, din = 0.0, None
    for n, e in enumerate(vs):
        for nn, ee in enumerate(vs):
            if nn <= n:
                continue
            d = e.distance(ee)
            if d > m_in:
                ln = LineString([(e.x, e.y), (ee.x, ee.y)])
                if ln.within(poly):
                    din, m_in = ln, d
    return din


rng = np.random.default_rng(3)
peor = 0.0
for k in range(25):
    n = rng.integers(6, 20)
    th = np.sort(rng.random(n) * 2 * np.pi)
    rr = 2 + rng.random(n) * 6
    p = np.c_[rr * np.cos(th), rr * np.sin(th)]
    p = np.vstack([p, p[:1]])
    poly = Polygon(p)
    if not poly.is_valid or poly.area <= 0:
        continue
    d0 = diag_original(poly)
    m = mo.metricas([ent(p)]).iloc[0]
    if d0 is None:
        assert not np.isfinite(m['major_len'])
        continue
    peor = max(peor, abs(d0.length - m['major_len']))
assert peor < 1e-3, peor
print("coincide con el bucle original de mlip_qgis.py en 25 poligonos (peor desvio %.1e m)" % peor)

# 7) sin shapely o con basura: NaN, nunca excepcion
malo = dict(tipo="Polygon", coords=[np.array([(0, 0), (1, 0)])], props=dict(id=9))
m = mo.metricas([malo]).iloc[0]
assert int(m['id']) == 9
print("poligono degenerado OK: no revienta")

print("OK morfologia")
