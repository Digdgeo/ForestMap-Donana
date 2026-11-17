"""
copas.py — Individuos, copas, capas de altura y volúmenes a partir del CHM.

Flujo
-----
1. `detectar_copas`   ápices (h-máximos o LMF adaptativo) + cuencas (watershed) = copas.
2. `capas_altura`     reclasificación del CHM en bandas (p. ej. cada 0,5 m) y volúmenes por capa.
3. `construir_vectores`  polígonos de copa, de capa (árbol × banda) y ápices.
4. `volumen_ocupado`  volumen por vóxeles a partir de la nube (opcional).
5. `emparejar_epocas` supervivencia / mortalidad / reclutamiento entre dos épocas.

Garantía: NO existe ningún límite superior de altura. Cada zona de copa conectada
con altura ≥ hmin recibe al menos un individuo (marcador forzado si hace falta) y los
objetos descartados (área mínima) quedan registrados en la tabla de descartes.
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

import vectores as vc


# ------------------------------------------------------------------ parámetros
@dataclass
class ParamsArboles:
    metodo: str = "hmax"            # "hmax" (h-máximos, recomendado) | "lmf" (compatible lidR/ForestMAP)
    sigma_m: float = 0.2            # suavizado gaussiano del CHM solo para buscar ápices
    prominencia_m: float = 0.05     # h-máximos: desnivel mínimo entre dos copas vecinas
    rel_min_copa: float = 0.25      # un píxel pertenece a la copa si h ≥ rel_min·h_ápice (0 = sin límite)
    ws_base_m: float = 1.5          # LMF: diámetro de ventana = ws_base + ws_k·h
    ws_k: float = 0.6
    ws_min_m: float = 2.0
    ws_max_m: float = 14.0
    hmin_m: float = 1.3             # altura mínima de un individuo (sin límite superior); por debajo = "suelo"
    hcorte_m: float = 1.3           # CHM ≥ hcorte = copa / cobertura (por debajo se trata como suelo: matojos, herbáceas)
    h_pino_m: float = 5.0           # copas con ápice ≥ este valor = pino (el resto, sabina)
    fusion_pinos: bool = True       # un pino = un individuo (no se trocea su copa grande)
    pino_fusion_hmin_m: float = 6.0 # solo se fusionan sub-copas de pinos grandes (ápice ≥ este valor)
    pino_sigma_m: float = 0.5       # suavizado y prominencia gruesos para delimitar copas de pino
    pino_prom_m: float = 0.5
    area_min_m2: float = 0.30       # copas menores se registran como descarte (ruido)
    ancho_capa_m: float = 0.5       # grosor de las capas de altura
    umbral_alto_m: float = 8.0      # los individuos ≥ este valor se marcan para auditoría

    def dict(self):
        return asdict(self)


# ------------------------------------------------------------------- utilidades
def suavizar(chm, res, sigma_m):
    if sigma_m <= 0:
        return chm.copy()
    return ndimage.gaussian_filter(chm, sigma=sigma_m / res, mode="nearest")


def _plateaus_a_puntos(binaria, imagen):
    lab, n = ndimage.label(binaria, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros((0, 2), int)
    pos = ndimage.maximum_position(imagen, lab, np.arange(1, n + 1))
    return np.array(pos, dtype=int).reshape(-1, 2)


def apices_hmax(chm, chm_s, res, prominencia_m, hmin_m):
    """Máximos con desnivel (dinámica) ≥ prominencia. Sin ventana ni límite de altura."""
    img = np.round(chm_s * 1000).astype(np.int32)
    h = max(1, int(round(prominencia_m * 1000)))
    mx = h_maxima(img, h)
    pts = _plateaus_a_puntos(mx > 0, chm_s)
    if len(pts) == 0:
        return pts
    raw = ndimage.maximum_filter(chm, size=3, mode="nearest")[pts[:, 0], pts[:, 1]]
    return pts[raw >= hmin_m]


def apices_lmf(chm, chm_s, res, p):
    """LMF con ventana variable según la altura del candidato (como lidR::lmf)."""
    cand = chm_s == ndimage.maximum_filter(chm_s, size=3, mode="nearest")
    cand &= chm >= p.hmin_m
    pts = _plateaus_a_puntos(cand, chm_s)
    keep = []
    ny, nx = chm_s.shape
    for k, (r, c) in enumerate(pts):
        h = chm_s[r, c]
        ws = float(np.clip(p.ws_base_m + p.ws_k * h, p.ws_min_m, p.ws_max_m))
        rad = max(1, int(round(ws / 2 / res)))
        r0, r1, c0, c1 = max(r - rad, 0), min(r + rad + 1, ny), max(c - rad, 0), min(c + rad + 1, nx)
        yy, xx = np.ogrid[r0 - r:r1 - r, c0 - c:c1 - c]
        disco = yy ** 2 + xx ** 2 <= rad ** 2
        if h >= chm_s[r0:r1, c0:c1][disco].max() - 1e-9:
            keep.append(k)
    return pts[keep]


# --------------------------------------------------------------- detección
def detectar_copas(chm, malla, res, p, excluir=None, roi=None):
    """Individuos y copas. Devuelve dict con labels, tabla, descartes y chm_s.

    chm      : CHM (m).   malla = (x0, y1, nx, ny).
    excluir  : máscara booleana de estructuras (antena, vallado…) que no son árboles.
    roi      : máscara booleana de la parcela. Un individuo pertenece a la parcela si su
               ápice cae dentro; sus píxeles fuera de la parcela se descartan.
    """
    x0, y1, _, _ = malla
    ny, nx = chm.shape
    chm = np.nan_to_num(chm, nan=0.0)
    if excluir is not None and excluir.any():
        chm = np.where(excluir, 0.0, chm)
    roi_m = np.ones(chm.shape, bool) if roi is None else roi
    hcorte = min(p.hcorte_m, p.hmin_m)
    chm_s = suavizar(chm, res, p.sigma_m)

    # 1) ápices
    if p.metodo == "lmf":
        pts = apices_lmf(chm, chm_s, res, p)
    else:
        pts = apices_hmax(chm, chm_s, res, p.prominencia_m, p.hmin_m)

    # 2) zona de copa, cuencas y garantía de completitud
    fg = chm_s >= hcorte
    origen = {}                                                     # id de marcador -> origen
    marc = np.zeros(chm.shape, np.int32)
    for k, (r, c) in enumerate(pts, 1):
        marc[r, c] = k
        origen[k] = "ápice"
    fg |= marc > 0
    img = np.round(chm_s * 1000).astype(np.int32)

    def _anade_marcadores(lab_actual, etiqueta, abrir=False, area_min=None):
        """Un marcador por cada zona de copa sin individuo cuya altura sea ≥ hmin.

        `abrir` elimina los anillos finos que deja el recorte relativo en el borde de
        las copas (no son individuos) y exige que la zona sea una mancha compacta.
        """
        resto = fg & (lab_actual == 0)
        if abrir:
            resto = ndimage.binary_opening(resto, structure=np.ones((3, 3)))
        comp, ncomp = ndimage.label(resto, structure=np.ones((3, 3)))
        if ncomp == 0:
            return 0
        area_min = p.area_min_m2 if area_min is None else area_min
        mx = ndimage.maximum(chm, comp, np.arange(1, ncomp + 1))
        ar = np.bincount(comp.ravel(), minlength=ncomp + 1)[1:] * res * res
        ids = [i for i in range(1, ncomp + 1) if mx[i - 1] >= p.hmin_m and ar[i - 1] >= area_min]
        if not ids:
            return 0
        pos = ndimage.maximum_position(chm_s, comp, ids)
        base = int(marc.max())
        for j, (r, c) in enumerate(pos):
            marc[r, c] = base + j + 1
            origen[base + j + 1] = etiqueta
        return len(ids)

    lab = watershed(-img, marc, mask=fg)
    _anade_marcadores(lab, "forzado (zona sin ápice)")
    lab = watershed(-img, marc, mask=fg)
    if p.rel_min_copa > 0:                                          # evita absorber arbustos vecinos
        for _ in range(2):
            n_tmp = int(lab.max())
            hm = ndimage.maximum(chm_s, lab, np.arange(1, n_tmp + 1))
            umbral = np.r_[0.0, p.rel_min_copa * hm]
            lab = np.where(chm_s >= umbral[lab], lab, 0)
            if _anade_marcadores(lab, "recuperado (2.ª pasada)", abrir=True,
                                 area_min=max(1.0, p.area_min_m2)) == 0:
                break
            lab = watershed(-img, marc, mask=fg)
        n_tmp = int(lab.max())
        hm = ndimage.maximum(chm_s, lab, np.arange(1, n_tmp + 1))
        umbral = np.r_[0.0, p.rel_min_copa * hm]
        lab = np.where(chm_s >= umbral[lab], lab, 0)
    if p.fusion_pinos:
        lab = _fusionar_pinos(lab, chm, chm_s, fg, res, p)
        origen = {k: v for k, v in origen.items() if k in set(np.unique(lab))}
    forzados = {k: v for k, v in origen.items() if v != "ápice"}

    # 3) atributos por copa (todo el dominio) y filtros
    n = int(lab.max())
    if n == 0:
        return dict(labels=np.zeros(chm.shape, np.int32), tabla=_tabla_vacia(), descartes=_descartes_vacio(),
                    chm_s=chm_s, params=p.dict(), n_forzados=0)
    idx = np.arange(1, n + 1)
    area_px = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    hmax = ndimage.maximum(chm, lab, idx)
    apex = np.array(ndimage.maximum_position(chm, lab, idx), dtype=int).reshape(-1, 2)
    area_in = np.bincount(lab[roi_m].ravel(), minlength=n + 1)[1:]
    apex_dentro = roi_m[apex[:, 0], apex[:, 1]]

    motivo = np.full(n, "", dtype=object)
    motivo[area_px * res * res < p.area_min_m2] = "área < mínima (ruido)"
    motivo[(motivo == "") & (hmax < p.hmin_m)] = "altura < mínima"
    motivo[(motivo == "") & (~apex_dentro)] = "ápice fuera de la parcela"
    ok = motivo == ""

    desc = pd.DataFrame({
        'x': x0 + (apex[:, 1] + 0.5) * res, 'y': y1 - (apex[:, 0] + 0.5) * res,
        'h_m': hmax, 'area_m2': area_px * res * res, 'motivo': motivo})[~ok & (motivo != "ápice fuera de la parcela")]

    nuevo_id = np.zeros(n + 1, np.int32)
    orden = np.argsort(-hmax[ok], kind="stable")                   # id 1 = el más alto
    nuevo_id[idx[ok][orden]] = np.arange(1, ok.sum() + 1)
    lab2 = nuevo_id[lab]
    lab2[~roi_m] = 0

    tabla = _tabla_copas(lab2, chm, malla, res, p, roi_m, lab, nuevo_id, area_px, area_in, forzados, ok)
    return dict(labels=lab2.astype(np.int32), chm_s=chm_s, descartes=desc.reset_index(drop=True),
                tabla=tabla, params=p.dict(), n_forzados=int((tabla['origen'] != 'ápice').sum()))


def _fusionar_pinos(lab, chm, chm_s, fg, res, p):
    """Un pino = un individuo. Delimita con parámetros gruesos las copas cuyo ápice ≥ h_pino_m y
    une en una sola las copas finas que sean sub-picos del mismo pino (≥ 80 % de su altura)."""
    if lab.max() == 0:
        return lab
    cs = suavizar(chm, res, p.pino_sigma_m)
    pts = apices_hmax(chm, cs, res, p.pino_prom_m, p.pino_fusion_hmin_m)
    if len(pts) == 0:
        return lab
    marc = np.zeros(chm.shape, np.int32)
    for k, (r, c) in enumerate(pts, 1):
        marc[r, c] = k
    grueso = watershed(-np.round(cs * 1000).astype(np.int32), marc, mask=(cs >= p.hcorte_m))
    n = int(lab.max())
    idx = np.arange(1, n + 1)
    hmax_f = ndimage.maximum(chm, lab, idx)
    cen = np.array(ndimage.maximum_position(chm, lab, idx), dtype=int).reshape(-1, 2)
    g_of = grueso[cen[:, 0], cen[:, 1]]                            # copa gruesa donde cae el ápice de cada copa fina
    hs_ap = cs[cen[:, 0], cen[:, 1]]
    hm_g = ndimage.maximum(chm, grueso, np.arange(1, len(pts) + 1))
    nueva = lab.copy()
    for g in range(1, len(pts) + 1):
        if hm_g[g - 1] < p.pino_fusion_hmin_m:
            continue
        miembros = idx[(g_of == g) & (hmax_f >= 0.8 * hm_g[g - 1]) & (hs_ap >= 0.6 * hm_g[g - 1])]
        if len(miembros) > 1:
            nueva[np.isin(lab, miembros)] = miembros[0]
    return nueva


def _tabla_vacia():
    return pd.DataFrame(columns=['id', 'x', 'y', 'h_m', 'h_media_m', 'h_p95_m', 'area_m2', 'diam_eq_m',
                                 'vol_m3', 'frac_roi', 'forzado', 'origen', 'alto', 'especie', 'cx', 'cy'])


def _descartes_vacio():
    return pd.DataFrame(columns=['x', 'y', 'h_m', 'area_m2', 'motivo'])


def _tabla_copas(lab2, chm, malla, res, p, roi_m, lab_full, nuevo_id, area_px_full, area_in_full, forzados, ok):
    x0, y1, _, _ = malla
    n = int(lab2.max())
    if n == 0:
        return _tabla_vacia()
    idx = np.arange(1, n + 1)
    cnt = np.bincount(lab2.ravel(), minlength=n + 1)[1:]
    suma = np.bincount(lab2.ravel(), weights=chm.ravel(), minlength=n + 1)[1:]
    hmax = ndimage.maximum(chm, lab2, idx)
    apex = np.array(ndimage.maximum_position(chm, lab2, idx), dtype=int).reshape(-1, 2)
    cen = np.array(ndimage.center_of_mass(np.ones_like(chm), lab2, idx)).reshape(-1, 2)
    p95 = ndimage.labeled_comprehension(chm, lab2, idx, lambda v: float(np.percentile(v, 95)), float, 0.0)
    # fracción dentro de la parcela (antes del recorte)
    viejo_a_nuevo = nuevo_id[1:]
    frac = np.ones(n)
    org = np.full(n, "ápice", dtype=object)
    for v_id in np.flatnonzero(viejo_a_nuevo > 0):
        nid = viejo_a_nuevo[v_id]
        frac[nid - 1] = area_in_full[v_id] / max(area_px_full[v_id], 1)
        org[nid - 1] = forzados.get(v_id + 1, "ápice")
    area = cnt * res * res
    return pd.DataFrame({
        'id': idx,
        'x': x0 + (apex[:, 1] + 0.5) * res, 'y': y1 - (apex[:, 0] + 0.5) * res,
        'h_m': hmax, 'h_media_m': suma / np.maximum(cnt, 1), 'h_p95_m': p95,
        'area_m2': area, 'diam_eq_m': 2 * np.sqrt(area / np.pi),
        'vol_m3': suma * res * res, 'frac_roi': frac, 'forzado': org != "ápice", 'origen': org,
        'alto': hmax >= p.umbral_alto_m,
        'especie': np.where(hmax >= p.h_pino_m, 'Pino', 'Sabina'),
        'cx': x0 + (cen[:, 1] + 0.5) * res, 'cy': y1 - (cen[:, 0] + 0.5) * res,
    })


# ---------------------------------------------------------------- capas de altura
def bordes_capas(hmax, ancho):
    """Bordes de capa [0, ancho, 2·ancho, …] hasta cubrir hmax (máx. 80 capas)."""
    ancho = float(ancho)
    while hmax / ancho > 80:
        ancho *= 2
    nb = int(np.floor(hmax / ancho)) + 1
    return np.arange(nb + 1) * ancho, ancho


def capas_altura(chm, labels, res, ancho, hcorte, mascara=None):
    """Volúmenes por capa y por individuo.

    Devuelve dict con:
      bordes, clase (ráster de índice de capa, -1 fuera), vol_capa (n_arboles × n_capas, estrato
      horizontal), area_clase (n_arboles × n_capas, píxeles de la clase de altura) y la tabla global.
    """
    m = labels > 0 if mascara is None else mascara & (chm >= hcorte)
    hmax = float(chm[m].max()) if m.any() else hcorte
    bordes, ancho = bordes_capas(max(hmax, ancho), ancho)
    nb = len(bordes) - 1
    clase = np.where(m, np.minimum(np.floor(chm / ancho).astype(int), nb - 1), -1)
    n = int(labels.max())
    vol = np.zeros((n, nb))
    area = np.zeros((n, nb))
    for b in range(nb):
        w = np.clip(chm - bordes[b], 0, ancho) * res * res
        if n:
            vol[:, b] = np.bincount(labels.ravel(), weights=np.where(labels > 0, w, 0).ravel(),
                                    minlength=n + 1)[1:]
        sel = (clase == b) & (labels > 0)
        if n and sel.any():
            area[:, b] = np.bincount(labels[sel], minlength=n + 1)[1:] * res * res
    # tabla global (todo el CHM ≥ hcorte de la máscara), slab = estrato horizontal
    filas = []
    for b in range(nb):
        w = np.clip(chm - bordes[b], 0, ancho) * res * res
        sel = clase == b
        filas.append(dict(capa=b + 1, h_min_m=bordes[b], h_max_m=bordes[b + 1],
                          area_clase_m2=float(sel.sum() * res * res),
                          vol_estrato_m3=float(np.where(m, w, 0).sum()),
                          vol_clase_m3=float(chm[sel].sum() * res * res)))
    glob = pd.DataFrame(filas)
    return dict(bordes=bordes, ancho=ancho, clase=clase, vol_capa=vol, area_clase=area, global_=glob, mascara=m)


# ---------------------------------------------------------------------- vectores
def _geom(partes):
    if len(partes) == 1:
        e, h = partes[0]
        return "Polygon", [e] + list(h)
    return "MultiPolygon", [[e] + list(h) for e, h in partes]


def construir_vectores(malla, res, labels, chm, tabla, capas, simplificar=0.0):
    """Entidades GeoJSON-like: copas, capas (árbol×banda), capas globales y ápices."""
    x0, y1, _, _ = malla
    out = {'copas': [], 'capas_arbol': [], 'capas_global': [], 'apices': [], 'lineas_copa': []}
    if len(tabla) == 0:
        return out

    pol = vc.poligonizar(labels, x0, y1, res, simplificar)
    for _, t in tabla.iterrows():
        partes = pol.get(int(t['id']))
        if not partes:
            continue
        tipo, coords = _geom(partes)
        out['copas'].append(dict(tipo=tipo, coords=coords, props=dict(
            id=int(t['id']), h_m=round(t['h_m'], 3), h_media=round(t['h_media_m'], 3),
            area_m2=round(t['area_m2'], 3), diam_eq=round(t['diam_eq_m'], 3),
            vol_m3=round(t['vol_m3'], 3), frac_roi=round(t['frac_roi'], 3),
            forzado=bool(t['forzado']), alto=bool(t['alto']), especie=str(t['especie']))))
        out['apices'].append(dict(tipo="Point", coords=(t['x'], t['y']), props=dict(
            id=int(t['id']), h_m=round(t['h_m'], 3), diam_eq=round(t['diam_eq_m'], 3),
            vol_m3=round(t['vol_m3'], 3), alto=bool(t['alto']), especie=str(t['especie']))))

    nb = len(capas['bordes']) - 1
    clase = capas['clase']
    sel = (labels > 0) & (clase >= 0)
    codigo = np.zeros(labels.shape, np.int64)
    codigo[sel] = labels[sel].astype(np.int64) * 1000 + clase[sel] + 1
    pol_c = vc.poligonizar(codigo, x0, y1, res, simplificar)
    # estadísticas por (árbol, capa)
    flat_c = codigo[sel]
    hs = chm[sel]
    cod_u, inv = np.unique(flat_c, return_inverse=True)
    sum_h = np.bincount(inv, weights=hs)
    cnt = np.bincount(inv)
    stats = {int(c): (s, k) for c, s, k in zip(cod_u, sum_h, cnt)}
    for c, partes in pol_c.items():
        s, k = stats[c]
        tid, b = c // 1000, c % 1000 - 1
        tipo, coords = _geom(partes)
        out['capas_arbol'].append(dict(tipo=tipo, coords=coords, props=dict(
            arbol=int(tid), capa=int(b + 1), h_min=round(float(capas['bordes'][b]), 3),
            h_max=round(float(capas['bordes'][b + 1]), 3), area_m2=round(k * res * res, 3),
            h_media=round(s / k, 3), vol_m3=round(s * res * res, 3))))

    # capas globales: todas las celdas de copa de la máscara
    cl = np.where(capas['mascara'], clase + 1, 0).astype(np.int64)
    pol_g = vc.poligonizar(cl, x0, y1, res, simplificar)
    for c, partes in pol_g.items():
        b = c - 1
        m = cl == c
        tipo, coords = _geom(partes)
        out['capas_global'].append(dict(tipo=tipo, coords=coords, props=dict(
            capa=int(c), h_min=round(float(capas['bordes'][b]), 3),
            h_max=round(float(capas['bordes'][b + 1]), 3),
            area_m2=round(float(m.sum() * res * res), 3),
            h_media=round(float(chm[m].mean()), 3),
            vol_m3=round(float(chm[m].sum() * res * res), 3))))
    return out


# ---------------------------------------------------------------- volumen por vóxeles
def volumen_ocupado(x, y, hag, labels, malla, res, vox=0.25):
    """Volumen ocupado (m³) por individuo, contando vóxeles con al menos un retorno."""
    x0, y1, nx, ny = malla
    n = int(labels.max())
    if n == 0 or len(x) == 0:
        return np.zeros(n)
    col = np.floor((x - x0) / res).astype(int)
    fil = np.floor((y1 - y) / res).astype(int)
    ok = (col >= 0) & (col < nx) & (fil >= 0) & (fil < ny) & (hag > 0)
    lab = np.zeros(len(x), np.int64)
    lab[ok] = labels[fil[ok], col[ok]]
    ok &= lab > 0
    if not ok.any():
        return np.zeros(n)
    ix = np.floor((x[ok] - x0) / vox).astype(np.int64)
    iy = np.floor((y1 - y[ok]) / vox).astype(np.int64)
    iz = np.floor(hag[ok] / vox).astype(np.int64)
    NX, NY, NZ = int(ix.max()) + 1, int(iy.max()) + 1, int(iz.max()) + 1
    clave = ((lab[ok] * NZ + iz) * NY + iy) * NX + ix
    u = np.unique(clave)
    lab_u = u // (NZ * NY * NX)
    return np.bincount(lab_u, minlength=n + 1)[1:].astype(float) * vox ** 3


# -------------------------------------------------------------------- resumen de rodal
CLASES_ALTURA = [0, 1, 2, 3, 4, 6, 8, 12, 100]


def resumen_rodal(tabla, chm, labels, roi, res, hcorte, excluir=None, capas=None):
    """Indicadores de la parcela (todo dentro del ROI)."""
    area_ha = roi.sum() * res * res / 1e4
    cob = (chm >= hcorte) & roi
    if excluir is not None:
        cob &= ~excluir
    cob_m2 = cob.sum() * res * res
    out = dict(area_roi_ha=area_ha, n_arboles=int(len(tabla)),
               densidad_ha=len(tabla) / area_ha if area_ha else np.nan,
               cobertura_pct=100 * cob_m2 / max(roi.sum() * res * res, 1e-9),
               cobertura_ha=cob_m2 / 1e4,
               volumen_total_m3=float(tabla['vol_m3'].sum()) if len(tabla) else 0.0,
               volumen_ha=float(tabla['vol_m3'].sum()) / area_ha if len(tabla) and area_ha else 0.0)
    if len(tabla):
        h = tabla['h_m']
        out.update(h_media=float(h.mean()), h_mediana=float(h.median()), h_p95=float(h.quantile(.95)),
                   h_max=float(h.max()), h_cv=float(h.std() / h.mean()) if h.mean() else np.nan,
                   diam_medio=float(tabla['diam_eq_m'].mean()), area_copa_media=float(tabla['area_m2'].mean()),
                   n_altos=int(tabla['alto'].sum()), n_forzados=int(tabla['forzado'].sum()))
    else:
        out.update(h_media=np.nan, h_mediana=np.nan, h_p95=np.nan, h_max=np.nan, h_cv=np.nan,
                   diam_medio=np.nan, area_copa_media=np.nan, n_altos=0, n_forzados=0)
    cl = pd.cut(tabla['h_m'], CLASES_ALTURA, right=False) if len(tabla) else None
    if cl is not None:
        g = tabla.groupby(cl, observed=False).agg(n=('id', 'count'), vol_m3=('vol_m3', 'sum'),
                                                    area_m2=('area_m2', 'sum')).reset_index()
        g['clase'] = [f"{int(i.left)}–{int(i.right)} m" if i.right < 100 else f"≥ {int(i.left)} m" for i in g['h_m']]
        out['clases_altura'] = g[['clase', 'n', 'area_m2', 'vol_m3']]
    else:
        out['clases_altura'] = pd.DataFrame(columns=['clase', 'n', 'area_m2', 'vol_m3'])
    return out


# ------------------------------------------------------------------- multitemporal
def emparejar_epocas(lab_a, lab_b, tab_a, tab_b, chm_a, chm_b, res, min_frac=0.25, perdida_mortalidad=0.6,
                     radio_m=1.2, tol_dh=0.10):
    """Empareja individuos de dos épocas (algoritmo húngaro).

    Candidato = ápices a ≤ `radio_m` (protocolo ICTS: cKDTree, 1,2 m) **o** copas que solapan ≥ `min_frac`.
    Coste = distancia entre ápices. Devuelve (df_a, df_b, resumen):
    df_a: destino de cada individuo de A (persiste / mortalidad probable / sin correspondencia),
    con `crecimiento` (creció / estable / perdió altura) en los que persisten; df_b: los nuevos de B.
    """
    na, nb = len(tab_a), len(tab_b)
    sel = (lab_a > 0) & (lab_b > 0)
    ov = np.zeros((na + 1, nb + 1))
    if sel.any():
        key = lab_a[sel].astype(np.int64) * (nb + 1) + lab_b[sel]
        u, c = np.unique(key, return_counts=True)
        ov[u // (nb + 1), u % (nb + 1)] = c
    ov = ov[1:, 1:]
    area_a = np.bincount(lab_a.ravel(), minlength=na + 1)[1:].astype(float)
    area_b = np.bincount(lab_b.ravel(), minlength=nb + 1)[1:].astype(float)
    frac = ov / np.maximum(np.minimum.outer(area_a, area_b), 1)
    id_b = np.full(na, -1)
    if na and nb:
        dist = np.hypot(tab_a['x'].values[:, None] - tab_b['x'].values[None, :],
                        tab_a['y'].values[:, None] - tab_b['y'].values[None, :])
        cand = (dist <= radio_m) | (frac >= min_frac)
        cost = np.where(cand, dist, 1e6)
        r, c = linear_sum_assignment(cost)
        for i, j in zip(r, c):
            if cost[i, j] < 1e6:
                id_b[i] = j
    multi_a = ((ov / np.maximum(area_b[None, :], 1)) >= min_frac).sum(axis=1) if nb else np.zeros(na, int)
    multi_b = ((ov / np.maximum(area_a[:, None], 1)) >= min_frac).sum(axis=0) if na else np.zeros(nb, int)

    # CHM de B sobre la copa de A (pérdida de copa)
    hm_b = np.zeros(na)
    if na:
        s = np.bincount(lab_a.ravel(), weights=np.nan_to_num(chm_b).ravel(), minlength=na + 1)[1:]
        hm_b = s / np.maximum(area_a, 1)
    ha = tab_a['h_media_m'].values if na else np.array([])
    perdida = np.where(ha > 0, 1 - hm_b / np.maximum(ha, 1e-9), 0.0)
    esp = lambda t, k: t.iloc[k]['especie'] if 'especie' in t else ''

    filas = []
    for i in range(na):
        a = tab_a.iloc[i]
        j = id_b[i]
        if j >= 0:
            b = tab_b.iloc[j]
            dh = b['h_m'] - a['h_m']
            filas.append(dict(id_a=int(a['id']), id_b=int(b['id']), estado="persiste", x=a['x'], y=a['y'],
                              especie_a=esp(tab_a, i), especie_b=esp(tab_b, j),
                              h_a=a['h_m'], h_b=b['h_m'], dh=dh,
                              crecimiento="creció" if dh >= tol_dh else ("perdió altura" if dh <= -tol_dh else "estable"),
                              vol_a=a['vol_m3'], vol_b=b['vol_m3'], dvol=b['vol_m3'] - a['vol_m3'],
                              area_a=a['area_m2'], area_b=b['area_m2'],
                              revisar=bool(multi_a[i] > 1 or multi_b[j] > 1)))
        else:
            estado = "mortalidad probable" if perdida[i] >= perdida_mortalidad else "sin correspondencia (revisar)"
            filas.append(dict(id_a=int(a['id']), id_b=-1, estado=estado, x=a['x'], y=a['y'],
                              especie_a=esp(tab_a, i), especie_b="",
                              h_a=a['h_m'], h_b=np.nan, dh=np.nan, crecimiento="", vol_a=a['vol_m3'], vol_b=np.nan,
                              dvol=np.nan, area_a=a['area_m2'], area_b=np.nan, revisar=estado.startswith("sin")))
    df_a = pd.DataFrame(filas)
    usados = set(int(j) for j in id_b if j >= 0)
    nuevos = []
    for j in range(nb):
        if j in usados:
            continue
        b = tab_b.iloc[j]
        nuevos.append(dict(id_b=int(b['id']), estado="nuevo (reclutamiento)", x=b['x'], y=b['y'],
                           especie_b=esp(tab_b, j), h_b=b['h_m'], vol_b=b['vol_m3'], area_b=b['area_m2'],
                           revisar=bool(multi_b[j] > 1)))
    df_b = pd.DataFrame(nuevos)
    pers = df_a[df_a['estado'] == 'persiste'] if len(df_a) else df_a
    cnt = lambda col, v: int((pers[col] == v).sum()) if len(pers) else 0
    resumen = dict(
        n_a=na, n_b=nb, n_persisten=int(len(pers)),
        n_mortalidad=int((df_a['estado'] == 'mortalidad probable').sum()) if len(df_a) else 0,
        n_sin_corr=int(df_a['estado'].str.startswith('sin').sum()) if len(df_a) else 0,
        n_nuevos=int(len(df_b)),
        n_crecieron=cnt('crecimiento', 'creció'), n_estables=cnt('crecimiento', 'estable'),
        n_perdieron=cnt('crecimiento', 'perdió altura'),
        mortalidad_pct=100 * (df_a['estado'] == 'mortalidad probable').sum() / na if na else np.nan,
        reclutamiento_pct=100 * len(df_b) / na if na else np.nan,
        dh_medio=float(pers['dh'].mean()) if len(pers) else np.nan,
        dh_mediana=float(pers['dh'].median()) if len(pers) else np.nan,
        dvol_total=float(pers['dvol'].sum()) if len(pers) else np.nan,
        radio_m=radio_m,
        n_revisar=int(df_a['revisar'].sum() + (df_b['revisar'].sum() if len(df_b) else 0)) if na else 0)
    for sp in ("Sabina", "Pino"):
        if na and 'especie_a' in df_a:
            da = df_a[df_a['especie_a'] == sp]
            resumen[f'{sp.lower()}_a'] = int(len(da))
            resumen[f'{sp.lower()}_persisten'] = int((da['estado'] == 'persiste').sum())
            resumen[f'{sp.lower()}_mortalidad'] = int((da['estado'] == 'mortalidad probable').sum())
            resumen[f'{sp.lower()}_sin_corr'] = int(da['estado'].str.startswith('sin').sum())
            resumen[f'{sp.lower()}_nuevos'] = int((df_b['especie_b'] == sp).sum()) if len(df_b) else 0
            resumen[f'{sp.lower()}_dh'] = float(pers[pers['especie_a'] == sp]['dh'].mean()) if len(pers) else np.nan
    return df_a, df_b, resumen
