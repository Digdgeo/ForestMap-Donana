"""
pipeline.py — Orquesta el procesado completo (sin interfaz): se usa desde la app y desde las pruebas.

    datos     = {etiqueta: {'xyz', 'suelo'}}                 (paso 2)
    modelos   = generar_modelos(datos, ...)                    (paso 3)
    est       = detectar_estructuras_epoca(...)                (paso 4)
    resultado = analizar_arboles(...)                          (paso 5)
    comp      = comparar_epocas(...)                           (paso 6)
"""
import numpy as np
import pandas as pd

import copas as cp
import estructuras as es
import morfologia as mo
import procesado as pr


# ----------------------------------------------------------------- paso 2
def preparar_epoca(xyz, rigidez=3, umbral=0.5):
    return dict(xyz=xyz, suelo=pr.clasificar_suelo(xyz, rigidez, umbral))


# ----------------------------------------------------------------- paso 3
DH_BORDES = np.round(np.arange(-1.6, 1.81, 0.2), 2)          # igual que la leyenda de cambios en QGIS


def generar_modelos(datos, res_dem=1.0, res_chm=0.5, eq=1.0, pit=0.025, f_min=3, f_med=15):
    xyzs = [d['xyz'] for d in datos.values()]
    m_dem = pr.malla_comun(xyzs, res_dem)
    m_chm = pr.malla_comun(xyzs, res_chm)
    out = dict(malla_dem=m_dem, malla_chm=m_chm, res_dem=res_dem, res_chm=res_chm, eq=eq, pit=pit,
               f_min=f_min, f_med=f_med, epocas={}, dh={})
    for et, d in datos.items():
        xyz, suelo = d['xyz'], d['suelo']
        dem = pr.generar_dem(xyz[suelo], m_dem, res_dem, f_min, f_med)
        dem_f = pr.generar_dem(xyz[suelo], m_chm, res_chm, f_min, f_med)
        dsm = pr.generar_dsm(xyz, m_chm, res_chm)
        chm = pr.generar_chm(xyz, dem_f, m_chm, res_chm, pit, dsm=dsm)
        hag = pr.altura_sobre_suelo(xyz, dem_f, m_chm, res_chm).astype(np.float32)
        out['epocas'][et] = dict(dem=dem, dem_chm=dem_f, dsm=dsm, chm=chm,
                                 hs=pr.hillshade(dem, res_dem), hag=hag)
    et = list(datos.keys())
    for a, b in zip(et[:-1], et[1:]):
        out['dh'][(a, b)] = out['epocas'][b]['chm'] - out['epocas'][a]['chm']
    return out


# ----------------------------------------------------------------- paso 4
def detectar_estructuras_epoca(modelos, datos, et, params=None, region=None, buscar_componentes=True):
    d = datos[et]
    ns = ~d['suelo']
    hag = modelos['epocas'][et]['hag']
    xyz = d['xyz']
    return es.detectar_estructuras(xyz[ns, 0], xyz[ns, 1], xyz[ns, 2], hag[ns].astype(float),
                                   params, region, buscar_componentes)


def mascara_estructuras(modelos, est, tabla=None):
    if est is None or est.vacio():
        return None
    forma = modelos['epocas'][next(iter(modelos['epocas']))]['chm'].shape
    return es.mascara_raster(est, modelos['malla_chm'], modelos['res_chm'], forma, tabla)


# ----------------------------------------------------------------- paso 5
def analizar_arboles(modelos, datos, et, params, roi, est=None, tabla_est=None, voxel=True,
                     morfologia=True):
    """Individuos, copas, capas de altura, volúmenes y vectores de una época."""
    malla, res = modelos['malla_chm'], modelos['res_chm']
    chm = np.nan_to_num(modelos['epocas'][et]['chm'])
    roi_m = roi.mascara(malla, res, chm.shape)
    excl = mascara_estructuras(modelos, est, tabla_est) if est is not None else None
    chm_eff = np.where(excl, 0.0, chm) if excl is not None else chm

    r = cp.detectar_copas(chm, malla, res, params, excluir=excl, roi=roi_m)
    labels, tabla = r['labels'], r['tabla'].copy()

    cobertura = roi_m & (~excl if excl is not None else True)
    cap = cp.capas_altura(chm_eff, labels, res, params.ancho_capa_m, params.hcorte_m, mascara=cobertura)

    if voxel and len(tabla):
        d = datos[et]
        ns = ~d['suelo']
        hag = modelos['epocas'][et]['hag'][ns].astype(float)
        vv = cp.volumen_ocupado(d['xyz'][ns, 0], d['xyz'][ns, 1], hag, labels, malla, res)
        tabla['vol_voxel_m3'] = vv[tabla['id'].values - 1]
    vec = cp.construir_vectores(malla, res, labels, chm_eff, tabla, cap, simplificar=res * 0.35)
    if morfologia and len(tabla):
        df_m, vec['lineas_copa'] = mo.metricas(vec['copas'], con_lineas=True)
        mo.anotar(vec['copas'], df_m)
        tabla = tabla.merge(df_m, on='id', how='left')
    rodal = cp.resumen_rodal(tabla, chm_eff, labels, roi_m, res, params.hcorte_m, excl, cap)

    # matriz de volumen por capa (formato ancho) para el CSV
    vc_df = pd.DataFrame(cap['vol_capa'], columns=[
        f"vol_{cap['bordes'][b]:.2f}-{cap['bordes'][b + 1]:.2f}m" for b in range(len(cap['bordes']) - 1)])
    vc_df.insert(0, 'id', np.arange(1, len(vc_df) + 1))
    return dict(et=et, params=params.dict(), labels=labels, tabla=tabla, descartes=r['descartes'],
                chm_eff=chm_eff, excl=excl, roi_mask=roi_m, cap=cap, vec=vec, rodal=rodal,
                vol_capas_arbol=vc_df, n_forzados=r['n_forzados'], roi_desc=roi.descripcion(),
                n_estructuras_excluidas=0 if est is None else int(
                    (est.tabla['excluir'] if tabla_est is None else tabla_est['excluir']).sum()))


# ----------------------------------------------------------------- paso 6
def comparar_epocas(modelos, arboles, a, b, roi, radio_m=1.2):
    malla, res = modelos['malla_chm'], modelos['res_chm']
    A, B = arboles[a], arboles[b]
    df_a, df_b, resumen = cp.emparejar_epocas(
        A['labels'], B['labels'], A['tabla'], B['tabla'], A['chm_eff'], B['chm_eff'], res, radio_m=radio_m)
    roi_m = roi.mascara(malla, res, A['chm_eff'].shape)
    excl = np.zeros(roi_m.shape, bool)
    for X in (A, B):
        if X['excl'] is not None:
            excl |= X['excl']
    valido = roi_m & ~excl
    dh = np.where(valido, B['chm_eff'] - A['chm_eff'], np.nan)
    cob_a = valido & (A['chm_eff'] >= A['params']['hcorte_m'])
    cob_b = valido & (B['chm_eff'] >= B['params']['hcorte_m'])
    sel = valido & (cob_a | cob_b)
    cls = pd.cut(pd.Series(dh[sel]), np.r_[-np.inf, DH_BORDES, np.inf])
    tabla_dh = cls.value_counts().sort_index().rename('px').reset_index()
    tabla_dh.columns = ['intervalo', 'px']
    tabla_dh['area_m2'] = tabla_dh['px'] * res * res
    tabla_dh['intervalo'] = tabla_dh['intervalo'].astype(str)
    resumen.update(
        cobertura_a_pct=100 * cob_a.sum() / max(valido.sum(), 1), cobertura_b_pct=100 * cob_b.sum() / max(valido.sum(), 1),
        dh_medio_raster=float(np.nanmean(dh[sel])) if sel.any() else np.nan,
        perdida_ha=float(((dh < -1.0) & sel).sum() * res * res / 1e4),
        ganancia_ha=float(((dh > 1.0) & sel).sum() * res * res / 1e4),
        vol_a=float(A['tabla']['vol_m3'].sum()), vol_b=float(B['tabla']['vol_m3'].sum()))
    return dict(a=a, b=b, df_a=df_a, df_b=df_b, resumen=resumen, dh=dh, tabla_dh=tabla_dh)
