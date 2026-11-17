"""
exportar.py — Paquete de resultados para QGIS (ráster, vectores, tablas, informe, estilos).

Estructura:  RASTER/  VECTORIALES/  TABLAS/  INFORME/  QGIS/
Todo recortado a la parcela (fuera = NoData / no se exporta).
"""
import io
import os
import zipfile

import numpy as np
import pandas as pd

import figuras as fg
import vectores as vc


# ---------------------------------------------------------------- ráster
def guardar_raster(ruta_sin_ext, arr, malla, res, roi, epsg, nodata=-9999.0):
    """GeoTIFF (rasterio) o, si no está instalado, ASCII grid + .prj. Devuelve la ruta escrita."""
    img, xs, ys = roi.recortar(arr.astype(float), malla, res)
    out = np.where(np.isfinite(img), img, nodata).astype(np.float32)
    x0, ytop = float(xs[0]), float(ys[0])
    try:
        import rasterio
        from rasterio.transform import from_origin
        ruta = ruta_sin_ext + ".tif"
        with rasterio.open(ruta, "w", driver="GTiff", height=out.shape[0], width=out.shape[1], count=1,
                           dtype="float32", crs=f"EPSG:{int(epsg)}", transform=from_origin(x0, ytop, res, res),
                           nodata=nodata, compress="deflate") as dst:
            dst.write(out, 1)
        return ruta
    except ImportError:
        ruta = ruta_sin_ext + ".asc"
        with open(ruta, "w") as f:
            f.write(f"ncols {out.shape[1]}\nnrows {out.shape[0]}\nxllcorner {x0}\nyllcorner {ytop - out.shape[0] * res}\n"
                    f"cellsize {res}\nNODATA_value {nodata}\n")
            np.savetxt(f, out, fmt="%.3f")
        open(ruta_sin_ext + ".prj", "w").write(f"EPSG:{int(epsg)}\n")
        return ruta


# ---------------------------------------------------------------- estilo QGIS
def qml_capas(bordes, extra_nodata=True):
    """Estilo .qml de pseudocolor discreto (una clase por banda de altura) para el CHM."""
    nb = len(bordes) - 1
    cols = fg.colores_capas(nb)
    items = ""
    for b in range(nb):
        items += (f'<item alpha="255" value="{bordes[b + 1]}" label="{bordes[b]:g} – {bordes[b + 1]:g} m" color="{cols[b]}"/>')
    return ('<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM"><qgis version="3.28" styleCategories="Symbology">'
            '<pipe><rasterrenderer type="singlebandpseudocolor" band="1" alphaBand="-1" opacity="1" '
            f'classificationMin="{bordes[0]}" classificationMax="{bordes[-1]}"><rastershader>'
            f'<colorrampshader colorRampType="DISCRETE" classificationMode="2" clip="0">{items}</colorrampshader>'
            '</rastershader></rasterrenderer></pipe></qgis>')


def qml_dh(bordes):
    """Estilo para ΔH (rojo = pérdida, azul = ganancia)."""
    n = len(bordes) - 1
    import matplotlib
    cm = matplotlib.colormaps["RdBu"]
    items = ""
    for b in range(n):
        c = matplotlib.colors.to_hex(cm((b + 0.5) / n))
        items += f'<item alpha="255" value="{bordes[b + 1]}" label="{bordes[b]:g} – {bordes[b + 1]:g} m" color="{c}"/>'
    return ('<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM"><qgis version="3.28"><pipe>'
            f'<rasterrenderer type="singlebandpseudocolor" band="1" classificationMin="{bordes[0]}" classificationMax="{bordes[-1]}">'
            f'<rastershader><colorrampshader colorRampType="DISCRETE" clip="0">{items}</colorrampshader></rastershader>'
            '</rasterrenderer></pipe></qgis>')


# ---------------------------------------------------------------- entidades de estructuras
def entidades_estructuras(est, tabla):
    pts, huellas = [], []
    for _, f in tabla.iterrows():
        props = dict(id=int(f['id']), tipo=str(f['tipo']), h_m=round(float(f['h_m']), 3), longitud=round(float(f['longitud_m']), 2),
                     confianza=str(f['confianza']), excluir=bool(f['excluir']), notas=str(f['notas'])[:200])
        pts.append(dict(tipo="Point", coords=(f['x'], f['y']), props=props))
        p = est.huellas.get(int(f['id']))
        if p is not None and len(p) > 2:
            x0, x1, y0, y1 = p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()
            huellas.append(dict(tipo="Polygon", coords=[[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]], props=props))
    return pts, huellas


def entidades_roi(roi):
    return [dict(tipo="Polygon", coords=[np.asarray(a) for a in roi.contornos()],
                 props=dict(nombre=roi.nombre, area_ha=round(roi.area_ha(), 4)))]


# ---------------------------------------------------------------- paquete completo
def exportar_todo(carpeta, modelos, roi, arb, estr, tablas_est=None, comp=None, epsg=25829,
                  formato_vector="geojson", html=None, et_raster=None, campo=None):
    """Escribe el paquete en `carpeta` y devuelve la lista de rutas relativas creadas."""
    tablas_est = tablas_est or {}
    m, res = modelos['malla_chm'], modelos['res_chm']
    creados = []
    sub = {k: os.path.join(carpeta, k) for k in ("RASTER", "VECTORIALES", "TABLAS", "INFORME", "QGIS")}
    for d in sub.values():
        os.makedirs(d, exist_ok=True)

    def reg(ruta):
        if ruta:
            creados.append(os.path.relpath(ruta, carpeta))

    ext = formato_vector.lower()
    for et, A in arb.items():
        ep = modelos['epocas'][et]
        reg(guardar_raster(os.path.join(sub['RASTER'], f"DEM_{et}"), ep['dem'], modelos['malla_dem'], modelos['res_dem'], roi, epsg))
        reg(guardar_raster(os.path.join(sub['RASTER'], f"Hillshade_{et}"), ep['hs'], modelos['malla_dem'], modelos['res_dem'], roi, epsg))
        reg(guardar_raster(os.path.join(sub['RASTER'], f"DSM_{et}"), ep['dsm'], m, res, roi, epsg))
        chm_ruta = guardar_raster(os.path.join(sub['RASTER'], f"CHM_{et}"), A['chm_eff'], m, res, roi, epsg)
        reg(chm_ruta)
        reg(guardar_raster(os.path.join(sub['RASTER'], f"Copas_ID_{et}"), A['labels'].astype(float), m, res, roi, epsg))
        cls = np.where(A['cap']['mascara'], A['cap']['clase'] + 1, 0).astype(float)
        reg(guardar_raster(os.path.join(sub['RASTER'], f"Capas_altura_{et}"), cls, m, res, roi, epsg, nodata=0))
        estilo = os.path.join(sub['QGIS'], f"CHM_{et}.qml")
        open(estilo, "w", encoding="utf-8").write(qml_capas(A['cap']['bordes']))
        reg(estilo)

        v = A['vec']
        for nombre, lista in (("Arboles_apices", v['apices']), ("Copas", v['copas']),
                              ("Copas_lineas", v.get('lineas_copa') or []),
                              ("Capas_altura_arbol", v['capas_arbol']), ("Capas_altura", v['capas_global'])):
            if lista:
                reg(vc.guardar_vector(os.path.join(sub['VECTORIALES'], f"{nombre}_{et}.{ext}"), lista, epsg, f"{nombre}_{et}"))
        if estr.get(et) is not None and not estr[et].vacio():
            tabla = tablas_est.get(et, estr[et].tabla)
            pts, hue = entidades_estructuras(estr[et], tabla)
            reg(vc.guardar_vector(os.path.join(sub['VECTORIALES'], f"Estructuras_{et}.{ext}"), pts, epsg, "Estructuras"))
            if hue:
                reg(vc.guardar_vector(os.path.join(sub['VECTORIALES'], f"Estructuras_huella_{et}.{ext}"), hue, epsg, "Huellas"))
            ruta = os.path.join(sub['TABLAS'], f"Estructuras_{et}.csv")
            tabla.to_csv(ruta, index=False, encoding="utf-8-sig"); reg(ruta)
        for nombre, df in (("Arboles", A['tabla']), ("Volumen_por_capa_arbol", A['vol_capas_arbol']),
                           ("Volumen_por_capa_global", A['cap']['global_']), ("Clases_altura", A['rodal']['clases_altura']),
                           ("Descartes", A['descartes'])):
            if df is not None and len(df):
                ruta = os.path.join(sub['TABLAS'], f"{nombre}_{et}.csv")
                df.to_csv(ruta, index=False, encoding="utf-8-sig"); reg(ruta)
        r = {k: v for k, v in A['rodal'].items() if not isinstance(v, pd.DataFrame)}
        ruta = os.path.join(sub['TABLAS'], f"Resumen_parcela_{et}.csv")
        pd.DataFrame([r]).to_csv(ruta, index=False, encoding="utf-8-sig"); reg(ruta)
        ruta = os.path.join(sub['TABLAS'], f"Parametros_{et}.csv")
        pd.DataFrame(sorted(A['params'].items()), columns=['parametro', 'valor']).to_csv(ruta, index=False, encoding="utf-8-sig"); reg(ruta)

    reg(vc.guardar_vector(os.path.join(sub['VECTORIALES'], f"Area_Interes.{ext}"), entidades_roi(roi), epsg, "Area_Interes"))

    if comp is not None:
        sfx = f"{comp['a']}_{comp['b']}"
        reg(guardar_raster(os.path.join(sub['RASTER'], f"Cambio_dH_{sfx}"), comp['dh'], m, res, roi, epsg))
        q = os.path.join(sub['QGIS'], f"Cambio_dH_{sfx}.qml")
        from pipeline import DH_BORDES
        open(q, "w", encoding="utf-8").write(qml_dh(list(DH_BORDES))); reg(q)
        for nombre, df in (("Cambios_individuos_A", comp['df_a']), ("Nuevos_individuos_B", comp['df_b']),
                           ("Cambios_clases_dH", comp['tabla_dh']), ("Cambios_resumen", pd.DataFrame([comp['resumen']]))):
            ruta = os.path.join(sub['TABLAS'], f"{nombre}_{sfx}.csv")
            df.to_csv(ruta, index=False, encoding="utf-8-sig"); reg(ruta)
    if campo:
        ruta = os.path.join(sub['TABLAS'], f"Comparacion_campo_{campo.get('anio', '')}.csv")
        campo['df'].to_csv(ruta, index=False, encoding="utf-8-sig"); reg(ruta)
    if html:
        ruta = os.path.join(sub['INFORME'], "Informe_ForestMap.html")
        open(ruta, "w", encoding="utf-8").write(html); reg(ruta)
    return creados


def comprimir(carpeta):
    """ZIP en memoria (para st.download_button)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for raiz, _, fs in os.walk(carpeta):
            for f in fs:
                p = os.path.join(raiz, f)
                z.write(p, os.path.relpath(p, carpeta))
    return buf.getvalue()
