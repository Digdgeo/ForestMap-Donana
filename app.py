"""
ForestMap Doñana V3 — LiDAR para el sabinar: modelos digitales, capas de altura de copa, individuos,
volúmenes, antena/vallado/sensores y cambios entre épocas.   Ejecutar:  streamlit run app.py
"""
import os
import re
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

import copas as cp
import exportar as ex
import figuras as fg
import graficos as gr
import informe as inf
import pipeline as pl
import procesado as pr
import roi as rm
import estructuras as es
import campo as cm

st.set_page_config(page_title="ForestMap Doñana V3", page_icon="🌲", layout="wide")
S = st.session_state

PASOS = ["⚙️ 1 · Configuración", "📂 2 · Carga y parcela", "🗺️ 3 · Modelos digitales", "📡 4 · Antena y estructuras",
         "🌲 5 · Árboles y capas", "🔄 6 · Cambios entre épocas", "📄 7 · Exportar e informe"]
for k, v in dict(epocas=[], epsg=25829, datos={}, roi=None, modelos=None, estr={}, tab_est={}, arb={}, comp=None,
                 sintetico=False, paquete=None, html=None, autor="", titulo="Sabinar de Doñana", campo_inv=None, campo=None, campo_anio=2011).items():
    S.setdefault(k, v)



def detectar_anio(nombre):
    """Intenta sacar el año del nombre de archivo: 'ojillo_22_w.las' -> '2022', 'x_2025.laz' -> '2025'."""
    base = os.path.basename(str(nombre))
    m = re.search(r"(?<!\d)(19|20)\d{2}(?!\d)", base)
    if m:
        return m.group(0)
    for t in re.split(r"[^0-9]+", os.path.splitext(base)[0]):
        if len(t) == 2 and 10 <= int(t) <= 49:
            return "20" + t
    return None


def aviso_falta(txt):
    st.info(txt)
    st.stop()


def figura_png(b, **kw):
    if b:
        st.image(b, **kw)


with st.sidebar:
    st.title("🌲 ForestMap Doñana")
    TAM = {"Normal": 100, "Grande": 118, "Muy grande": 136, "Enorme": 156}
    tam = st.select_slider("🔠 Tamaño de letra", list(TAM), value="Grande", key="tam_letra",
                           help="Agranda textos, iconos y ayudas (?). Útil para proyectar en una pantalla o una reunión.")
    z = TAM[tam]
    st.markdown(
        f"""<style>
        html {{ font-size: {z}%; }}
        [data-testid="stTooltipIcon"] svg {{ width: 1.35rem; height: 1.35rem; }}
        [data-testid="stTooltipIcon"] {{ margin-left: .35rem; }}
        [data-testid="stWidgetLabel"] p {{ font-size: 1.05rem; font-weight: 600; }}
        [data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] {{ font-size: 1rem; }}
        [data-testid="stTabs"] button p {{ font-size: 1.1rem; }}
        [data-testid="stMetricLabel"] p {{ font-size: 1rem; }}
        [data-testid="stExpander"] summary p {{ font-size: 1.05rem; }}
        </style>""", unsafe_allow_html=True)
    paso = st.radio("Flujo de trabajo", PASOS, key="paso")
    st.divider()
    st.caption("Estado")
    st.write(f"{'✅' if S.datos else '⬜'} Nubes: {', '.join(S.datos) or '—'}")
    st.write(f"{'✅' if S.roi else '⬜'} Parcela: {S.roi.descripcion() if S.roi else '—'}")
    st.write(f"{'✅' if S.modelos else '⬜'} Modelos digitales")
    st.write(f"{'✅' if S.estr else '⬜'} Estructuras: {', '.join(S.estr) or '—'}")
    st.write(f"{'✅' if S.arb else '⬜'} Árboles: {', '.join(S.arb) or '—'}")
    if S.sintetico:
        st.warning("Datos SINTÉTICOS de demostración")
    with st.expander("📖 Glosario (palabras clave)"):
        st.markdown(
            "- **LiDAR / LAS**: el escáner láser del avión o dron. El archivo LAS es una nube de millones de puntos con su posición (X, Y) y su altura (Z).\n"
            "- **Época**: cada vuelo/año. Para ver cambios hacen falta dos.\n"
            "- **Suelo**: los puntos que tocan el terreno. Sirven para saber a qué altura está el suelo.\n"
            "- **CSF**: el programa que separa suelo y vegetación (imagina una tela que cae sobre los puntos).\n"
            "- **DEM**: mapa del terreno sin vegetación (el suelo).\n"
            "- **DSM**: mapa de la superficie, con la copa de los árboles incluida.\n"
            "- **CHM**: DSM − DEM = **altura de la vegetación** sobre el suelo. Es el mapa clave.\n"
            "- **Ápice**: la punta más alta de un árbol; cada ápice = un árbol.\n"
            "- **Copa**: la parte de arriba del árbol vista desde el aire.\n"
            "- **Parcela / ROI**: la zona de estudio. Todo lo de fuera se descarta.\n"
            "- **Capas**: cortes horizontales cada 0,5 m de altura para medir cuánta copa hay a cada nivel.\n"
            "- **EPSG 25829**: el sistema de coordenadas de Doñana (UTM 29N).")


# ===================================================================== 1
if paso == PASOS[0]:
    st.header("⚙️ 1 · Configuración")
    with st.expander("ℹ️ ¿Qué se hace en este paso?", expanded=False):
        st.write("Datos generales del trabajo: el sistema de coordenadas y el título/autor que saldrán en el informe. Normalmente no hay que tocar nada, salvo el título.")
    st.info("Las épocas (años) se definen en el paso 2: cada archivo LAS lleva su año, que se detecta del nombre del archivo y se puede corregir.")
    S.epsg = st.number_input("🧭 EPSG del sistema de coordenadas", 1000, 99999, int(S.epsg),
                             help="Es el «idioma» de las coordenadas de tus archivos. Para Doñana es 25829 (ETRS89 / UTM 29N). No lo cambies salvo que tus datos vengan en otro sistema.")
    S.titulo = st.text_input("📝 Título del informe", S.titulo, help="Aparece en la portada del informe HTML.")
    S.autor = st.text_input("👤 Autor/a", S.autor, help="Tu nombre o el del laboratorio; sale en la portada del informe.")
    st.success(f"Épocas cargadas: {', '.join(S.datos) or 'ninguna todavía'}. Con dos o más épocas se activa el análisis de cambios (paso 6).")

# ===================================================================== 2
elif paso == PASOS[1]:
    st.header("📂 2 · Carga de nubes de puntos y parcela")
    with st.expander("ℹ️ ¿Qué se hace en este paso?", expanded=False):
        st.write("1) Indica el archivo LAS de cada año (con su ruta, es lo más rápido). 2) Pulsa «Cargar y clasificar suelo»: la app lee los puntos y separa el suelo de la vegetación. 3) Carga la parcela (.gpkg) para quedarte solo con la zona de estudio. 4) Opcional: el inventario de campo para validar el recuento.")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("☁️ Nubes LAS/LAZ")
        st.caption("Cada nube lleva su año (se detecta del nombre del archivo y se puede corregir). Archivos de más de 2 GB: use la ruta en disco.")
        if S.get("pend_lab"):
            for i_, l_ in enumerate(S.pend_lab):
                S[f"lab_{i_}"] = l_
            S["nep"] = len(S.pend_lab)
            S.pend_lab = None
        S.setdefault("nep", max(2, len(S.epocas)))
        n_ep = int(st.number_input("🗓️ Número de épocas (años) a cargar", 1, 6, key="nep", help="Cuántos años vas a cargar. Con 1 solo se analiza ese año; con 2 (o más) se activa la comparación de cambios entre años."))
        subidos, etiquetas = [], []
        for i in range(n_ep):
            S.setdefault(f"lab_{i}", S.epocas[i] if i < len(S.epocas) else "")
            st.markdown(f"**Nube {i + 1}**")
            f_ = st.file_uploader(f"📁 Archivo LAS/LAZ {i + 1}", type=["las", "laz"], key=f"up_{i}",
                                  help="Arrastra aquí el archivo. Con archivos grandes (varios cientos de MB) es mejor usar la ruta de abajo.")
            ruta_ = st.text_input(f"…o ruta del archivo en este ordenador ({i + 1})", key=f"ruta_{i}",
                                  placeholder="/home/usuario/datos/ojillo_22_w.las",
                                  help="Pega la ruta completa del archivo en este ordenador (ej. /home/rauladmin/Descargas/ojillo_22_w.las). Es más rápido que subirlo y gasta menos memoria.")
            rr = os.path.expanduser(ruta_.strip()) if ruta_.strip() else ""
            fuente = f_ if f_ is not None else (rr if rr and os.path.isfile(rr) else None)
            if ruta_.strip() and fuente is None:
                st.error(f"No existe el archivo: {ruta_}")
            nombre = f_.name if f_ is not None else (rr if fuente else "")
            if nombre and S.get(f"auto_{i}") != nombre:      # archivo nuevo en esta ranura: proponer su año
                S[f"auto_{i}"] = nombre
                a_ = detectar_anio(nombre)
                if a_ and S.get(f"lab_{i}") != a_:
                    S[f"lab_{i}"] = a_
                    st.rerun()
            lab = st.text_input(f"🏷️ Año / etiqueta de la nube {i + 1}", key=f"lab_{i}",
                                help="El año de este vuelo (se rellena solo desde el nombre del archivo). Es el nombre con el que aparecerá en tablas e informe.")
            subidos.append(fuente)
            etiquetas.append(lab.strip())
        ok_et = all(etiquetas) and len(set(etiquetas)) == len(etiquetas)
        if not ok_et:
            st.warning("Cada nube necesita una etiqueta distinta (por ejemplo, el año).")
        dens = st.slider("🔹 Densidad máxima (puntos/m²)", 50, 5000, 1000, 50,
                         help="Si el archivo tiene demasiados puntos, se aligera hasta este máximo por m². Más bajo = más rápido y menos memoria; más alto = más detalle pero más lento. 1000 es un buen punto de partida y no cambia las alturas.")
        rig = st.select_slider("🧵 Rigidez del CSF (cómo se «tensa» el suelo)", [1, 2, 3], 3,
                               help="Controla cómo de flexible es la «tela» que se apoya sobre los puntos para encontrar el suelo. 3 = rígida: para terreno llano (Doñana). 1 = muy flexible: para laderas o relieve accidentado. Si ves que el suelo sale demasiado ondulado, sube; si se come el terreno real, baja.")
        modo_suelo = st.radio("🟫 Suelo", ["Clasificar (CSF / morfológico)", "Usar la clase 2 del archivo (suelo ya clasificado)"], horizontal=True,
                              help="«Clasificar»: la app calcula el suelo con el mismo método en todos los años (recomendado para comparar años). «Usar la clase 2»: se fía del suelo que ya trae el archivo; solo si todos los años se clasificaron igual.")
        if st.button("Cargar y clasificar suelo", type="primary", disabled=not (all(x is not None for x in subidos) and ok_et)):
            S.datos, S.modelos, S.estr, S.tab_est, S.arb, S.comp, S.sintetico, S.campo = {}, None, {}, {}, {}, None, False, None
            if S.get('roi') is not None and 'sintética' in S.roi.nombre:
                S.roi = None
            pares = list(zip(etiquetas, subidos))
            if all(e_.isdigit() for e_, _ in pares):        # de antigua a reciente
                pares.sort(key=lambda p: int(p[0]))
            S.epocas = [e_ for e_, _ in pares]
            for e, f in pares:
                with st.spinner(f"Leyendo {e}…"):
                    xyz, cl = pr.leer_las(f, dens, con_clase=True)
                    if modo_suelo.startswith("Usar") and (cl == 2).any():
                        S.datos[e] = dict(xyz=xyz, suelo=(cl == 2))
                    else:
                        S.datos[e] = pl.preparar_epoca(xyz, rig)
            st.success("Nubes cargadas.")
        st.subheader("📋 Inventario de campo (opcional)")
        st.caption("Excel con IDENT, LAT, LON, Altura y Cambio… (p. ej. datos_ojillo_11.xlsx). Se usa para validar el recuento.")
        fi = st.file_uploader("📋 Inventario de campo (.xlsx)", type=["xlsx"], key="up_campo",
                              help="El Excel con los árboles medidos a mano (IDENT, LAT, LON, altura…). Sirve para comprobar si el recuento del LiDAR se parece al de campo.")
        S.campo_anio = st.number_input("📅 Año del inventario", 1990, 2100, int(S.campo_anio), help="El año en que se midieron los árboles en campo (para Ojillo: 2011).")
        if fi is not None and st.button("Cargar inventario"):
            try:
                S.campo_inv = cm.leer_inventario(fi)
                S.campo = None
                st.success(f"{len(S.campo_inv)} fichas leídas ({int(S.campo_inv.vivo.sum())} vivas).")
            except Exception as ex_:
                st.error(f"No se pudo leer el inventario: {ex_}")
    if os.environ.get("FORESTMAP_DEMO") == "1":      # solo para desarrollo/pruebas
        with c2:
            st.subheader("Datos de demostración")
            st.caption("Escena sintética (no son datos reales) para probar el flujo completo.")
            if st.button("Cargar escena sintética"):
                import sintetico as sy
                ep2 = S.epocas if len(S.epocas) >= 2 else ["2024", "2026"]
                S.epocas = ep2
                S.pend_lab = list(ep2)
                esc = sy.escenario(epocas=tuple(ep2))
                S.datos = {e: dict(xyz=esc['xyz'][e], suelo=pr.clasificar_suelo_morfologico(esc['xyz'][e])) for e in ep2}
                S.modelos, S.estr, S.tab_est, S.arb, S.comp, S.sintetico = None, {}, {}, {}, None, True
                ext = rm.extension([d['xyz'] for d in S.datos.values()])
                cx, cy = (ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2
                L, th = 99.4 / 2, np.radians(14)
                R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                esq = np.array([[-L, -L], [L, -L], [L, L], [-L, L], [-L, -L]]) @ R.T + [cx, cy]
                S.roi = rm.Roi((esq[:, 0].min(), esq[:, 0].max(), esq[:, 1].min(), esq[:, 1].max()), anillos=[esq], nombre="Parcela (sintética)")
                st.rerun()
    if S.datos:
        st.divider()
        st.subheader("📐 Parcela de estudio (ROI)")
        st.caption("Todo lo que queda fuera de la parcela se descarta en mapas, estadísticas y exportaciones.")
        ext = rm.extension([d['xyz'] for d in S.datos.values()])
        modo = st.radio("✂️ Modo de recorte", ["Polígono (.gpkg / .geojson / .zip)", "Parcela cuadrada", "Coordenadas", "Sin borde", "Toda la nube"], horizontal=True,
                        help="Cómo se define la zona de estudio. Lo normal: «Polígono» con el archivo parcela.gpkg del laboratorio. Las otras opciones sirven si no hay polígono: un cuadrado de X hectáreas, unas coordenadas, la nube quitando un borde, o la nube entera.")
        roi_nueva = None
        if modo.startswith("Polígono"):
            f = st.file_uploader("📐 Archivo de la parcela", type=["gpkg", "geojson", "zip"], key="poly", help="El contorno de la parcela de seguimiento (parcela.gpkg). Todo lo que quede fuera se descarta.")
            if f is not None:
                try:
                    pol = rm.leer_poligono(f, f.name, int(S.epsg))
                    roi_nueva = rm.construir_roi(ext, "poligono", poligono=pol)
                except Exception as ex_:
                    st.error(f"No se pudo leer el polígono: {ex_}")
        elif modo == "Parcela cuadrada":
            ha = st.number_input("Hectáreas", 0.1, 50.0, 1.0, 0.1, help="Superficie del cuadrado, colocado en el centro de la nube.")
            dx = st.number_input("Desplazamiento X (m)", -500.0, 500.0, 0.0, help="Mueve el cuadrado hacia el este (+) o el oeste (−).")
            dy = st.number_input("Desplazamiento Y (m)", -500.0, 500.0, 0.0, help="Mueve el cuadrado hacia el norte (+) o el sur (−).")
            roi_nueva = rm.construir_roi(ext, "parcela", hectareas=ha, dx=dx, dy=dy)
        elif modo == "Coordenadas":
            cc = st.columns(4)
            co = [cc[i].number_input(n, value=float(v), help="Límite de la parcela en coordenadas UTM (metros): oeste, este, sur y norte, en este orden.") for i, (n, v) in enumerate(zip(["X mín (oeste)", "X máx (este)", "Y mín (sur)", "Y máx (norte)"], ext))]
            roi_nueva = rm.construir_roi(ext, "coords", coords=co)
        elif modo == "Sin borde":
            roi_nueva = rm.construir_roi(ext, "borde", borde=st.number_input("Borde (m)", 0.0, 100.0, 10.0, help="Metros que se recortan en el contorno de la nube; los bordes suelen tener menos puntos y peor calidad."))
        else:
            roi_nueva = rm.construir_roi(ext, "nube")
        if roi_nueva is not None and st.button("Aplicar parcela", type="primary"):
            S.roi = roi_nueva
            S.modelos, S.estr, S.tab_est, S.arb, S.comp = None, {}, {}, {}, None
            st.rerun()
        if S.roi:
            st.success(S.roi.descripcion())
        filas = []
        for e, d in S.datos.items():
            dentro = S.roi.puntos(d['xyz'][:, 0], d['xyz'][:, 1]) if S.roi else np.ones(len(d['xyz']), bool)
            filas.append(dict(época=e, puntos=len(d['xyz']), en_parcela=int(dentro.sum()), suelo_pct=round(100 * d['suelo'].mean(), 1)))
        st.dataframe(pd.DataFrame(filas), hide_index=True)

# ===================================================================== 3
elif paso == PASOS[2]:
    st.header("🗺️ 3 · Modelos digitales (DEM · DSM · CHM)")
    with st.expander("ℹ️ ¿Qué se hace en este paso?", expanded=False):
        st.write("Se convierten los puntos en mapas: el DEM (el terreno), el DSM (la superficie con las copas) y el CHM (la altura de la vegetación = DSM − DEM). Con el CHM se detectan después los árboles. Pulsa «Generar modelos» y espera.")
    if not S.datos or S.roi is None:
        aviso_falta("Cargue las nubes y defina la parcela en el paso 2.")
    preset = st.radio("🔍 Resolución", ["Rápida (DEM 2 m · CHM 1 m)", "Estándar (DEM 1 m · CHM 0,5 m)", "Fina (DEM 1 m · CHM 0,25 m)"], index=1, horizontal=True,
                      help="Tamaño del píxel de los mapas. Más fino = más detalle pero más lento. La «Estándar» (CHM 0,5 m) es la que recomendamos para separar copas; con 1 m se funden copas vecinas y salen menos árboles.")
    rd, rc = {0: (2.0, 1.0), 1: (1.0, 0.5), 2: (1.0, 0.25)}[["R", "E", "F"].index(preset[0])]
    with st.expander("🔧 Parámetros avanzados (normalmente no hace falta tocarlos)"):
        a, b, c, d_ = st.columns(4)
        eq = a.number_input("〰️ Equidistancia curvas (m)", 0.1, 5.0, 1.0, help="Cada cuántos metros de altura se dibuja una curva de nivel en el mapa. Solo afecta al dibujo, no a los árboles.")
        pit = b.number_input("🕳️ Relleno de huecos del CHM (m)", 0.0, 1.0, 0.025, 0.005, format="%.3f", help="Rellena agujeritos del mapa de alturas donde el láser no llegó a la copa. Más alto = mapa más liso pero se pierde detalle fino. Déjalo como está.")
        f_min = c.number_input("⬇️ Filtro mínimo DEM (px)", 1, 15, 3, help="Para el terreno se toma el punto más bajo de una ventanita de este tamaño, para que la vegetación baja no se cuele como si fuera suelo. Más alto = terreno más bajo y más «limpio».")
        f_med = d_.number_input("🌊 Suavizado DEM (px)", 1, 41, 15, help="Alisa el terreno para quitar pequeños picos. Más alto = terreno más suave. Doñana es muy llana, por eso un valor alto funciona bien.")
    if st.button("Generar modelos", type="primary"):
        with st.spinner("Generando DEM, DSM y CHM…"):
            S.modelos = pl.generar_modelos(S.datos, rd, rc, eq, pit, f_min, f_med)
            S.estr, S.tab_est, S.arb, S.comp = {}, {}, {}, None
    mod = S.modelos
    if mod is None:
        st.stop()
    et = st.selectbox("🗓️ Época", list(S.datos), index=len(S.datos) - 1, help="Año que quieres ver en los mapas.")
    ep = mod['epocas'][et]
    t1, t2, t3, t4 = st.tabs(["⛰️ DEM + sombreado", "🌳 DSM", "📏 CHM (alturas)", "✂️ Perfil vertical"])
    with t1:
        st.plotly_chart(gr.raster_plotly(ep['dem'], mod['malla_dem'], mod['res_dem'], S.roi, "DEM", "Earth", "m s.n.m."), use_container_width=True)
        figura_png(fg.mapa_dem_hillshade(ep['dem'], ep['hs'], mod['malla_dem'], mod['res_dem'], S.roi, f"DEM + sombreado — {et}"))
    with t2:
        st.plotly_chart(gr.raster_plotly(ep['dsm'], mod['malla_chm'], mod['res_chm'], S.roi, "DSM", "Cividis", "m"), use_container_width=True)
    with t3:
        st.plotly_chart(gr.raster_plotly(ep['chm'], mod['malla_chm'], mod['res_chm'], S.roi, "CHM", "Viridis", "m"), use_container_width=True)
    with t4:
        b = S.roi.bounds
        c1, c2, c3, c4, c5 = st.columns(5)
        st.caption("Un perfil es un «corte lateral» del bosque: dibuja los puntos en una banda entre dos extremos, como si lo miraras de lado.")
        x0 = c1.number_input("X inicio", value=float(b[0]), help="Coordenada X (este) del punto donde empieza el corte."); y0 = c2.number_input("Y inicio", value=float((b[2] + b[3]) / 2), help="Coordenada Y (norte) del punto donde empieza el corte.")
        x1 = c3.number_input("X fin", value=float(b[1]), help="Coordenada X del final del corte."); y1 = c4.number_input("Y fin", value=float((b[2] + b[3]) / 2), help="Coordenada Y del final del corte.")
        w = c5.number_input("Ancho banda (m)", 0.5, 20.0, 3.0, help="Grosor de la rebanada: se dibujan los puntos a este ancho a cada lado de la línea. Más ancho = más puntos pero más mezclados.")
        d = S.datos[et]
        hay_arb = et in S.arb
        modo_txt = st.radio("🎨 Colorear por", ["Suelo / vegetación", "Cada individuo", "Especie"], horizontal=True,
                            help="«Cada individuo» pinta cada árbol detectado con su color y su número (el mismo de la tabla del paso 5). «Especie» distingue sabina y pino. Necesita haber hecho antes el paso 5 (Árboles y capas) de esta época.")
        if modo_txt != "Suelo / vegetación" and not hay_arb:
            st.info("🌲 Para ver los individuos hay que ejecutar antes el paso 5 · Árboles y capas con esta época. Mientras tanto se muestra el perfil normal.")
        modo = {"Suelo / vegetación": "tipo", "Cada individuo": "individuo", "Especie": "especie"}[modo_txt]
        figura_png(fg.perfil_vertical(d['xyz'], ep['hag'], d['suelo'], (x0, y0), (x1, y1), w, S.estr.get(et),
                                      arb=S.arb.get(et), modelos=mod, modo=modo if hay_arb else "tipo"))

# ===================================================================== 4
elif paso == PASOS[3]:
    st.header("📡 4 · Antena, vallado y sensores")
    with st.expander("ℹ️ ¿Qué se hace en este paso?", expanded=False):
        st.write("En la parcela hay una torre/antena, brazos con sensores y quizá un vallado. El láser los ve como «cosas altas y delgadas» y podrían contarse como árboles. Aquí se detectan para excluirlos del recuento. Puedes marcar o desmarcar «excluir» a mano en la tabla.")
    if S.modelos is None:
        aviso_falta("Genere los modelos en el paso 3.")
    with st.expander("🔧 Parámetros de detección"):
        P = es.ParamsEstructuras()
        c = st.columns(3)
        P.h_semilla_m = c[0].number_input("📡 Altura mínima del mástil (m)", 1.0, 30.0, P.h_semilla_m,
                                              help="Por debajo de esta altura, un objeto delgado no se considera mástil. Bájala si no detecta una antena baja; súbela si marca árboles altos.")
        P.sobresale_min_m = c[1].number_input("⬆️ Debe sobresalir de su entorno (m)", 0.5, 10.0, P.sobresale_min_m,
                                                    help="Cuánto tiene que destacar sobre la vegetación de alrededor para ser un mástil. Un árbol se funde con sus vecinos; una antena sobresale como un palo.")
        P.radio_busqueda_m = c[2].number_input("⭕ Radio de búsqueda alrededor (m)", 5.0, 80.0, P.radio_busqueda_m,
                                                help="Distancia alrededor del mástil donde se miran los brazos y sensores. Más grande = busca más lejos.")
        P.largo_vallado_min_m = c[0].number_input("🚧 Longitud mínima del vallado (m)", 1.0, 50.0, P.largo_vallado_min_m,
                                                    help="Una línea recta de puntos tiene que medir al menos esto para considerarse valla.")
        P.grosor_vegetacion_m = c[1].number_input("📏 Grosor máximo estructura (m)", 0.3, 3.0, P.grosor_vegetacion_m,
                                                  help="Las estructuras son finas; la vegetación es gruesa y «esponjosa». Por encima de este grosor se considera vegetación, no estructura.")
        P.tam_max_instrumento_m = c[2].number_input("📟 Tamaño máx. instrumento (m)", 0.5, 10.0, P.tam_max_instrumento_m,
                                                    help="Tamaño máximo de una caja o sensor suelto. Objetos mayores no se tratan como instrumentos.")
        P.recinto_lado_max_m = c[0].number_input("🟧 Lado máx. del recinto cuadrado (m)", 2.0, 15.0, P.recinto_lado_max_m,
                                                  help="Se busca una valla cuadrada alrededor del mástil de entre 1,5 m y este lado. Súbelo si el recinto es más grande.")
        P.mostrar_dudosos = c[1].checkbox("🔍 Mostrar objetos dudosos", P.mostrar_dudosos,
                                          help="Muestra también los objetos pequeños y sueltos de confianza «baja», que casi siempre son arbustos y no instrumentos. Por defecto se ocultan para no confundir.")
    if st.button("Detectar estructuras", type="primary"):
        S.estr, S.tab_est, S.arb, S.comp = {}, {}, {}, None
        for e in S.datos:
            with st.spinner(f"Estructuras {e}…"):
                S.estr[e] = pl.detectar_estructuras_epoca(S.modelos, S.datos, e, P)
                S.tab_est[e] = S.estr[e].tabla.copy()
    if not S.estr:
        st.stop()
    et = st.selectbox("🗓️ Época", list(S.estr), index=len(S.estr) - 1, help="Año cuyas estructuras quieres revisar.")
    est = S.estr[et]
    if est.vacio():
        st.warning("No se ha detectado ninguna estructura. Pruebe a bajar la altura mínima del mástil.")
        st.stop()
    st.caption("Marque **excluir** en lo que sea infraestructura (no se contará como árbol). Desmárquelo si es un falso positivo.")
    st.caption("✏️ Puedes corregir a mano **excluir**, **h_m** (altura) y **notas**: si conoces la altura real de la antena o de la valla, escríbela. "
               "El láser suele quedarse corto con mástiles muy finos, porque la punta casi no devuelve señal.")
    ed = st.data_editor(S.tab_est[et], hide_index=True, disabled=[c for c in S.tab_est[et].columns if c not in ('excluir', 'h_m', 'notas')],
                        key=f"ed_{et}", use_container_width=True)
    if not ed.equals(S.tab_est[et]):
        S.tab_est[et] = ed
        S.arb, S.comp = {}, None
    ep = S.modelos['epocas'][et]
    figura_png(fg.plano_estructuras(S.datos[et]['xyz'], ep['hag'], S.datos[et]['suelo'], est))
    st.markdown("**🔎 Detalle de la antena y su recinto** (planta ampliada y dos cortes laterales: aquí se ve cuánto sube el mástil y la altura de la valla)")
    figura_png(fg.detalle_antena(S.datos[et]['xyz'], ep['hag'], S.datos[et]['suelo'], est))
    for log in est.log:
        st.caption(log)

# ===================================================================== 5
elif paso == PASOS[4]:
    st.header("🌲 5 · Árboles, capas de altura y volúmenes")
    with st.expander("ℹ️ ¿Qué se hace en este paso?", expanded=False):
        st.write("Se busca la punta (ápice) de cada árbol en el mapa de alturas y se dibuja su copa. Todo lo de más de 5 m se llama pino; el resto, sabina. Por debajo de 1,3 m se considera suelo/matorral y no se cuenta. También se calcula el volumen por capas de 0,5 m. Lo más importante es la «Sensibilidad».")
    if S.modelos is None or S.roi is None:
        aviso_falta("Genere los modelos digitales (paso 3).")
    P = cp.ParamsArboles()
    PRE = {"Conservadora (menos copas)": (0.3, 0.15, 0.3), "Equilibrada": (0.2, 0.05, 0.3),
           "Máxima separación (más copas)": (0.15, 0.03, 0.3), "Censo máximo (≈ 700 en el Ojillo, riesgo de partir copas)": (0.10, 0.015, 0.5)}
    pre = st.radio("🎚️ Sensibilidad (cuántos árboles separa)", list(PRE), index=1, horizontal=True,
                   help="Cuánto de «quisquillosa» es la búsqueda. Conservadora: menos copas, más grandes (puede juntar vecinos). Máxima separación: más copas (puede partir una en dos). «Equilibrada» es la opción defendible. «Censo máximo» se acerca al recuento de campo (~700) pero parte copas: úsala solo para comparar.")
    P.sigma_m, P.prominencia_m, P.area_min_m2 = PRE[pre]
    st.caption("Por debajo de 1,3 m todo se trata como suelo (matojos). Más de 5 m = pino. Lo tapado bajo una copa cuenta como uno.")
    with st.expander("🔧 Parámetros avanzados"):
        a, b, c = st.columns(3)
        P.metodo = a.selectbox("🔎 Método de ápices", ["hmax", "lmf"], format_func=lambda m: "H-máximos (recomendado)" if m == "hmax" else "Ventana variable (lidR)",
                           help="Cómo se encuentran las puntas de los árboles. H-máximos: busca picos que destacan una cierta altura sobre su entorno (recomendado). Ventana variable: método clásico de lidR, mira el punto más alto en una ventana que crece con la altura.")
        P.sigma_m = b.number_input("🌫️ Suavizado (m)", 0.0, 2.0, P.sigma_m, 0.05, help="Difumina un poco el mapa de alturas antes de buscar picos. Más suavizado = menos árboles (se funden las puntas cercanas). Menos = más árboles, pero también más falsos picos.")
        P.prominencia_m = c.number_input("⛰️ Prominencia (m)", 0.02, 1.0, P.prominencia_m, 0.01, help="Cuánto tiene que destacar una punta sobre el «valle» que la separa de la siguiente para contarse como árbol distinto. Bajo = cuenta más árboles; alto = junta los que están pegados.")
        P.hmin_m = a.number_input("📏 Altura mínima individuo / suelo (m)", 0.2, 5.0, P.hmin_m, 0.1, help="Por debajo de esta altura todo se considera suelo o matorral y no se cuenta como árbol. Acordado con el laboratorio: 1,3 m.")
        P.hcorte_m = P.hmin_m
        P.h_pino_m = b.number_input("🌲 Altura de pino (m)", 3.0, 30.0, P.h_pino_m, 0.5, help="Por encima de esta altura el individuo se cuenta como pino; por debajo, como sabina. Regla del laboratorio: 5 m.")
        P.fusion_pinos = c.checkbox("🌲 Un pino = un individuo", P.fusion_pinos, help="Los pinos tienen copas grandes con varias puntas. Con esta casilla, las puntas que pertenecen al mismo pino se unen y se cuenta uno solo.")
        P.rel_min_copa = a.number_input("✂️ Límite relativo de copa", 0.0, 0.8, P.rel_min_copa, 0.05, help="Para dibujar la copa se ignoran las zonas más bajas que este porcentaje de la altura de su punta (0,3 = el 30 %). Evita que una copa «se coma» el matorral de debajo. Más alto = copas más ajustadas.")
        P.ancho_capa_m = b.number_input("🍰 Grosor de capa (m)", 0.25, 2.0, P.ancho_capa_m, 0.25, help="El árbol se corta en rebanadas horizontales de este grosor para calcular cuánta copa hay a cada altura. Estándar: 0,5 m.")
        P.umbral_alto_m = c.number_input("🔝 Umbral de auditoría 'alto' (m)", 3.0, 30.0, P.umbral_alto_m, help="Los árboles más altos que esto se listan aparte para revisarlos con calma (nunca se borran). Sirve para detectar cosas raras, como una antena confundida con un árbol.")
        P.area_min_m2 = a.number_input("📐 Área mínima de copa (m²)", 0.05, 5.0, float(P.area_min_m2), help="Las copas más pequeñas que esto se descartan o se unen a una vecina. Más alto = menos árboles diminutos (que suelen ser ruido).")
        voxel = b.checkbox("🧊 Calcular volumen ocupado por vóxeles", True, help="Calcula el volumen de copa contando cubitos de 25 cm que contienen puntos. Es más lento pero más realista que el volumen de «envoltura». Desmárcalo si quieres rapidez.")
        morf = c.checkbox("📏 Parámetros de forma de la copa", True, help="Añade a la tabla y a la capa de copas el eje mayor, la anchura máxima, la elongación y los índices de forma del polígono (las mismas fórmulas que la herramienta «Max Line Inside Polygon» de QGIS). Cuesta unos milisegundos por copa.")
    if st.button("Detectar árboles y calcular capas", type="primary"):
        S.arb, S.comp = {}, None
        for e in S.datos:
            with st.spinner(f"Árboles {e}…"):
                S.arb[e] = pl.analizar_arboles(S.modelos, S.datos, e, P, S.roi, S.estr.get(e), S.tab_est.get(e), voxel, morf)
        S.campo = None
    if not S.arb:
        st.stop()
    et = st.selectbox("🗓️ Época a visualizar", list(S.arb), index=len(S.arb) - 1, help="Año cuyos árboles quieres ver.")
    A = S.arb[et]; r = A['rodal']; p = A['params']
    npino = int((A['tabla']['especie'] == 'Pino').sum())
    k = st.columns(7)
    k[0].metric("🌳 Individuos", r['n_arboles'], help="Total de árboles detectados en la parcela."); k[1].metric("🌿 Sabinas", r['n_arboles'] - npino, help="Individuos de hasta 5 m."); k[2].metric("🌲 Pinos", npino, help="Individuos de más de 5 m (regla del laboratorio).")
    k[3].metric("📊 Densidad (ind/ha)", f"{r['densidad_ha']:.0f}", help="Árboles por hectárea."); k[4].metric("📏 H media (m)", f"{r['h_media']:.2f}", help="Altura media de los árboles.")
    k[5].metric("🔝 H máx (m)", f"{r['h_max']:.1f}", help="Altura del árbol más alto."); k[6].metric("🧊 Volumen (m³)", f"{r['volumen_total_m3']:.0f}", help="Volumen total de copa de todos los árboles.")
    c1, c2 = st.columns(2)
    mc = c1.checkbox("🟢 Contornos de copa", True, help="Dibuja el borde de cada copa en el mapa."); ma = c2.checkbox("📍 Ápices", True, help="Dibuja un punto en la punta de cada árbol.")
    st.plotly_chart(gr.capas_plotly(A, S.modelos, S.roi, S.estr.get(et), mostrar_copas=mc, mostrar_apices=ma), use_container_width=True)
    t1, t2, t3, t4, t5, t6 = st.tabs(["🌳 Individuos", "🔝 Auditoría de altos", "🍰 Capas y volúmenes", "📊 Distribuciones", "🎚️ Sensibilidad", "📋 Campo"])
    with t1:
        st.dataframe(A['tabla'], hide_index=True, use_container_width=True)
    with t2:
        altos = A['tabla'][A['tabla']['alto']].sort_values('h_m', ascending=False)
        st.success(f"{len(altos)} individuos ≥ {p['umbral_alto_m']:g} m conservados (sin límite superior de altura).")
        st.dataframe(altos, hide_index=True)
        if A['n_forzados']:
            st.info(f"{A['n_forzados']} copas añadidas por la regla de completitud (columna 'forzado'): revíselas.")
        st.subheader("Descartes registrados")
        st.dataframe(A['descartes'], hide_index=True)
    with t3:
        st.plotly_chart(gr.volumen_capas(A['cap']['global_']), use_container_width=True)
        st.dataframe(A['cap']['global_'], hide_index=True)
        st.caption("Volumen por árbol y capa (CSV completo en la exportación):")
        st.dataframe(A['vol_capas_arbol'], hide_index=True)
    with t4:
        st.plotly_chart(gr.histograma_alturas(A['tabla'], p['umbral_alto_m']), use_container_width=True)
        st.plotly_chart(gr.dispersion(A['tabla']), use_container_width=True)
        st.dataframe(r['clases_altura'], hide_index=True)
    with t5:
        st.caption("Número de copas según la prominencia: una curva plana indica un recuento estable.")
        if st.button("📈 Calcular curva de sensibilidad", help="Cuenta los árboles con distintas prominencias. Si la curva se aplana, el recuento es estable; si cae mucho, el número depende mucho del ajuste."):
            chm = np.nan_to_num(S.modelos['epocas'][et]['chm'])
            roi_m = S.roi.mascara(S.modelos['malla_chm'], S.modelos['res_chm'], chm.shape)
            xs, ns = [], []
            for pv in [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
                q = cp.ParamsArboles(**{**p, 'prominencia_m': pv})
                rr = cp.detectar_copas(chm, S.modelos['malla_chm'], S.modelos['res_chm'], q, excluir=A['excl'], roi=roi_m)
                xs.append(pv); ns.append(len(rr['tabla']))
            import plotly.graph_objects as go
            f = go.Figure(go.Scatter(x=xs, y=ns, mode="lines+markers"))
            f.update_layout(height=320, template="plotly_white", xaxis_title="Prominencia (m)", yaxis_title="N.º de copas")
            st.plotly_chart(f, use_container_width=True)

    with t6:
        if S.campo_inv is None:
            st.info("Cargue el inventario de campo en el paso 2 para validar el recuento.")
        else:
            inv = S.campo_inv
            if st.button("📋 Comparar con el inventario de campo", help="Empareja cada árbol medido en campo con el del LiDAR (a menos de 2 m) tras corregir el desfase de coordenadas. «Tapados bajo copa»: árboles de campo que están debajo de otro y el láser no ve desde arriba."):
                vivos = inv[inv.vivo & (inv.altura_m >= p['hmin_m'])]
                dx, dy, _n = cm.estimar_desplazamiento(A['tabla'], vivos)
                cmp_, cdf = cm.comparar_con_campo(A['tabla'], A['labels'], S.modelos['malla_chm'], S.modelos['res_chm'], inv, S.roi, (dx, dy), p['hmin_m'])
                ficha = cm.resumen_campo(inv.assign(x=inv.x + dx, y=inv.y + dy), S.roi, p['hmin_m'])
                S.campo = dict(cmp=cmp_, df=cdf, anio=int(S.campo_anio), desplaza=(dx, dy), ficha=ficha, et=et)
            if S.campo:
                c = S.campo['cmp']
                k = st.columns(5)
                k[0].metric("Fichas vivas ≥ 1,3 m", c['n_campo']); k[1].metric("Individuos LiDAR", c['n_lidar'])
                k[2].metric("Emparejados", c['n_emparejados']); k[3].metric("Tapados bajo copa", c['n_tapados'])
                k[4].metric("Desplazamiento E/N (m)", f"{c['desplaza_x']:+.1f} / {c['desplaza_y']:+.1f}")
                figura_png(fg.mapa_campo(A, S.modelos, S.roi, S.campo['df']))
                st.dataframe(S.campo['df'], hide_index=True)

# ===================================================================== 6
elif paso == PASOS[5]:
    st.header("🔄 6 · Cambios entre épocas")
    with st.expander("ℹ️ ¿Qué se hace en este paso?", expanded=False):
        st.write("Se comparan dos años: cada árbol del año inicial se busca en el año final. Resultado: persiste (sigue ahí), creció, mortalidad probable (ya no está), sin correspondencia (revisar) y nuevo (no estaba).")
    if len(S.arb) < 2:
        aviso_falta("Necesita al menos dos épocas con árboles detectados (paso 5).")
    ks = list(S.arb)
    c1, c2 = st.columns(2)
    a = c1.selectbox("⏮️ Época inicial", ks, index=0, help="El año más antiguo (punto de partida).")
    b = c2.selectbox("⏭️ Época final", ks, index=len(ks) - 1, help="El año más reciente, con el que se compara.")
    if a == b:
        aviso_falta("Elija dos épocas distintas.")
    radio = st.number_input("🎯 Radio de emparejamiento de ápices (m)", 0.3, 5.0, 1.2, 0.1, help="Un árbol del año inicial y otro del final son «el mismo» si sus puntas están a menos de esta distancia (o sus copas se solapan). Protocolo ICTS: 1,2 m. Más grande = empareja más, pero puede unir árboles distintos.")
    if st.button("🔄 Comparar", type="primary"):
        S.comp = pl.comparar_epocas(S.modelos, S.arb, a, b, S.roi, radio)
    comp = S.comp
    if comp is None:
        st.stop()
    r = comp['resumen']
    k = st.columns(7)
    k[0].metric("✅ Persisten", r['n_persisten'], help="Árboles que siguen en el mismo sitio."); k[1].metric("📈 Crecieron", r['n_crecieron'], help="Persisten y su altura ha subido más de 10 cm."); k[2].metric("💀 Mortalidad probable", r['n_mortalidad'], help="Estaban en el año inicial y ya no hay nada en el final. Conviene verificarlo con ortofoto o en campo.")
    k[3].metric("🔍 A revisar", r['n_sin_corr'], help="Sin pareja clara: pueden ser errores de coincidencia entre vuelos."); k[4].metric("🌱 Nuevos", r['n_nuevos'], help="Aparecen en el año final y no estaban antes (reclutamiento).")
    k[5].metric("📏 ΔH medio (m)", f"{r['dh_medio']:.2f}", help="Cambio medio de altura de los árboles que persisten."); k[6].metric("🧊 Δ volumen (m³)", f"{r['dvol_total']:+.0f}", help="Cambio total de volumen de copa entre los dos años.")
    m, res = S.modelos['malla_chm'], S.modelos['res_chm']
    st.plotly_chart(gr.raster_plotly(comp['dh'], m, res, S.roi, f"ΔH del CHM {comp['a']} → {comp['b']}", unidad="m", divergente=True), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        figura_png(fg.mapa_estados(comp, m, res, S.roi, S.arb[comp['b']]['chm_eff'], "Estado de individuos"))
    with c2:
        figura_png(fg.dispersion_h(comp)); figura_png(fg.demografia_especies(comp))
    st.dataframe(comp['tabla_dh'], hide_index=True)
    st.dataframe(comp['df_a'], hide_index=True)
    st.warning("«Mortalidad probable» y «sin correspondencia» deben verificarse (ortofoto o campo): el solape de copas y los desajustes entre vuelos generan falsos positivos.")

# ===================================================================== 7
else:
    st.header("📄 7 · Exportar e informe")
    with st.expander("ℹ️ ¿Qué se hace en este paso?", expanded=False):
        st.write("Se genera el informe (un archivo HTML que se abre en el navegador y se puede imprimir en PDF) y un paquete ZIP con los mapas, las tablas y los vectores para abrir en QGIS.")
    if not S.arb:
        aviso_falta("Detecte árboles primero (paso 5).")
    fmt = st.radio("🗂️ Formato vectorial", ["geojson", "gpkg", "shp"], horizontal=True,
                   help="Formato de las copas y árboles para QGIS. GeoJSON siempre funciona; GPKG y SHP necesitan geopandas instalado.")
    if st.button("📄 Generar informe y paquete QGIS", type="primary"):
        with st.spinner("Generando figuras, informe y archivos…"):
            meta = dict(titulo=S.titulo, autor=S.autor, crs=f"EPSG:{int(S.epsg)}")
            S.html = inf.generar_informe(S.modelos, S.datos, S.roi, S.estr, S.arb, S.comp, S.tab_est, meta, S.sintetico, campo=S.campo)
            with tempfile.TemporaryDirectory() as td:
                try:
                    S.creados = ex.exportar_todo(td, S.modelos, S.roi, S.arb, S.estr, S.tab_est, S.comp, int(S.epsg), fmt, S.html, campo=S.campo)
                    S.paquete = ex.comprimir(td)
                except Exception as e_:
                    S.paquete = None
                    st.error(f"Error al exportar ({fmt}): {e_}")
    if S.html:
        st.download_button("⬇ Informe (HTML autocontenido; imprimir → PDF)", S.html.encode("utf-8"), "Informe_ForestMap.html", "text/html")
    if S.paquete:
        st.download_button("⬇ Paquete completo (ZIP: RASTER, VECTORIALES, TABLAS, INFORME, QGIS)", S.paquete, "ForestMap_resultados.zip", "application/zip")
        st.caption("Archivos incluidos:")
        st.code("\n".join(S.get('creados', [])))
        st.info("En QGIS: cargue RASTER/CHM_*.tif (o .asc), pulse «Estilo → Cargar estilo» con QGIS/CHM_*.qml y añada VECTORIALES/Copas_*.")