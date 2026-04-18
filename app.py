import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import gaussian_kde
from dotenv import load_dotenv
import os

load_dotenv()

#  CONFIGURACIÓN GENERAL
st.set_page_config(
    page_title="Prueba de Hipotesis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#  ESTÉTICA
PIXEL_CSS = """
<style>
/* Fuente */
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

/* Reset y fondo negro */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #0a0a0a !important;
    color: #e0e0e0 !important;
    font-family: 'Press Start 2P', monospace !important;
}

/* Eliminar padding superior de Streamlit */
[data-testid="stAppViewContainer"] > .main {
    padding-top: 1rem;
}

/* Ocultar menú hamburguesa y footer */
#MainMenu, footer, header { visibility: hidden; }

/* Títulos */
h1, h2, h3 {
    font-family: 'Press Start 2P', monospace !important;
    text-shadow: 3px 3px 0px #000, 0 0 10px #00ff88;
    letter-spacing: 2px;
}

/* Botones */
.stButton > button {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.65rem !important;
    background-color: #1a472a !important;
    color: #00ff88 !important;
    border: 3px solid #00ff88 !important;
    border-radius: 0 !important;
    box-shadow: 4px 4px 0px #00ff88 !important;
    padding: 0.6rem 1.2rem !important;
    image-rendering: pixelated;
    transition: all 0.08s steps(1) !important;
    cursor: pointer;
}
.stButton > button:hover {
    background-color: #00ff88 !important;
    color: #0a0a0a !important;
    box-shadow: 2px 2px 0px #007744 !important;
    transform: translate(2px, 2px);
}
.stButton > button:active {
    box-shadow: 0px 0px 0px #007744 !important;
    transform: translate(4px, 4px);
}

/* Selectbox / inputs */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.6rem !important;
    background-color: #111 !important;
    color: #00ff88 !important;
    border: 2px solid #00ff88 !important;
    border-radius: 0 !important;
}

/* Slider */
.stSlider > div > div > div > div {
    background-color: #00ff88 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 3px solid #00ff88;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.55rem !important;
    background-color: #111 !important;
    color: #888 !important;
    border: 2px solid #333 !important;
    border-radius: 0 !important;
    padding: 0.5rem 0.8rem !important;
}
.stTabs [aria-selected="true"] {
    background-color: #1a472a !important;
    color: #00ff88 !important;
    border-color: #00ff88 !important;
}

/* Divider */
hr {
    border: none;
    border-top: 3px dashed #00ff88;
    margin: 1.5rem 0;
}

/* Sidebar a la derecha */
[data-testid="stSidebar"] {
    right: 0;
    left: auto !important;
    border-left: 3px solid #00ff88;
    border-right: none;
    background-color: #0d0d0d !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.55rem !important;
    color: #00ff88 !important;
}

/* Caja estilizada */
.pixel-box {
    border: 3px solid #00ff88;
    box-shadow: 6px 6px 0px #007744;
    background-color: #0d1f16;
    padding: 1.5rem;
    margin: 1rem 0;
    image-rendering: pixelated;
}

/* Caja de advertencia */
.pixel-box-warn {
    border: 3px solid #ffcc00;
    box-shadow: 6px 6px 0px #996600;
    background-color: #1a1500;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
}

/* Caja de éxito */
.pixel-box-ok {
    border: 3px solid #00ff88;
    box-shadow: 6px 6px 0px #007744;
    background-color: #0d1f16;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
}

/* Tabla de datos */
[data-testid="stDataFrame"] {
    border: 2px solid #00ff88 !important;
}
[data-testid="stDataFrame"] table {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.45rem !important;
    background-color: #0a0a0a !important;
    color: #00ff88 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #00ff88 !important;
    background-color: #0d1f16 !important;
    border-radius: 0 !important;
}
[data-testid="stFileUploader"] * {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.5rem !important;
    color: #00ff88 !important;
}

/* Radio buttons */
.stRadio > div {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.55rem !important;
    color: #00ff88 !important;
}

/* Texto de bienvenida */
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}
.blink-cursor::after {
    content: '█';
    animation: blink 1s steps(1) infinite;
    color: #00ff88;
}

/* Badge */
.level-badge {
    display: inline-block;
    background: #00ff88;
    color: #000;
    font-family: 'Press Start 2P', monospace;
    font-size: 0.5rem;
    padding: 0.2rem 0.5rem;
    box-shadow: 3px 3px 0 #007744;
}
</style>
"""
st.markdown(PIXEL_CSS, unsafe_allow_html=True)

#  PANTALLA DE INICIO
def render_home():
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='text-align:center; padding: 2rem 0;'>
            <div style='font-family:"Press Start 2P",monospace; font-size:2.2rem;
                        color:#00ff88; text-shadow: 4px 4px 0 #000, 0 0 20px #00ff88;
                        line-height:1.6; letter-spacing:4px;'>
                📊 Prueba de hipotesis
            </div>
            <div style='font-family:"Press Start 2P",monospace; font-size:0.65rem;
                        color:#888; margin-top:1rem; letter-spacing:2px;'>
                — ESTADISTICA INFERENCIAL —
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 3, 1])
    with col_center:
        st.markdown(
            """
            <div class='pixel-box' style='text-align:center;'>
                <p style='font-size:1rem; line-height:2.2; color:#ccc;'>
                    Bienvenido a este programa.<br><br>
                    Carga tus datos, visualiza distribuciones,<br>
                    realiza pruebas de hipótesis<br>
                    y consulta el módulo de IA.<br><br>
                    Realiza un doble click en los botones para acceder a su contenido.<br><br>
                    <span class='blink-cursor' style='color:#00ff88; font-size:1rem;'>
                        Gracias por usar el programa!
                    </span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            "<div style='text-align:center; font-family:\"Press Start 2P\",monospace;"
            "font-size:0.5rem; color:#555; margin-bottom:0.5rem;'>▼ SELECCIONA MÓDULO ▼</div>",
            unsafe_allow_html=True,
        )

        btn1, btn2, btn3 = st.columns(3)
        with btn1:
            if st.button("CARGAR DATOS 📂", use_container_width=True):
                st.session_state["modulo"] = "carga"
                st.rerun()
        with btn2:
            if st.button("DISTRIBUCIONES 📈", use_container_width=True):
                st.session_state["modulo"] = "visualizacion"
                st.rerun()
        with btn3:
            if st.button("HIPÓTESIS 🧪", use_container_width=True):
                st.session_state["modulo"] = "pruebas"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        col_ia, _ = st.columns([1, 1])
        with col_ia:
            if st.button("Módulo de IA 🤖", use_container_width=True):
                st.session_state["modulo"] = "ia"
                st.rerun()


#  MÓDULO 1 — CARGA DE DATOS
def render_carga():
    st.markdown(
        "<div style='font-family:\"Press Start 2P\",monospace; font-size:1.1rem;"
        "color:#00ff88; text-shadow:3px 3px 0 #000; margin-bottom:1rem;'>"
        "CARGA DE DATOS 📂</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    fuente = st.radio(
        "Selecciona la fuente de datos:",
        ["Subir archivo CSV 📁", "Generar datos sintéticos 🎲"],
        horizontal=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    df = None

    #Opción A: CSV
    if "CSV" in fuente:
        st.markdown(
            "<div class='pixel-box-warn'>"
            "<span style='font-size:1rem; color:#ffcc00;'>"
            "⚠ El archivo debe ser .csv con encabezados en la primera fila.</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        archivo = st.file_uploader("     Sube tu archivo CSV:", type=["csv"])
        if archivo is not None:
            try:
                df = pd.read_csv(archivo)
                st.markdown(
                    "<div class='pixel-box-ok'>"
                    f"<span style='font-size:0.5rem; color:#00ff88;'>"
                    f"✔ Archivo cargado: {archivo.name} — "
                    f"{df.shape[0]} filas × {df.shape[1]} columnas</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.markdown(
                    f"<div class='pixel-box-warn'><span style='font-size:0.5rem;"
                    f"color:#ff4444;'>✘ Error al leer el archivo: {e}</span></div>",
                    unsafe_allow_html=True,
                )

    #Opción B: Datos sintéticos
    else:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            n = st.number_input("Tamaño de muestra (n):", min_value=30,
                                max_value=10000, value=100, step=10)
        with col_b:
            distribucion = st.selectbox(
                "Distribución:",
                ["Normal", "Uniforme", "Sesgada (log-normal)"],
            )
        with col_c:
            semilla = st.number_input("Semilla aleatoria:", min_value=0,
                                      max_value=9999, value=42, step=1)

        rng = np.random.default_rng(int(semilla))

        if distribucion == "Normal":
            mu    = st.slider("Media (μ):", -100.0, 100.0, 0.0, 0.5)
            sigma = st.slider("Desv. estándar (σ):", 0.1, 50.0, 1.0, 0.1)
            datos = rng.normal(loc=mu, scale=sigma, size=int(n))
        elif distribucion == "Uniforme":
            low  = st.slider("Mínimo:", -100.0, 0.0, 0.0, 0.5)
            high = st.slider("Máximo:", 0.1, 200.0, 10.0, 0.5)
            datos = rng.uniform(low=low, high=high, size=int(n))
        else:
            mu_ln    = st.slider("Media (log):", 0.0, 5.0, 0.0, 0.1)
            sigma_ln = st.slider("Desv. (log):", 0.1, 2.0, 0.5, 0.05)
            datos = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=int(n))

        df = pd.DataFrame({"valor": datos})
        st.markdown(
            "<div class='pixel-box-ok'>"
            f"<span style='font-size:1rem; color:#00ff88;'>"
            f"✔ Datos sintéticos generados: {int(n)} observaciones — "
            f"distribución {distribucion}</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    #Selección de variable y preview
    if df is not None:
        st.markdown("<hr>", unsafe_allow_html=True)

        columnas_num = df.select_dtypes(include=[np.number]).columns.tolist()
        if not columnas_num:
            st.markdown(
                "<div class='pixel-box-warn'><span style='font-size:0.5rem; color:#ff4444;'>"
                "✘ No se encontraron columnas numéricas en el archivo.</span></div>",
                unsafe_allow_html=True,
            )
        else:
            col_sel, col_stats = st.columns([1, 2])

            with col_sel:
                variable = st.selectbox("Variable de análisis:", columnas_num)

            with col_stats:
                serie = df[variable].dropna()
                st.markdown(
                    f"""
                    <div class='pixel-box' style='font-size:1rem; line-height:2;'>
                        <span style='color:#888;'>N válido:</span>
                        <span style='color:#00ff88;'>{len(serie)}</span><br>
                        <span style='color:#888;'>Media:</span>
                        <span style='color:#00ff88;'>{serie.mean():.4f}</span><br>
                        <span style='color:#888;'>Desv. std:</span>
                        <span style='color:#00ff88;'>{serie.std():.4f}</span><br>
                        <span style='color:#888;'>Mín / Máx:</span>
                        <span style='color:#00ff88;'>{serie.min():.4f} / {serie.max():.4f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<div style='font-family:\"Press Start 2P\",monospace; font-size:0.5rem;"
                "color:#555; margin:0.5rem 0;'>▼ VISTA PREVIA (primeras 10 filas)</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(df.head(10), use_container_width=True, hide_index=False)

            # Guardar en session_state para módulos futuros
            st.session_state["df"]       = df
            st.session_state["variable"] = variable
            st.session_state["serie"]    = serie.values

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<div class='pixel-box-ok' style='text-align:center;'>"
                "<span style='font-size:1rem; color:#00ff88;'>"
                "✔ Datos listos. Puedes continuar a los otros módulos.</span>"
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅  VOLVER AL INICIO"):
        st.session_state["modulo"] = "inicio"
        st.rerun()


# ──────────────────────────────────────────────
#  MÓDULO 2 — VISUALIZACIÓN DE DISTRIBUCIONES
# ──────────────────────────────────────────────

def _pixel_fig():
    """Devuelve una figura Matplotlib con estética pixel/dark."""
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0d1f16")
    ax.tick_params(colors="#00ff88", labelsize=7)
    ax.xaxis.label.set_color("#00ff88")
    ax.yaxis.label.set_color("#00ff88")
    ax.title.set_color("#00ff88")
    for spine in ax.spines.values():
        spine.set_edgecolor("#00ff88")
        spine.set_linewidth(1.5)
    ax.grid(color="#1a3d2a", linestyle="--", linewidth=0.6, alpha=0.7)
    return fig, ax


def _analisis_automatico(serie):
    """Retorna un dict con conclusiones sobre la distribución."""
    skew     = stats.skew(serie)
    kurt     = stats.kurtosis(serie)   # exceso de kurtosis
    q1, q3   = np.percentile(serie, [25, 75])
    iqr      = q3 - q1
    lim_inf  = q1 - 1.5 * iqr
    lim_sup  = q3 + 1.5 * iqr
    n_out    = int(np.sum((serie < lim_inf) | (serie > lim_sup)))

    # Shapiro-Wilk (sólo si n <= 5000)
    if len(serie) <= 5000:
        _, p_sw = stats.shapiro(serie)
    else:
        _, p_sw = stats.normaltest(serie)   # D'Agostino si n grande

    return {
        "skew":    skew,
        "kurt":    kurt,
        "n_out":   n_out,
        "p_norm":  p_sw,
        "lim_inf": lim_inf,
        "lim_sup": lim_sup,
    }


def render_visualizacion():
    st.markdown(
        "<div style='font-family:\"Press Start 2P\",monospace; font-size:1.1rem;"
        "color:#00ff88; text-shadow:3px 3px 0 #000; margin-bottom:1rem;'>"
        "DISTRIBUCIONES 📈</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Verificar que haya datos cargados ─────
    if "serie" not in st.session_state or st.session_state["serie"] is None:
        st.markdown(
            "<div class='pixel-box-warn' style='text-align:center;'>"
            "<span style='font-size:0.8rem; color:#ffcc00;'>"
            "⚠ No hay datos cargados.<br><br>"
            "Ve primero al módulo <b>CARGAR DATOS</b> y carga o genera tu dataset."
            "</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📂  IR A CARGAR DATOS"):
            st.session_state["modulo"] = "carga"
            st.rerun()
        if st.button("⬅  VOLVER AL INICIO"):
            st.session_state["modulo"] = "inicio"
            st.rerun()
        return

    serie    = st.session_state["serie"]
    variable = st.session_state.get("variable", "variable")

    # ── Opciones de gráficas ───────────────────
    st.markdown(
        "<div style='font-family:\"Press Start 2P\",monospace; font-size:0.5rem;"
        "color:#555; margin-bottom:0.4rem;'>▼ OPCIONES DE VISUALIZACIÓN</div>",
        unsafe_allow_html=True,
    )

    col_op1, col_op2, col_op3, col_op4 = st.columns(4)
    with col_op1:
        mostrar_hist = st.checkbox("Histograma", value=True)
    with col_op2:
        mostrar_kde  = st.checkbox("Curva KDE", value=True)
    with col_op3:
        mostrar_box  = st.checkbox("Boxplot", value=True)
    with col_op4:
        n_bins = st.slider("Bins del histograma:", 5, 80, 20, 1)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── HISTOGRAMA + KDE ──────────────────────
    if mostrar_hist or mostrar_kde:
        st.markdown(
            "<div style='font-family:\"Press Start 2P\",monospace;"
            "font-size:0.55rem; color:#00ff88; margin-bottom:0.3rem;'>"
            "■ HISTOGRAMA" + (" + KDE" if mostrar_kde else "") + "</div>",
            unsafe_allow_html=True,
        )

        fig, ax = _pixel_fig()

        if mostrar_hist:
            ax.hist(
                serie,
                bins=n_bins,
                color="#1a472a",
                edgecolor="#00ff88",
                linewidth=1.2,
                alpha=0.85,
                label="Frecuencia",
                density=mostrar_kde,   # normaliza si va con KDE
            )

        if mostrar_kde and len(serie) > 3:
            kde    = gaussian_kde(serie, bw_method="scott")
            x_kde  = np.linspace(serie.min(), serie.max(), 400)
            y_kde  = kde(x_kde)
            ax.plot(x_kde, y_kde, color="#00ffcc", linewidth=2.5,
                    linestyle="--", label="KDE")

        ax.set_xlabel(variable, fontsize=8, fontfamily="monospace")
        ax.set_ylabel("Densidad" if mostrar_kde else "Frecuencia",
                      fontsize=8, fontfamily="monospace")
        ax.set_title(f"Distribución de '{variable}'",
                     fontsize=9, fontfamily="monospace", pad=10)
        ax.legend(facecolor="#0d1f16", edgecolor="#00ff88",
                  labelcolor="#00ff88", fontsize=7)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── BOXPLOT ───────────────────────────────
    if mostrar_box:
        st.markdown(
            "<div style='font-family:\"Press Start 2P\",monospace;"
            "font-size:0.55rem; color:#00ff88; margin:0.8rem 0 0.3rem;'>"
            "■ BOXPLOT</div>",
            unsafe_allow_html=True,
        )

        fig2, ax2 = _pixel_fig()
        bp = ax2.boxplot(
            serie,
            vert=False,
            patch_artist=True,
            widths=0.5,
            boxprops=dict(facecolor="#1a472a", color="#00ff88", linewidth=1.5),
            medianprops=dict(color="#00ffcc", linewidth=2.5),
            whiskerprops=dict(color="#00ff88", linewidth=1.5, linestyle="--"),
            capprops=dict(color="#00ff88", linewidth=2),
            flierprops=dict(marker="D", color="#ff4444", markersize=5,
                            markerfacecolor="#ff4444"),
        )

        ax2.set_xlabel(variable, fontsize=8, fontfamily="monospace")
        ax2.set_title(f"Boxplot de '{variable}'",
                      fontsize=9, fontfamily="monospace", pad=10)
        ax2.set_yticks([])

        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # ── ANÁLISIS AUTOMÁTICO ───────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:\"Press Start 2P\",monospace;"
        "font-size:0.55rem; color:#00ff88; margin-bottom:0.6rem;'>"
        "■ ANÁLISIS AUTOMÁTICO</div>",
        unsafe_allow_html=True,
    )

    res = _analisis_automatico(serie)

    # Conclusión normalidad
    es_normal = res["p_norm"] > 0.05
    icono_norm = "✔" if es_normal else "✘"
    color_norm = "#00ff88" if es_normal else "#ff4444"
    texto_norm = "Sí parece normal (p > 0.05)" if es_normal \
                 else "No parece normal (p ≤ 0.05)"

    # Conclusión sesgo
    sk = res["skew"]
    if abs(sk) < 0.5:
        texto_sesgo = f"Sin sesgo significativo (skew = {sk:.3f})"
        color_sesgo = "#00ff88"
    elif sk > 0:
        texto_sesgo = f"Sesgo positivo / cola derecha (skew = {sk:.3f})"
        color_sesgo = "#ffcc00"
    else:
        texto_sesgo = f"Sesgo negativo / cola izquierda (skew = {sk:.3f})"
        color_sesgo = "#ffcc00"

    # Conclusión outliers
    n_out = res["n_out"]
    color_out = "#00ff88" if n_out == 0 else "#ff4444"
    texto_out = f"{n_out} outlier(s) detectado(s) (método IQR)"

    st.markdown(
        f"""
        <div class='pixel-box' style='font-size:1rem; line-height:2.4;'>
            <span style='color:#888;'>¿Distribución normal?</span><br>
            <span style='color:{color_norm};'>{icono_norm} {texto_norm}</span>
            <br><br>
            <span style='color:#888;'>¿Hay sesgo?</span><br>
            <span style='color:{color_sesgo};'>► {texto_sesgo}</span>
            <br><br>
            <span style='color:#888;'>¿Hay outliers?</span><br>
            <span style='color:{color_out};'>► {texto_out}</span>
            <br><br>
            <span style='color:#555; font-size:1rem;'>
                Kurtosis (exceso): {res["kurt"]:.3f} &nbsp;|&nbsp;
                Límite IQR inferior: {res["lim_inf"]:.3f} &nbsp;|&nbsp;
                Límite IQR superior: {res["lim_sup"]:.3f}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅  VOLVER AL INICIO"):
        st.session_state["modulo"] = "inicio"
        st.rerun()


# ──────────────────────────────────────────────
#  MÓDULO 3 — PRUEBA DE HIPÓTESIS (PRUEBA Z)
# ──────────────────────────────────────────────

def _calcular_z(media_muestral, mu0, sigma, n):
    """Calcula el estadístico Z."""
    return (media_muestral - mu0) / (sigma / np.sqrt(n))


def _pvalue(z, tipo_cola):
    """Calcula el p-value según el tipo de cola."""
    if tipo_cola == "Bilateral":
        return 2 * (1 - stats.norm.cdf(abs(z)))
    elif tipo_cola == "Cola derecha (H1: μ > μ₀)":
        return 1 - stats.norm.cdf(z)
    else:  # Cola izquierda
        return stats.norm.cdf(z)


def _grafica_z(z_calc, alpha, tipo_cola):
    """Curva normal con región de rechazo sombreada."""
    fig, ax = _pixel_fig()
    x = np.linspace(-4.2, 4.2, 600)
    y = stats.norm.pdf(x)

    # Curva principal
    ax.plot(x, y, color="#00ff88", linewidth=2)
    ax.fill_between(x, y, color="#1a472a", alpha=0.3)

    # Regiones de rechazo
    if tipo_cola == "Bilateral":
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ax.fill_between(x, y, where=(x >= z_crit),  color="#ff4444", alpha=0.55, label=f"Rechazo (α/2)")
        ax.fill_between(x, y, where=(x <= -z_crit), color="#ff4444", alpha=0.55)
        ax.axvline( z_crit, color="#ff4444", linewidth=1.5, linestyle="--")
        ax.axvline(-z_crit, color="#ff4444", linewidth=1.5, linestyle="--")
    elif tipo_cola == "Cola derecha (H1: μ > μ₀)":
        z_crit = stats.norm.ppf(1 - alpha)
        ax.fill_between(x, y, where=(x >= z_crit), color="#ff4444", alpha=0.55, label="Rechazo (α)")
        ax.axvline(z_crit, color="#ff4444", linewidth=1.5, linestyle="--")
    else:
        z_crit = stats.norm.ppf(alpha)
        ax.fill_between(x, y, where=(x <= z_crit), color="#ff4444", alpha=0.55, label="Rechazo (α)")
        ax.axvline(z_crit, color="#ff4444", linewidth=1.5, linestyle="--")

    # Línea del Z calculado
    ax.axvline(z_calc, color="#ffcc00", linewidth=2.5, linestyle="-",
               label=f"Z calc = {z_calc:.4f}")

    ax.set_xlabel("Z", fontsize=8, fontfamily="monospace")
    ax.set_ylabel("Densidad", fontsize=8, fontfamily="monospace")
    ax.set_title("Distribución Z con región de rechazo", fontsize=9,
                 fontfamily="monospace", pad=10)
    ax.legend(facecolor="#0d1f16", edgecolor="#00ff88",
              labelcolor="#00ff88", fontsize=7)
    plt.tight_layout()
    return fig


def render_pruebas():
    st.markdown(
        "<div style='font-family:\"Press Start 2P\",monospace; font-size:1.1rem;"
        "color:#00ff88; text-shadow:3px 3px 0 #000; margin-bottom:1rem;'>"
        "HIPÓTESIS 🧪</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Verificar datos ────────────────────────
    if "serie" not in st.session_state or st.session_state["serie"] is None:
        st.markdown(
            "<div class='pixel-box-warn' style='text-align:center;'>"
            "<span style='font-size:0.8rem; color:#ffcc00;'>"
            "⚠ No hay datos cargados.<br><br>"
            "Ve primero al módulo <b>CARGAR DATOS</b>."
            "</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📂  IR A CARGAR DATOS"):
            st.session_state["modulo"] = "carga"
            st.rerun()
        if st.button("⬅  VOLVER AL INICIO"):
            st.session_state["modulo"] = "inicio"
            st.rerun()
        return

    serie    = st.session_state["serie"]
    variable = st.session_state.get("variable", "variable")
    n        = len(serie)
    media_m  = float(np.mean(serie))
    std_m    = float(np.std(serie, ddof=1))

    # ── Info de la muestra ─────────────────────
    st.markdown(
        f"""
        <div class='pixel-box' style='font-size:1rem; line-height:2.2;'>
            <span style='color:#888;'>Variable:</span>
            <span style='color:#00ff88;'>{variable}</span>&nbsp;&nbsp;
            <span style='color:#888;'>|&nbsp; n =</span>
            <span style='color:#00ff88;'>{n}</span>&nbsp;&nbsp;
            <span style='color:#888;'>|&nbsp; x̄ =</span>
            <span style='color:#00ff88;'>{media_m:.4f}</span>&nbsp;&nbsp;
            <span style='color:#888;'>|&nbsp; s =</span>
            <span style='color:#00ff88;'>{std_m:.4f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if n < 30:
        st.markdown(
            "<div class='pixel-box-warn'><span style='font-size:1rem; color:#ffcc00;'>"
            "⚠ n < 30: La prueba Z asume n ≥ 30 o varianza poblacional conocida. "
            "Interpreta con precaución.</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:\"Press Start 2P\",monospace; font-size:0.5rem;"
        "color:#555; margin-bottom:0.6rem;'>▼ DEFINIR HIPÓTESIS</div>",
        unsafe_allow_html=True,
    )

    # ── Parámetros de la prueba ────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        mu0 = st.number_input(
            "H₀: μ₀ (media hipotética):",
            value=0.0, step=0.1, format="%.4f",
        )
    with col2:
        sigma = st.number_input(
            "σ poblacional conocida:",
            min_value=0.0001, value=round(std_m, 4),
            step=0.01, format="%.4f",
        )
    with col3:
        alpha = st.selectbox(
            "Nivel de significancia (α):",
            [0.01, 0.05, 0.10],
            index=1,
            format_func=lambda x: f"{x} ({int(x*100)}%)",
        )

    tipo_cola = st.radio(
        "Tipo de prueba:",
        ["Bilateral", "Cola derecha (H1: μ > μ₀)", "Cola izquierda (H1: μ < μ₀)"],
        horizontal=True,
    )

    # Mostrar hipótesis en texto
    if tipo_cola == "Bilateral":
        h1_texto = f"H₁: μ ≠ {mu0}"
    elif "derecha" in tipo_cola:
        h1_texto = f"H₁: μ > {mu0}"
    else:
        h1_texto = f"H₁: μ < {mu0}"

    st.markdown(
        f"""
        <div class='pixel-box' style='font-size:1rem; line-height:2.2; text-align:center;'>
            <span style='color:#888;'>H₀:</span>
            <span style='color:#00ff88;'> μ = {mu0}</span>
            &nbsp;&nbsp;&nbsp;
            <span style='color:#888;'>vs</span>
            &nbsp;&nbsp;&nbsp;
            <span style='color:#ffcc00;'>{h1_texto}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Cálculo ────────────────────────────────
    if st.button("▶  EJECUTAR PRUEBA Z", use_container_width=False):
        z_calc = _calcular_z(media_m, mu0, sigma, n)
        p_val  = _pvalue(z_calc, tipo_cola)
        rechazar = p_val < alpha

        # Guardar resultados en session_state para el módulo de IA
        st.session_state["resultado_z"] = {
            "variable":      variable,
            "n":             n,
            "media_muestral": media_m,
            "sigma":         sigma,
            "mu0":           mu0,
            "alpha":         alpha,
            "tipo_cola":     tipo_cola,
            "z_calc":        z_calc,
            "p_value":       p_val,
            "rechazar_h0":   rechazar,
            "h1_texto":      h1_texto,
        }

        # ── Resultados numéricos ───────────────
        color_dec = "#ff4444" if rechazar else "#00ff88"
        decision  = "SE RECHAZA H₀" if rechazar else "NO SE RECHAZA H₀"
        icono_dec = "✘" if rechazar else "✔"

        st.markdown(
            f"""
            <div class='pixel-box' style='font-size:1rem; line-height:2.4;'>
                <span style='color:#888;'>Estadístico Z:</span>
                <span style='color:#ffcc00; font-size:1rem;'> {z_calc:.6f}</span>
                <br>
                <span style='color:#888;'>p-value:</span>
                <span style='color:#ffcc00; font-size:1rem;'> {p_val:.6f}</span>
                <br>
                <span style='color:#888;'>α:</span>
                <span style='color:#00ff88;'> {alpha}</span>
                <br><br>
                <span style='color:#888;'>Decisión:</span>
                <span style='color:{color_dec}; font-size:1rem;'>
                    {icono_dec} {decision}
                </span>
                <br><br>
                <span style='color:#555; font-size:0.5rem;'>
                    {"p-value < α → evidencia suficiente para rechazar H₀"
                      if rechazar else
                     "p-value ≥ α → evidencia insuficiente para rechazar H₀"}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Gráfica ────────────────────────────
        st.markdown(
            "<div style='font-family:\"Press Start 2P\",monospace; font-size:0.55rem;"
            "color:#00ff88; margin: 0.8rem 0 0.3rem;'>■ CURVA Z CON REGIÓN DE RECHAZO</div>",
            unsafe_allow_html=True,
        )
        fig_z = _grafica_z(z_calc, alpha, tipo_cola)
        st.pyplot(fig_z, use_container_width=True)
        plt.close(fig_z)

        # ── Aviso hacia módulo IA ──────────────
        st.markdown(
            "<div class='pixel-box-ok' style='text-align:center; margin-top:1rem;'>"
            "<span style='font-size:1rem; color:#00ff88;'>"
            "✔ Resultados guardados. Puedes consultarlos con el "
            "<b>Módulo de IA 🤖</b> para obtener una interpretación automática."
            "</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅  VOLVER AL INICIO"):
        st.session_state["modulo"] = "inicio"
        st.rerun()


# ──────────────────────────────────────────────
#  MÓDULO 4 — IA (PLACEHOLDER PREPARADO)
# ──────────────────────────────────────────────

def _construir_prompt(r):
    """Construye el prompt que se enviará a Groq con el resumen estadístico."""
    decision = "SE RECHAZA H₀" if r["rechazar_h0"] else "NO SE RECHAZA H₀"
    return f"""Se realizó una prueba Z con los siguientes parámetros:

    - Variable analizada: {r["variable"]}
    - Tamaño de muestra: n = {r["n"]}
    - Media muestral: x̄ = {r["media_muestral"]:.4f}
    - Media hipotética (H₀): μ₀ = {r["mu0"]}
    - Desviación estándar poblacional: σ = {r["sigma"]:.4f}
    - Nivel de significancia: α = {r["alpha"]}
    - Tipo de prueba: {r["tipo_cola"]}
    - Hipótesis alternativa: {r["h1_texto"]}

    Resultados:
    - Estadístico Z calculado = {r["z_calc"]:.6f}
    - p-value = {r["p_value"]:.6f}
    - Decisión automática: {decision}

    Con base en esta información, responde lo siguiente en español:
    1. ¿Se rechaza H₀? Explica la decisión de forma clara.
    2. ¿Los supuestos de la prueba Z son razonables dado el tamaño de muestra?
    3. ¿Qué se puede inferir estadísticamente de estos resultados?
    4. ¿Hay alguna precaución o limitación que el estudiante deba considerar?

    Responde de forma didáctica y sin usar formato Markdown (sin asteriscos, sin #). Usa texto plano. Es obligatorio que respondas las 4 preguntas completas y numeradas. No omitas ninguna."""


def _consultar_gemini(api_key, prompt):
    """Llama a la API oficial de Google Gemini y retorna el texto de respuesta."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    modelo = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config={
            "temperature": 0.4,
            "max_output_tokens": 2048,
        },
    )
    respuesta = modelo.generate_content(prompt)
    return respuesta.text

def render_ia():
    st.markdown(
        "<div style='font-family:\"Press Start 2P\",monospace; font-size:1.1rem;"
        "color:#00ff88; text-shadow:3px 3px 0 #000; margin-bottom:1rem;'>"
        "MÓDULO DE IA 🤖</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    resultado = st.session_state.get("resultado_z", None)

    # ── Sin prueba ejecutada ───────────────────
    if resultado is None:
        st.markdown(
            "<div class='pixel-box-warn' style='text-align:center;'>"
            "<span style='font-size:0.8rem; color:#ffcc00;'>"
            "⚠ Aún no hay resultados de una prueba Z.<br><br>"
            "Ve primero al módulo <b>HIPÓTESIS 🧪</b> y ejecuta la prueba."
            "</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("IR A HIPÓTESIS 🧪"):
            st.session_state["modulo"] = "pruebas"
            st.rerun()
        if st.button("⬅  VOLVER AL INICIO"):
            st.session_state["modulo"] = "inicio"
            st.rerun()
        return

    # ── Resumen de la prueba ───────────────────
    r = resultado
    st.markdown(
        "<div style='font-family:\"Press Start 2P\",monospace; font-size:0.5rem;"
        "color:#555; margin-bottom:0.6rem;'>▼ RESUMEN DE LA PRUEBA EJECUTADA</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class='pixel-box' style='font-size:0.65rem; line-height:2.4;'>
            <span style='color:#888;'>Variable:</span>
            <span style='color:#00ff88;'> {r["variable"]}</span>
            &nbsp;&nbsp;
            <span style='color:#888;'>| n:</span>
            <span style='color:#00ff88;'> {r["n"]}</span>
            &nbsp;&nbsp;
            <span style='color:#888;'>| x̄:</span>
            <span style='color:#00ff88;'> {r["media_muestral"]:.4f}</span>
            &nbsp;&nbsp;
            <span style='color:#888;'>| μ₀:</span>
            <span style='color:#00ff88;'> {r["mu0"]}</span>
            <br>
            <span style='color:#888;'>Z calc:</span>
            <span style='color:#ffcc00;'> {r["z_calc"]:.6f}</span>
            &nbsp;&nbsp;
            <span style='color:#888;'>| p-value:</span>
            <span style='color:#ffcc00;'> {r["p_value"]:.6f}</span>
            &nbsp;&nbsp;
            <span style='color:#888;'>| α:</span>
            <span style='color:#00ff88;'> {r["alpha"]}</span>
            <br>
            <span style='color:#888;'>Decisión:</span>
            <span style='color:{"#ff4444" if r["rechazar_h0"] else "#00ff88"};'>
                {"✘ SE RECHAZA H₀" if r["rechazar_h0"] else "✔ NO SE RECHAZA H₀"}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── API Key ────────────────────────────────
    st.markdown(
        "<div style='font-family:\"Press Start 2P\",monospace; font-size:0.5rem;"
        "color:#555; margin-bottom:0.4rem;'>▼ CONFIGURACIÓN DE LA IA</div>",
        unsafe_allow_html=True,
    )

    api_key = os.getenv("GROQ_API_KEY") or st.text_input(
        "Ingresa tu API Key de Gemini:",
        type="password",
        placeholder="sk-or-...",
        help="Obtén tu clave gratuita en https://openrouter.ai/keys",
    )

    if os.getenv("GROQ_API_KEY"):
        st.markdown(
            "<div style='font-size:0.45rem; color:#00ff88; margin-top:-0.5rem; margin-bottom:0.8rem;'>"
            "✔ API Key cargada desde .env</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='font-size:0.45rem; color:#555; margin-top:-0.5rem; margin-bottom:0.8rem;'>"
            "La clave NO se almacena. Solo se usa durante esta sesión.</div>",
            unsafe_allow_html=True,
        )

    # Mostrar el prompt que se enviará
    with st.expander("👁  Ver prompt que se enviará a Gemini"):
        st.code(_construir_prompt(r), language=None)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Botón consultar ────────────────────────
    if st.button("CONSULTAR A LA IA ⚙️", use_container_width=False):
        if not api_key or len(api_key.strip()) < 10:
            st.markdown(
                "<div class='pixel-box-warn'><span style='font-size:0.65rem; color:#ff4444;'>"
                "✘ Ingresa una API Key válida antes de continuar.</span></div>",
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Consultando al módulo de IA..."):
                try:
                    prompt   = _construir_prompt(r)
                    respuesta = _consultar_gemini(api_key.strip(), prompt)
                    st.session_state["respuesta_ia"] = respuesta
                except Exception as e:
                    st.session_state["respuesta_ia"] = None
                    st.markdown(
                        f"<div class='pixel-box-warn'><span style='font-size:0.65rem; color:#ff4444;'>"
                        f"✘ Error al contactar la API: {e}</span></div>",
                        unsafe_allow_html=True,
                    )

    # ── Mostrar respuesta ──────────────────────
    respuesta_guardada = st.session_state.get("respuesta_ia", None)
    if respuesta_guardada:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-family:\"Press Start 2P\",monospace; font-size:0.55rem;"
            "color:#00ff88; margin-bottom:0.6rem;'>■ RESPUESTA DEL ORÁCULO</div>",
            unsafe_allow_html=True,
        )
        # Formatear la respuesta línea por línea para respetar el CSS pixel
        lineas = respuesta_guardada.strip().split("\n")
        lineas_html = "".join(
            f"<span>{linea}</span><br>" if linea.strip() else "<br>"
            for linea in lineas
        )
        
        st.markdown(
            f"""
            <div class='pixel-box' style='font-size:0.65rem; line-height:2.2; color:#ccc;
                overflow-y: auto; max-height: 500px;'>
                {lineas_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Comparación decisión estudiante vs IA ─
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-family:\"Press Start 2P\",monospace; font-size:0.5rem;"
            "color:#555; margin-bottom:0.6rem;'>▼ COMPARA TU DECISIÓN</div>",
            unsafe_allow_html=True,
        )

        decision_auto = "SE RECHAZA H₀" if r["rechazar_h0"] else "NO SE RECHAZA H₀"
        col_a, col_b = st.columns(2)

        with col_a:
            decision_user = st.radio(
                "¿Cuál fue tu decisión?",
                ["SE RECHAZA H₀", "NO SE RECHAZA H₀"],
                index=0,
            )

        with col_b:
            if decision_user == decision_auto:
                st.markdown(
                    "<div class='pixel-box-ok' style='text-align:center; margin-top:1rem;'>"
                    "<span style='font-size:0.7rem; color:#00ff88;'>"
                    "✔ ¡CORRECTO!<br><br>Tu decisión coincide<br>con el resultado."
                    "</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='pixel-box-warn' style='text-align:center; margin-top:1rem;'>"
                    f"<span style='font-size:0.65rem; color:#ffcc00;'>"
                    f"✘ Revisa tu decisión.<br><br>"
                    f"La respuesta correcta es:<br>"
                    f"<b style='color:#ff4444;'>{decision_auto}</b>"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅  VOLVER AL INICIO"):
        st.session_state["respuesta_ia"] = None
        st.session_state["modulo"] = "inicio"
        st.rerun()


#  ESTADO INICIAL
if "modulo" not in st.session_state:
    st.session_state["modulo"] = "inicio"

if "resultado_z" not in st.session_state:
    st.session_state["resultado_z"] = None

if "respuesta_ia" not in st.session_state:
    st.session_state["respuesta_ia"] = None


#  ROUTER PRINCIPAL
modulo = st.session_state["modulo"]

if modulo == "inicio":
    render_home()

elif modulo == "carga":
    render_carga()

elif modulo == "visualizacion":
    render_visualizacion()

elif modulo == "pruebas":
    render_pruebas()

elif modulo == "ia":
    render_ia()

else:
    st.markdown(
        f"""
        <div class='pixel-box' style='text-align:center; margin-top:3rem;'>
            <p style='font-size:0.6rem; color:#888;'>
                [ MÓDULO <span style='color:#00ff88;'>
                {st.session_state['modulo'].upper()}
                </span> — no encontrado ]
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("⬅  VOLVER AL INICIO"):
        st.session_state["modulo"] = "inicio"
        st.rerun()