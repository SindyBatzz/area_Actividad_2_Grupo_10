"""
Laboratorio Final — Starbucks Rewards
App de Streamlit para explorar el Qini del T-learner y la tabla de políticas del Paso 3.

Correr con:  streamlit run streamlit_starter.py
"""

from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Laboratorio Final — Starbucks Rewards", page_icon="☕", layout="wide")

GREEN = "#00704A"
ORANGE = "#C4512F"
GRAY = "#9A988E"
FEATURES = ["recency_days", "frequency", "monetary", "email_open_rate", "tenure_days"]

# Compatibilidad: np.trapezoid existe desde NumPy 2.0; antes se llamaba np.trapz
_trapz = getattr(np, "trapezoid", None) or np.trapz


@st.cache_data
def load_data():
    # Ruta relativa al archivo, para que funcione igual local y en Streamlit Cloud
    return pd.read_csv(Path(__file__).parent / "data" / "uplift_campaign.csv")


@st.cache_data
def entrenar_modelos(df: pd.DataFrame, test_size: float, seed: int):
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["treatment_group"])
    test = test.copy()
    X_test = test[FEATURES]

    # Modelo de riesgo/churn (Paso 1) — para comparar contra el T-learner
    train_churn = train.copy()
    train_churn["churned"] = 1 - train_churn["responded_60d"]
    modelo_riesgo = LogisticRegression(max_iter=1000).fit(train_churn[FEATURES], train_churn["churned"])
    test["prob_churn"] = modelo_riesgo.predict_proba(X_test)[:, 1]

    # T-learner (Paso 2): un modelo por grupo
    train_t = train[train["treatment_group"] == "treatment"]
    train_c = train[train["treatment_group"] == "control"]
    modelo_tratado = LogisticRegression(max_iter=1000).fit(train_t[FEATURES], train_t["responded_60d"])
    modelo_control = LogisticRegression(max_iter=1000).fit(train_c[FEATURES], train_c["responded_60d"])

    test["p_tratado"] = modelo_tratado.predict_proba(X_test)[:, 1]
    test["p_control"] = modelo_control.predict_proba(X_test)[:, 1]
    test["uplift_estimado"] = test["p_tratado"] - test["p_control"]
    return train, test


def valor_de_la_politica(test_df, seleccionados):
    sel = test_df.loc[seleccionados]
    valor_tratado = sel.loc[sel["treatment_group"] == "treatment", "utilidad_neta_60d"].mean()
    valor_control = sel.loc[sel["treatment_group"] == "control", "margin_60d"].mean()
    return valor_tratado - valor_control, len(sel)


def curva_qini(test_df, score_col, steps=40):
    orden = test_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    es_tratado = (orden["treatment_group"] == "treatment").values
    respondio = orden["responded_60d"].values

    acumulado_tratados = np.cumsum(np.where(es_tratado, respondio, 0))
    acumulado_control = np.cumsum(np.where(~es_tratado, respondio, 0))
    n_tratados_acum = np.cumsum(es_tratado)
    n_control_acum = np.cumsum(~es_tratado)

    n = len(orden)
    xs, ys = [0.0], [0.0]
    for i in np.linspace(1, n, steps).astype(int):
        nt, nc = n_tratados_acum[i - 1], n_control_acum[i - 1]
        tasa_t = acumulado_tratados[i - 1] / nt if nt > 0 else 0
        tasa_c = acumulado_control[i - 1] / nc if nc > 0 else 0
        xs.append(i / n)
        ys.append((tasa_t - tasa_c) * n)
    return np.array(xs), np.array(ys)


def coeficiente_qini(xs, ys, n):
    area_modelo = _trapz(ys, xs)
    area_azar = _trapz(np.linspace(0, ys[-1], len(xs)), xs)
    return (area_modelo - area_azar) / n


@st.cache_data
def tabla_de_politicas(test_df: pd.DataFrame, score_col: str):
    ordenado = test_df.sort_values(score_col, ascending=False)
    filas = []
    for pct in range(10, 101, 10):
        n_c = int(len(test_df) * pct / 100)
        valor_cliente, n = valor_de_la_politica(test_df, ordenado.head(n_c).index)
        filas.append({
            "% contactado": pct,
            "Clientes contactados": n,
            "Valor neto por cliente ($)": round(valor_cliente, 2),
            "Valor neto total ($)": round(valor_cliente * n, 2),
        })
    return pd.DataFrame(filas)


# --- Título y carga de datos ---
st.title("☕ Laboratorio Final — Starbucks Rewards")
st.caption(
    "¿A qué porcentaje de clientes conviene enviarle el cupón de $5? "
    "Esta app ordena a los clientes por su uplift estimado y muestra cuánto dinero neto deja cada nivel de contacto."
)

df = load_data()
train, test = entrenar_modelos(df, test_size=0.30, seed=42)

# --- Controles interactivos ---
with st.sidebar:
    st.header("Parámetros")
    pct_contactar = st.slider(
        "% de la base a contactar", min_value=10, max_value=100, value=20, step=10,
        help="Se contacta a los clientes con mayor uplift estimado primero.",
    )
    criterio = st.selectbox(
        "Ordenar clientes por",
        options=["Uplift estimado (T-learner)", "Riesgo de churn"],
        help="Compara la estrategia de uplift contra targetear a los de mayor riesgo.",
    )
    mostrar_riesgo = st.checkbox("Mostrar curva del modelo de riesgo en el Qini", value=True)

score_col = "uplift_estimado" if criterio.startswith("Uplift") else "prob_churn"

# --- Métricas de la política elegida ---
tabla = tabla_de_politicas(test, score_col)
fila = tabla[tabla["% contactado"] == pct_contactar].iloc[0]
mejor = tabla.loc[tabla["Valor neto total ($)"].idxmax()]

c1, c2, c3 = st.columns(3)
c1.metric("Clientes a contactar", f"{int(fila['Clientes contactados']):,}")
c2.metric("Valor neto por cliente", f"${fila['Valor neto por cliente ($)']:+.2f}")
c3.metric(
    "Valor neto total", f"${fila['Valor neto total ($)']:+,.2f}",
    delta=f"{fila['Valor neto total ($)'] - mejor['Valor neto total ($)']:+,.2f} vs. el óptimo",
)
st.success(
    f"Con el criterio **{criterio}**, el valor total se maximiza contactando al "
    f"**{int(mejor['% contactado'])}%** de la base (${mejor['Valor neto total ($)']:,.2f})."
)

# --- Curva Qini ---
col_qini, col_tabla = st.columns([3, 2])

with col_qini:
    st.subheader("Curva Qini")
    xs_u, ys_u = curva_qini(test, "uplift_estimado")
    qini_u = coeficiente_qini(xs_u, ys_u, len(test))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs_u, ys_u, color=GREEN, linewidth=3, label=f"T-learner (Qini={qini_u:+.3f})")
    if mostrar_riesgo:
        xs_r, ys_r = curva_qini(test, "prob_churn")
        qini_r = coeficiente_qini(xs_r, ys_r, len(test))
        ax.plot(xs_r, ys_r, color=ORANGE, linewidth=3, label=f"Riesgo/churn (Qini={qini_r:+.3f})")
    ax.plot([0, 1], [0, ys_u[-1]], color=GRAY, linestyle="--", label="Al azar")
    ax.axvline(pct_contactar / 100, color="black", linestyle=":", linewidth=1.5, label=f"Su elección: {pct_contactar}%")
    ax.set_xlabel("% de la base contactada (de prueba)")
    ax.set_ylabel("Respuestas incrementales acumuladas")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Mientras más arriba de la línea gris esté la curva, mejor ordena el modelo a quién vale la pena contactar.")

# --- Tabla de políticas ---
with col_tabla:
    st.subheader("Tabla de políticas")

    def resaltar(row):
        estilo = "background-color: #d4edda; font-weight: bold" if row["% contactado"] == pct_contactar else ""
        return [estilo] * len(row)

    st.dataframe(
        tabla.style.apply(resaltar, axis=1).format({
            "Clientes contactados": "{:,}",
            "Valor neto por cliente ($)": "${:+.2f}",
            "Valor neto total ($)": "${:+,.2f}",
        }),
        hide_index=True, width="stretch",
    )
    st.bar_chart(tabla.set_index("% contactado")["Valor neto total ($)"], color=GREEN)
    st.caption(
        "Valor neto = utilidad de los tratados (margen − cupón − envío) menos el margen del control "
        "en ese mismo grupo de clientes."
    )
