"""Aplicacion Streamlit para clasificar mensajes SMS con el modelo LSTM."""

import json

import streamlit as st
from tensorflow import keras

from src import config
from src import data as data_utils

st.set_page_config(
    page_title="Detector de Spam con LSTM",
    page_icon="📱",
    layout="centered",
)

UMBRAL = 0.5

EJEMPLOS = {
    "✅ Mensaje legitimo": "Hey how are you? Are we still meeting for lunch at 1pm?",
    "📢 Spam de sorteo": "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121 to receive entry",
    "⚠️ Spam de premio urgente": "URGENT! You have won a 2000 pound cash prize. Call 0871-872-9758 to claim your reward",
    "👋 Texto normal": "Ok lar... Joking wif u oni...",
}


@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar_modelo():
    modelo = keras.models.load_model(config.MODEL_PATH)
    tokenizer = data_utils.load_tokenizer()
    return modelo, tokenizer


def main() -> None:
    st.title("📱 Detector de mensajes Spam con LSTM")
    st.caption(
        "Red neuronal recurrente LSTM entrenada con el corpus "
        "*SMS Spam Collection v1* de la Universidad de California (UCI)."
    )

    if not config.MODEL_PATH.exists():
        st.error("No se encuentra el modelo entrenado. Ejecuta `python train.py`.")
        st.stop()

    modelo, tokenizer = cargar_modelo()

    if config.METRICS_PATH.exists():
        metricas = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
        prueba = metricas["prueba"]
        st.sidebar.header("📊 Rendimiento del modelo")
        st.sidebar.metric("Accuracy en prueba", f"{prueba['accuracy']:.2%}")
        st.sidebar.metric("F1 (spam)", f"{prueba['f1']:.2%}")
        st.sidebar.metric("ROC-AUC", f"{prueba['roc_auc']:.4f}")
        st.sidebar.caption(
            f"Evaluado sobre {metricas['divisiones']['prueba']:,} mensajes "
            "que el modelo nunca vio durante el entrenamiento.".replace(",", ".")
        )
        st.sidebar.divider()
        st.sidebar.caption(
            f"Red: LSTM de {metricas['hiperparametros']['lstm_units']} unidades "
            f"· {metricas['parametros_entrenables']:,} parámetros".replace(",", ".")
        )

    st.subheader("✍️ Escribe un mensaje")
    texto = st.text_area(
        "Texto del SMS",
        height=120,
        placeholder="Pega aquí el mensaje que quieres clasificar...",
        key="mensaje",
    )

    if st.button("🔍 Clasificar mensaje", type="primary"):
        if not texto.strip():
            st.warning("Escribe un mensaje antes de clasificar.")
        else:
            probabilidad = data_utils.predict_text(modelo, tokenizer, texto)
            es_spam = probabilidad >= UMBRAL

            if es_spam:
                st.error(f"🚨 SPAM detectado ({probabilidad:.2%} de probabilidad)")
            else:
                st.success(f"✅ Mensaje legítimo ({probabilidad:.2%} de probabilidad)")

            st.progress(probabilidad)
            c1, c2 = st.columns(2)
            c1.metric("Probabilidad de spam", f"{probabilidad:.2%}")
            c2.metric(
                "Probabilidad de legítimo", f"{1 - probabilidad:.2%}"
            )

            limpio = data_utils.clean_text([texto])[0]
            st.markdown("**Texto tras el preprocesamiento**")
            st.code(limpio or "(vacío)", language=None)

    st.divider()
    st.subheader("💬 ¿No tienes un mensaje a mano?")
    for etiqueta, ejemplo in EJEMPLOS.items():
        if st.button(etiqueta, key=etiqueta, use_container_width=True):
            st.session_state["mensaje"] = ejemplo
            st.rerun()

    st.caption(
        "El modelo se entrenó con `python train.py`. "
        "La app carga ese mismo modelo: lo que ves aqui es exactamente lo evaluado."
    )


if __name__ == "__main__":
    main()