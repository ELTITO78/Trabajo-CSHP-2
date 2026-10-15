"""Figuras del analisis exploratorio, el entrenamiento y la evaluacion."""

import matplotlib

matplotlib.use("Agg")

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from src import config

sns.set_theme(style="whitegrid", context="notebook")

FIGURES_DIR = config.FIGURES_DIR
SAMPLES_DIR = config.SAMPLES_DIR


def _guardar(fig, nombre: str, directorio=None) -> str:
    destino = (directorio or FIGURES_DIR) / nombre
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(destino)


def plot_distribucion_clases(etiquetas: np.ndarray) -> str:
    cuentas = np.bincount(etiquetas.astype(int), minlength=config.NUM_CLASSES)
    total = cuentas.sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    barras = ax.bar(config.CLASS_NAMES_ES, cuentas, color=["#4c72b0", "#c44e52"])
    for barra, cuenta in zip(barras, cuentas):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            cuenta + total * 0.01,
            f"{cuenta}\n({cuenta / total:.1%})",
            ha="center",
            va="bottom",
        )
    ax.set_title("Distribucion de clases del corpus")
    ax.set_ylabel("Numero de mensajes")
    return _guardar(fig, "01_distribucion_clases.png")


def plot_longitud_mensajes(textos_limpios: list[str], y: np.ndarray) -> str:
    longitudes = np.array([len(t.split()) for t in textos_limpios])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.hist(longitudes, bins=40, color="#4c72b0", edgecolor="white")
    ax1.set_title("Longitud de los mensajes (palabras)")
    ax1.set_xlabel("Palabras")
    ax1.set_ylabel("Mensajes")
    ax1.axvline(config.MAX_LEN, color="#c44e52", linestyle="--", label=f"MAX_LEN={config.MAX_LEN}")
    ax1.legend()

    datos = [longitudes[y == clase] for clase in range(config.NUM_CLASSES)]
    ax2.boxplot(datos, tick_labels=config.CLASS_NAMES_ES)
    ax2.set_title("Longitud segun clase")
    ax2.set_ylabel("Palabras")
    fig.tight_layout()
    return _guardar(fig, "02_longitud_mensajes.png")


def plot_palabras_frecuentes(textos_limpios: list[str], y: np.ndarray, n: int = 15) -> str:
    """Palabras mas frecuentes en ham frente a spam (EDA)."""
    fig, axes = plt.subplots(1, config.NUM_CLASSES, figsize=(14, 4.5))
    for clase in range(config.NUM_CLASSES):
        subset = [t for t, etiqueta in zip(textos_limpios, y) if etiqueta == clase]
        palabras = [p for t in subset for p in t.split()]
        conteo = Counter(palabras).most_common(n)
        palabras_top = [p for p, _ in conteo][::-1]
        valores = [v for _, v in conteo][::-1]
        axes[clase].barh(palabras_top, valores, color=["#4c72b0", "#c44e52"][clase])
        axes[clase].set_title(f"Palabras mas frecuentes: {config.CLASS_NAMES_ES[clase]}")
        axes[clase].set_xlabel("Apariciones")
    fig.tight_layout()
    return _guardar(fig, "03_palabras_frecuentes.png")


def plot_ejemplos_clase(textos: list[str], y: np.ndarray, n: int = 4) -> str:
    """Muestra un fragmento de mensajes reales de cada clase."""
    fig, axes = plt.subplots(1, 2 * n, figsize=(3.2 * 2 * n, 4))
    for clase in range(config.NUM_CLASSES):
        indices = np.where(y == clase)[0][:n]
        for k, idx in enumerate(indices):
            ax = axes[clase * n + k]
            ax.axis("off")
            ax.set_title(
                f"[{config.CLASS_NAMES_ES[clase]}]",
                fontsize=9,
                color="#4c72b0" if clase == 0 else "#c44e52",
            )
            ax.text(
                0.0,
                0.95,
                textos[idx][:120].replace("\n", " "),
                fontsize=7,
                va="top",
                wrap=True,
            )
    fig.suptitle("Ejemplos reales del corpus")
    fig.tight_layout()
    return _guardar(fig, "04_ejemplos_clase.png")


def plot_curvas_entrenamiento(history: dict) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["loss"], label="Entrenamiento", color="#4c72b0")
    ax.plot(history["val_loss"], label="Validacion", color="#c44e52")
    ax.set_title("Curva de loss por epoca")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Binary crossentropy")
    ax.legend()
    fig.tight_layout()
    return _guardar(fig, "05_curvas_entrenamiento.png")


def plot_matriz_confusion(cm, nombres) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nombres).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title("Matriz de confusion (prueba)")
    fig.tight_layout()
    return _guardar(fig, "06_matriz_confusion.png")


def plot_matriz_confusion_normalizada(cm, nombres) -> str:
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=nombres).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format=".2f"
    )
    ax.set_title("Matriz de confusion normalizada por fila")
    fig.tight_layout()
    return _guardar(fig, "07_matriz_confusion_normalizada.png")


def plot_curva_roc(y_true, probabilidades) -> str:
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_true, probabilidades)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="#c44e52", label=f"LSTM (AUC = {auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
    ax.set_xlabel("Falsos positivos")
    ax.set_ylabel("Verdaderos positivos")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _guardar(fig, "08_curva_roc.png")


def grafico_ejemplos_predicciones(
    textos: list[str], y_true: np.ndarray, probabilidades: np.ndarray
) -> str:
    """Mosaico con las predicciones mas relevantes (aciertos y fallos)."""
    predicciones = (probabilidades >= 0.5).astype(int)
    aciertos = np.where(predicciones == y_true)[0]
    fallos = np.where(predicciones != y_true)[0]

    seleccionados = list(aciertos[:5]) + list(fallos[:5])
    fig, axes = plt.subplots(2, 5, figsize=(15, 5))
    for ax, idx in zip(axes.ravel(), seleccionados):
        ax.axis("off")
        ok = predicciones[idx] == y_true[idx]
        ax.set_title(
            f"{'OK' if ok else 'FALLO'}\nreal: {config.CLASS_NAMES_ES[int(y_true[idx])]} | "
            f"pred: {config.CLASS_NAMES_ES[int(predicciones[idx])]}",
            fontsize=8,
            color="#2ca02c" if ok else "#d62728",
        )
        ax.text(0.0, 0.9, textos[idx][:110], fontsize=7, va="top", wrap=True)
    fig.suptitle("Predicciones de prueba: aciertos (arriba) y fallos (abajo)")
    fig.tight_layout()
    return _guardar(fig, "predicciones_ejemplo.png", SAMPLES_DIR)