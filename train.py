"""Entrena el clasificador de spam con LSTM y guarda todos los artefactos.

Uso:
    python train.py
    python train.py --epochs 30 --batch-size 32
"""

import argparse
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tensorflow import keras

from src import config, data as data_utils, model as model_utils, visualization as viz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el modelo LSTM de spam.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--max-len", type=int, default=config.MAX_LEN)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--force-download", action="store_true", help="Vuelve a descargar el corpus."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("1/6  Preparando datos")
    print("=" * 70)
    datos = data_utils.prepare(force_download=args.force_download)

    x_train, y_train = datos["x_train"], datos["y_train"]
    x_val, y_val = datos["x_val"], datos["y_val"]
    x_test, y_test = datos["x_test"], datos["y_test"]
    textos = datos["mensajes"]
    textos_limpios = datos["mensajes_limpios"]
    etiquetas = datos["etiquetas"]
    tokenizer = datos["tokenizer"]

    print(f"Entrenamiento: {x_train.shape}  (spam: {int(y_train.sum())})")
    print(f"Validacion:    {x_val.shape}  (spam: {int(y_val.sum())})")
    print(f"Prueba:        {x_test.shape}  (spam: {int(y_test.sum())})")
    print(f"Vocabulario:   {len(tokenizer.word_index)} palabras")

    print("=" * 70)
    print("2/6  Analisis exploratorio")
    print("=" * 70)
    figuras = [
        viz.plot_distribucion_clases(etiquetas),
        viz.plot_longitud_mensajes(textos_limpios, etiquetas),
        viz.plot_palabras_frecuentes(textos_limpios, etiquetas),
        viz.plot_ejemplos_clase(textos, etiquetas),
    ]
    for figura in figuras:
        print(f"  -> {Path(figura).name}")

    print("=" * 70)
    print("3/6  Construyendo el modelo")
    print("=" * 70)
    vocab_size = min(len(tokenizer.word_index) + 1, config.MAX_VOCAB)
    modelo = model_utils.build_model(vocab_size, max_len=config.MAX_LEN)
    modelo.summary()
    modelo = model_utils.compile_model(modelo)
    callbacks = model_utils.build_callbacks

    print("=" * 70)
    print("4/6  Entrenando")
    print("=" * 70)
    inicio = time.time()
    with tempfile.TemporaryDirectory() as temporal:
        checkpoint_path = Path(temporal) / "checkpoint.keras"
        history = modelo.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks(checkpoint_path),
            verbose=2,
        ).history

        # El checkpoint se resuelve fuera del temporal para no borrarlo.
        mejor = keras.models.load_model(checkpoint_path)
        mejor.save(config.MODEL_PATH)

    minutos = (time.time() - inicio) / 60
    print(f"Tiempo de entrenamiento: {minutos:.1f} min")

    print("=" * 70)
    print("5/6  Evaluando")
    print("=" * 70)
    probabilidades = mejor.predict(x_test, verbose=0).ravel()
    predicciones = (probabilidades >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, predicciones)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predicciones, average="binary", zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilidades)
    cm = confusion_matrix(y_test, predicciones)
    informe = classification_report(
        y_test, predicciones, target_names=config.CLASS_NAMES_ES, zero_division=0
    )
    print(informe)
    print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
          f"Recall: {recall:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

    print("=" * 70)
    print("6/6  Guardando artefactos")
    print("=" * 70)

    viz.plot_curvas_entrenamiento(history)
    viz.plot_matriz_confusion(cm, config.CLASS_NAMES_ES)
    viz.plot_matriz_confusion_normalizada(cm, config.CLASS_NAMES_ES)
    viz.plot_curva_roc(y_test, probabilidades)

    # Ejemplos tomados del conjunto de prueba, no del corpus completo.
    idx_test = datos["idx_test"]
    seleccion = np.random.default_rng(args.seed).permutation(len(idx_test))[: config.NUM_EXAMPLES]
    ejemplos = [
        {
            "texto": textos[idx_test[i]],
            "real": config.CLASS_NAMES[int(y_test[i])],
            "predicho": config.CLASS_NAMES[int(predicciones[i])],
            "probabilidad_spam": float(probabilidades[i]),
            "acierto": bool(predicciones[i] == y_test[i]),
        }
        for i in seleccion
    ]
    _guardar_ejemplos(ejemplos)
    viz.grafico_ejemplos_predicciones(
        [e["texto"] for e in ejemplos],
        np.array([config.CLASS_NAMES.index(e["real"]) for e in ejemplos]),
        np.array([e["probabilidad_spam"] for e in ejemplos]),
    )

    tokenizer_json = tokenizer.to_json()
    config.TOKENIZER_PATH.write_text(tokenizer_json, encoding="utf-8")

    mejor_epoca = int(np.argmin(history["val_loss"])) + 1
    metricas = {
        "dataset": {
            "nombre": "SMS Spam Collection v1 (UCI)",
            "fuente": config.DATA_URL,
            "mensajes_totales": int(len(etiquetas)),
            "ham": int((etiquetas == 0).sum()),
            "spam": int((etiquetas == 1).sum()),
        },
        "hiperparametros": {
            "max_vocab": config.MAX_VOCAB,
            "max_len": config.MAX_LEN,
            "embedding_dim": config.EMBEDDING_DIM,
            "lstm_units": config.LSTM_UNITS,
            "dense_units": config.DENSE_UNITS,
            "dropout": config.DROPOUT,
            "batch_size": args.batch_size,
            "learning_rate": config.LEARNING_RATE,
            "seed": args.seed,
        },
        "divisiones": {
            "entrenamiento": int(len(x_train)),
            "validacion": int(len(x_val)),
            "prueba": int(len(x_test)),
            "spam_entrenamiento": int(y_train.sum()),
            "spam_validacion": int(y_val.sum()),
            "spam_prueba": int(y_test.sum()),
        },
        "prueba": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "informe": informe,
            "matriz_confusion": cm.tolist(),
        },
        "mejor_epoca": mejor_epoca,
        "epocas_entrenadas": len(history["loss"]),
        "parametros_entrenables": int(modelo.count_params()),
        "minutos_entrenamiento": round(minutos, 2),
        "ejemplos_predicciones": ejemplos,
    }
    config.METRICS_PATH.write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    config.HISTORY_PATH.write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    aciertos = sum(e["acierto"] for e in ejemplos)
    print(f"\nModelo:    {config.MODEL_PATH}")
    print(f"Metricas:  {config.METRICS_PATH}")
    print(f"Ejemplos:  {aciertos}/{len(ejemplos)} correctos")
    print("\nListo. Ejecuta `python -m streamlit run app.py` para usar el clasificador.")


def _guardar_ejemplos(ejemplos: list[dict]) -> None:
    import csv

    destino = config.SAMPLES_DIR / "predicciones_ejemplo.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=["texto", "real", "predicho", "probabilidad_spam", "acierto"],
        )
        escritor.writeheader()
        for ejemplo in ejemplos:
            fila = dict(ejemplo)
            fila["probabilidad_spam"] = round(fila["probabilidad_spam"], 4)
            escritor.writerow(fila)


if __name__ == "__main__":
    main()