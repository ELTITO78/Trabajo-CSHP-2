"""Predice si un mensaje SMS es spam desde la terminal.

Uso:
    python predict.py "Free entry! Win a prize, txt to 87121"
    python predict.py --archivo mensajes.txt
"""

import argparse
import sys
from pathlib import Path

from tensorflow import keras

from src import config
from src import data as data_utils

UMBRAL = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clasificador de spam con LSTM.")
    parser.add_argument("texto", nargs="?", help="Mensaje a clasificar.")
    parser.add_argument("--archivo", type=Path, help="Fichero con un mensaje por linea.")
    return parser.parse_args()


def mostrar(mensaje: str, probabilidad: float) -> None:
    etiqueta = "SPAM" if probabilidad >= UMBRAL else "HAM (legitimo)"
    print(f"\nMensaje:  {mensaje}")
    print(f"Prediccion: {etiqueta}")
    print(f"Probabilidad de spam: {probabilidad:.2%}")


def main() -> None:
    args = parse_args()

    if not config.MODEL_PATH.exists():
        sys.exit("No existe el modelo. Ejecuta primero `python train.py`.")

    modelo = keras.models.load_model(config.MODEL_PATH)
    tokenizer = data_utils.load_tokenizer()

    if args.archivo:
        lineas = [
            linea.strip()
            for linea in args.archivo.read_text(encoding="utf-8").splitlines()
            if linea.strip()
        ]
        print(f"Clasificando {len(lineas)} mensajes de {args.archivo}...")
        for linea in lineas:
            mostrar(linea, data_utils.predict_text(modelo, tokenizer, linea))
        return

    if not args.texto:
        sys.exit("Indica un mensaje entre comillas o usa --archivo.")

    mostrar(args.texto, data_utils.predict_text(modelo, tokenizer, args.texto))


if __name__ == "__main__":
    main()