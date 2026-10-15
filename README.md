# Detector de mensajes Spam con red neuronal recurrente (LSTM)

Proyecto de Deep Learning que clasifica mensajes SMS en **legítimo (`ham`)** o
**spam`**, usando una red **LSTM** de Keras y una interfaz de **Streamlit**.

- **Dataset:** SMS Spam Collection v1 (UCI) — 5.574 mensajes en inglés.
- **Tarea:** clasificación binaria de texto.
- **Resultado:** **98.69 % de accuracy** y **0.9964 de ROC-AUC** en el conjunto
  de prueba (837 mensajes que el modelo nunca vio).

---

## 1. Estructura del proyecto

```
.
├── app.py                       # Aplicación Streamlit
├── train.py                     # Entrenamiento completo y guardado de artefactos
├── predict.py                   # Predicción desde la terminal
├── requirements.txt
├── README.md
├── data/                        # Corpus (se descarga solo, no se versiona)
├── notebooks/
│   └── experimentacion_rnn_spam.ipynb
├── src/
│   ├── config.py                # Rutas, hiperparámetros y constantes
│   ├── data.py                  # Descarga, limpieza, tokenizado y división
│   ├── model.py                 # Arquitectura LSTM y callbacks
│   └── visualization.py         # Figuras del EDA, entrenamiento y evaluación
└── artifacts/
    ├── spam_lstm.keras          # Modelo entrenado
    ├── tokenizer.json           # Vocabulario usado por el modelo
    ├── metrics.json             # Métricas de la ejecución de referencia
    ├── history.json             # Historial de entrenamiento
    ├── figures/                 # 8 figuras
    └── samples/                 # 10 predicciones de prueba (PNG + CSV)
```

---

## 2. Instalación

Se recomienda Python 3.10 a 3.12.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

No hace falta descargar nada: `train.py` baja el corpus de la UCI la primera vez.

---

## 3. Ejecución

### 3.1. Entrenar el modelo

```bash
python train.py
```

Opciones disponibles:

```bash
python train.py --epochs 30        # más épocas
python train.py --batch-size 32    # lotes más pequeños
python train.py --max-len 200      # secuencias más largas
python train.py --force-download   # vuelve a bajar el corpus
```

Tarda unos **1-2 minutos en CPU**. El script genera todas las figuras y métricas
en `artifacts/`.

### 3.2. Abrir la aplicación Streamlit

```bash
python -m streamlit run app.py
```

Se abre en `http://localhost:8501`. Puedes escribir un SMS y obtener la
probabilidad de que sea spam, con botones de mensajes de ejemplo.

### 3.3. Predicción desde la terminal

```bash
python predict.py "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121"
python predict.py --archivo mensajes.txt
```

---

## 4. Dataset

**SMS Spam Collection v1**, recopilado por la Universidad de California (Irvine)
para investigar el filtrado de spam SMS.

| | |
|---|---|
| Mensajes | 5.574 |
| Legítimos (`ham`) | 4.827 (86.6 %) |
| Spam | 747 (13.4 %) |
| Formato | TSV: `etiqueta<TAB>texto` |
| Fuente | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |

El corpus está **muy desbalanceado**. Esto importa: un modelo que prediga
siempre "ham" ya acertaría el 86.6 %, así que la accuracy por sí sola no
demuestra que el modelo funcione. Por eso evaluamos también recall y F1 de la
clase `spam`.

---

## 5. Análisis exploratorio

| Figura | Descripción |
|---|---|
| `artifacts/figures/01_distribucion_clases.png` | 4.827 ham frente a 747 spam |
| `artifacts/figures/02_longitud_mensajes.png` | Histograma y boxplot de longitud |
| `artifacts/figures/03_palabras_frecuentes.png` | 15 palabras más frecuentes por clase |
| `artifacts/figures/04_ejemplos_clase.png` | Mensajes reales de ham y spam |

Hallazgos principales:

- **Los spams son más largos**: 23.9 palabras de media frente a 14.3 de los ham.
  La longitud ya es una señal fuerte.
- **El vocabulario distingue mucho**: "free", "win", "call", "txt", "claim"
  dominan en los spam; en los ham aparecen sobre todo palabras de conversación
  ("you", "how", "ok", "call me later").
- Los mensajes están escritos en clave, sin acentos ni mayúsculas.

---

## 6. Limpieza y preprocesamiento

1. Todo el texto pasa a **minúsculas**.
2. Se sustituyen las **URLs**, **correos**, **cantidades de dinero** y
   **números de teléfono** por marcadores (`<url>`, `<email>`, `<dinero>`,
   `<num>`). Así el modelo generaliza a números que no ha visto.
3. Se eliminan los signos de puntuación y se colapsan los espacios.
4. Un `Tokenizer` de Keras asigna un índice a cada palabra (`<OOV>` para las
   desconocidas), con un vocabulario máximo de 10.000 términos.
5. Las secuencias se **rellenan por la izquierda** hasta `MAX_LEN = 160` y se
   recortan por la derecha. El relleno por la izquierda es el que espera una
   LSTM: los últimos tokens del mensaje son siempre los más recientes.

---

## 7. División de los datos

División **estratificada** 70 / 15 / 15 que conserva la proporción de spam:

| Conjunto | Mensajes | Spam |
|---|---|---|
| Entrenamiento | 3.901 | 523 |
| Validación | 836 | 112 |
| Prueba | 837 | 112 |

El `Tokenizer` se ajusta **solo con el conjunto de entrenamiento**, de modo que
ninguna información del conjunto de prueba llega al preprocesamiento.

---

## 8. Arquitectura del modelo

```
Entrada: 160 índices de palabra
  ↓
Embedding(128)          # 818.048 parámetros
  ↓
SpatialDropout1D(0.2)   # regularización sobre las embeddings
  ↓
LSTM(128)               # 131.584 parámetros
  ↓
Dense(64, relu)
  ↓
Dropout(0.5)
  ↓
Dense(1, sigmoid)       # probabilidad de spam
```

**Total: 957.953 parámetros entrenables.**

Una red recurrente no "entiende" palabras, solo números: los índices son
arbitrarios y no capturan relaciones semánticas. La capa `Embedding` resuelve esto
aprendiendo una representación vectorial de 128 dimensiones para cada palabra
durante el propio entrenamiento.

| Hiperparámetro | Valor |
|---|---|
| Longitud de secuencia (`MAX_LEN`) | 160 |
| Vocabulario máximo | 10.000 |
| Dimensión del embedding | 128 |
| Unidades LSTM | 128 |
| Dropout | 0.5 |
| Optimizador | Adam, `lr = 1e-3` |
| Pérdida | `binary_crossentropy` |
| Batch size | 64 |
| Épocas (máx.) | 20 |

**Callbacks:** `ModelCheckpoint` y `EarlyStopping(patience=4)` sobre `val_loss`,
más `ReduceLROnPlateau(factor=0.5)` para bajar la tasa de aprendizaje cuando la
validación deja de mejorar.

---

## 9. Resultados

Ejecución de referencia (`artifacts/metrics.json`):

| Métrica | Valor |
|---|---|
| **Accuracy** | **98.69 %** |
| **Precision (spam)** | 95.50 % |
| **Recall (spam)** | 94.64 % |
| **F1 (spam)** | 95.07 % |
| **ROC-AUC** | **0.9964** |
| Mensajes de prueba | 837 |

Informe completo de scikit-learn:

```
              precision    recall  f1-score   support

       ham      0.99      0.99      0.99       725
      spam      0.96      0.95      0.95       112

    accuracy                           0.99       837
   macro avg       0.97      0.97      0.97       837
weighted avg       0.99      0.99      0.99       837
```

### Matriz de confusión

| | Predice ham | Predice spam |
|---|---|---|
| **Real ham** | 720 | 5 |
| **Real spam** | 6 | 106 |

De los 112 mensajes de spam reales, el modelo detecta **106**; solo 6 se le
escapan.

### Comparación con líneas base

| Modelo | Accuracy |
|---|---|
| Predecir siempre "ham" | 86.6 % |
| **LSTM de este proyecto** | **98.69 %** |

### Curvas y figuras

| Artefacto | Descripción |
|---|---|
| `artifacts/figures/05_curvas_entrenamiento.png` | Loss de entrenamiento y validación por época |
| `artifacts/figures/06_matriz_confusion.png` | Matriz de confusión |
| `artifacts/figures/07_matriz_confusion_normalizada.png` | Matriz normalizada por fila |
| `artifacts/figures/08_curva_roc.png` | Curva ROC (AUC = 0.9964) |

### 10 predicciones de prueba

De las 10 predicciones de ejemplo guardadas en
`artifacts/samples/predicciones_ejemplo.png` y `.csv`, **9 fueron correctas**.

El único fallo es un **spam no detectado** (falso negativo): el mensaje
`FreeMsg>FAV XMAS TONES!Reply REAL` se clasificaba como legítimo con solo un
4.6 % de probabilidad. Es un caso difícil: el mensaje es muy corto y está escrito
con una abreviatura (`FAV`) que el preprocesamiento reduce a un token poco
informativo. Aun así, el modelo acierta 9 de 10 spams en el conjunto completo.

---

## 10. Ejecución en Google Colab

Todo lo que se hizo en el notebook está disponible en
`notebooks/experimentacion_rnn_spam.ipynb`, que es autocontenido: descarga el
corpus, hace el EDA, preprocesa, entrena, evalúa y muestra ejemplos.

```python
# Si necesitas instalar las dependencias en Colab
!pip install -q scikit-learn seaborn
```

---

## 11. Guardado del modelo

`train.py` guarda:

- `artifacts/spam_lstm.keras` — el modelo con los pesos de la mejor época
  (la de menor `val_loss`).
- `artifacts/tokenizer.json` — el vocabulario, imprescindible para que la app
  interprete los índices igual que durante el entrenamiento.
- `artifacts/history.json` y `artifacts/metrics.json`.

Ambos son los mismos archivos que carga `app.py`, así que **la predicción que
ves en la web es exactamente la que generó las métricas de la sección 9**.

Para regenerar todo:

```bash
python train.py
```

---

## 12. Limitaciones

- El corpus es pequeño (5.574 mensajes) y está en inglés de principios de los
  2000, escrito en clave. Con otros idiomas haría falta reentrenar.
- El dataset es antiguo: los patrones de spam actuales (WhatsApp, phishing
  moderno) no aparecen en él.
- El modelo no explica sus decisiones: solo devuelve una probabilidad.