# 🏛️ SRI — Reuters Corpus Benchmark

Sistema de Recuperación de Información (SRI) implementado sobre el corpus Reuters ModApte, que permite comparar y evaluar cuatro modelos de recuperación: **Jaccard**, **TF-IDF**, **BM25** y **Embeddings Semánticos**. El proyecto ofrece dos modos de uso: una interfaz de consola ligera y una aplicación web interactiva con benchmarking integrado.

---

## 📁 Estructura del Proyecto

```
Proyecto-IB/
├── app.py                    # Interfaz gráfica interactiva y Benchmarking (Streamlit)
├── main.py                   # Interfaz ligera por consola (CLI)
├── informe_tecnico.ipynb     # Cuaderno de experimentación y análisis de resultados
├── data/
│   └── ModApte_train.csv     # Corpus original de noticias financieras (Requerido)
└── src/
    ├── preprocessor.py       # Limpieza, tokenización y stemming (NLTK)
    ├── indexer.py            # Construcción del índice invertido
    ├── evaluator.py          # Cálculo de métricas (Precision, Recall, MAP)
    ├── qrels.py              # Gestor del Ground Truth y alineación de documentos
    └── models/
        ├── base_model.py     # Clase abstracta base
        ├── jaccard.py        # Modelo Jaccard
        ├── tfidf.py          # Modelo TF-IDF
        ├── bm25.py           # Modelo Probabilístico BM25
        └── embeddings.py     # Modelo Semántico Vectorial
```

---

## ⚙️ Requisitos e Instalación

### Requisitos del sistema

- Python **3.9+**
- pip

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

Si no existe `requirements.txt`, instala manualmente los paquetes necesarios:

```bash
pip install pandas nltk streamlit scikit-learn rank_bm25 sentence-transformers faiss-cpu
```

### Descarga de recursos NLTK

El preprocesador depende de stopwords y el tokenizador de NLTK. Ejecuta esto una sola vez en Python:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Corpus de datos

Coloca el archivo `ModApte_train.csv` dentro de la carpeta `data/` antes de ejecutar cualquier interfaz. El archivo debe contener al menos las columnas `text` y `topics`.

```
Proyecto-IB/
└── data/
    └── ModApte_train.csv   ← Requerido
```

---

## 🖥️ Interfaz de Consola — `main.py`

Modo ligero pensado para realizar búsquedas rápidas desde la terminal sin necesidad de levantar un servidor web. Carga los tres modelos léxicos (Jaccard, TF-IDF y BM25) y permite realizar consultas de forma interactiva.

### Ejecución

```bash
python main.py
```

### Uso

Al iniciar, el sistema carga el corpus y construye los índices automáticamente. Luego presenta un menú para seleccionar el modelo y escribir una consulta:

```
1. Jaccard  2. TF-IDF  3. BM25  4. Salir
Modelo: 2
Consulta: crude oil prices OPEC

Modelo: TF-IDF
Top 5 resultados:
[1] Document_42 | Score: 0.8731 | Saudi Arabia announced today a reduction in...
[2] Document_7  | Score: 0.8102 | OPEC ministers convened in Vienna to discuss...
...
```

---

## 🌐 Interfaz Gráfica — `app.py`

Aplicación web completa construida con **Streamlit** que incluye búsqueda interactiva para los cuatro modelos y un panel de benchmarking automatizado con métricas de evaluación.

### Ejecución

```bash
streamlit run app.py
```

Esto abrirá automáticamente el navegador en `http://localhost:8501`.

### Pestañas disponibles

| Pestaña | Descripción |
|---|---|
| 🔍 Modelo Jaccard | Búsqueda basada en similitud de conjuntos binarios |
| 📈 Modelo TF-IDF | Búsqueda por ponderación estadística con similitud coseno |
| 🔥 Modelo BM25 | Búsqueda probabilística con saturación de términos |
| 🧠 Embeddings Semánticos | Búsqueda densa con vectores de oraciones |
| 📊 Panel Comparativo | Benchmark sobre 10 consultas predefinidas con Precision, Recall y MAP |

En el **Panel Comparativo**, al presionar *Ejecutar Benchmark*, el sistema corre automáticamente las 10 consultas de prueba, genera el Ground Truth desde las categorías del corpus y calcula las métricas para cada modelo, mostrando:

- Tabla de resultados globales (MAP, Precision@5, Recall@5)
- Gráfico de barras comparativo
- Matriz de coincidencias (overlap) entre modelos

---

## 📓 Cuaderno de Experimentación — `informe_tecnico.ipynb`

Contiene el análisis exploratorio del corpus, los experimentos de evaluación y la discusión de resultados. Se recomienda ejecutarlo con **Jupyter Lab** o **Jupyter Notebook**.

### Ejecución

```bash
jupyter lab informe_tecnico.ipynb
# o bien
jupyter notebook informe_tecnico.ipynb
```

El cuaderno está organizado en las siguientes secciones:

1. Carga y exploración del corpus Reuters
2. Preprocesamiento y construcción del índice invertido
3. Implementación y evaluación de cada modelo
4. Comparación de métricas (Precision, Recall, MAP)
5. Análisis cualitativo de resultados y conclusiones

---

## 🤖 Descripción de los Modelos

### 🔍 Jaccard (`src/models/jaccard.py`)

Modelo de recuperación binaria basado en similitud de conjuntos. Representa cada documento como un conjunto de términos únicos (presencia/ausencia) y calcula la similitud como la razón entre la intersección y la unión de los conjuntos de términos del documento y la consulta.

- **Ventaja:** Simple, sin parámetros, útil como línea base.
- **Limitación:** No considera la frecuencia de los términos; es sensible a stopwords mal filtradas y a vocabularios con alto solapamiento casual.

### 📈 TF-IDF (`src/models/tfidf.py`)

Modelo vectorial que pondera cada término según su frecuencia en el documento (TF) y su rareza en el corpus (IDF). Las consultas y documentos se proyectan en un espacio vectorial y la relevancia se mide con similitud coseno.

- **Ventaja:** Captura la importancia relativa de los términos; robusto ante documentos de diferente longitud gracias a la normalización coseno.
- **Limitación:** No modela la saturación de términos; asume independencia entre palabras.

### 🔥 BM25 (`src/models/bm25.py`)

Modelo probabilístico de última generación que extiende TF-IDF incorporando saturación de términos (los conteos altos aportan decrecientemente) y normalización por longitud de documento respecto al promedio del corpus. Controlado por los parámetros `k1` y `b`.

- **Ventaja:** Mejor rendimiento empírico en recuperación de texto libre que TF-IDF clásico; estándar de facto en buscadores modernos.
- **Limitación:** Sigue siendo un modelo léxico; falla ante sinónimos o paráfrasis.

### 🧠 Embeddings Semánticos (`src/models/embeddings.py`)

Modelo de recuperación densa basado en Sentence Transformers. Codifica documentos y consultas en vectores densos de alta dimensión que capturan el significado semántico más allá de la coincidencia exacta de tokens. Utiliza una base vectorial (FAISS o similar) para búsqueda eficiente por similitud.

- **Ventaja:** Maneja sinónimos, paráfrasis y lenguaje natural descriptivo; útil para consultas cortas o abstractas.
- **Limitación:** Puede perder especificidad en entidades concretas o términos técnicos muy específicos; mayor costo computacional.

---

## 📏 Métricas de Evaluación

El módulo `src/evaluator.py` calcula las siguientes métricas comparando los documentos recuperados con el Ground Truth generado desde las etiquetas de categoría del corpus:

| Métrica | Descripción |
|---|---|
| **Precision@K** | Fracción de documentos recuperados en el top-K que son relevantes |
| **Recall@K** | Fracción de documentos relevantes que fueron recuperados en el top-K |
| **MAP** | Mean Average Precision — promedio de las AP de todas las consultas |

El Ground Truth se construye automáticamente en `src/qrels.py` filtrando el corpus por la categoría correspondiente a cada consulta (ej. `"crude"` para la consulta sobre petróleo).

---

## 💡 Notas adicionales

- El sistema usa caché de Streamlit (`@st.cache_resource`) para que el corpus y los índices se carguen una sola vez por sesión.
- El preprocesamiento aplica lowercasing, eliminación de caracteres no alfanuméricos, remoción de stopwords y stemming Snowball en inglés.
- Para agregar un nuevo modelo, extiende la clase base `src/models/base_model.py` e implementa los métodos `index()` y `search(query, k)`.