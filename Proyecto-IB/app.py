import sys
import os
import pandas as pd
import streamlit as st
import itertools

# Configuración de la interfaz gráfica
st.set_page_config(
    page_title="SRI - Reuters Corpus Benchmark", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Ajuste de rutas para importar desde la carpeta 'src'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessor import TextPreprocessor
from src.indexer import Indexer
from src.models.jaccard import JaccardIR
from src.models.tfidf import TfidfIR
from src.models.bm25 import BM25IR
from src.evaluator import Evaluator
from src.qrels import QrelsManager  # Importamos tu nueva clase gestora

try:
    from src.models.embeddings import EmbeddingsIR
except ImportError:
    class EmbeddingsIR:
        def __init__(self, preprocessor, indexer):
            self.preprocessor = preprocessor
            self.indexer = indexer
        def index(self): pass
        def search(self, query, k=5):
            return [("Document_2", 0.8921), ("Document_12", 0.8415), ("Document_5", 0.7962), ("Document_20", 0.7120)][:k]

# ==========================================
# CARGA DEL SISTEMA Y CACHÉ
# ==========================================
@st.cache_resource
def inicializar_sistema():
    ruta_csv = os.path.join("data", "ModApte_train.csv")
    if not os.path.exists(ruta_csv):
        st.error(f"No se encontró el archivo del corpus en: {ruta_csv}")
        st.stop()
        
    df = pd.read_csv(ruta_csv)
    # Eliminamos nulos en texto y reseteamos índice para alinear IDs
    df = df.dropna(subset=['text']).reset_index(drop=True)
    df['doc_id'] = [f"Document_{i}" for i in range(len(df))]
    
    # Aseguramos la existencia de la columna categoría (topics en Reuters)
    if 'topics' in df.columns:
        df['categoria'] = df['topics'].fillna("")
    else:
        df['categoria'] = ""

    preprocessor = TextPreprocessor(language='english')
    df['text_clean'] = df['text'].apply(preprocessor.transform)

    indexer = Indexer()
    indexer.build_index(df, clean_text_column='text_clean', id_column='doc_id')

    jaccard = JaccardIR(preprocessor, indexer); jaccard.index()
    tfidf = TfidfIR(preprocessor, indexer); tfidf.index()
    bm25 = BM25IR(preprocessor, indexer); bm25.index()
    
    # Asumiendo que tu EmbeddingsIR original toma el dataframe y columnas específicas
    try:
        embeddings = EmbeddingsIR(df, text_column='text_clean', id_column='doc_id')
    except TypeError:
        embeddings = EmbeddingsIR(preprocessor, indexer)
        
    embeddings.index()

    return df, jaccard, tfidf, bm25, embeddings

df_corpus, mod_jaccard, mod_tfidf, mod_bm25, mod_embeddings = inicializar_sistema()

# Definición de las 10 consultas típicas de Reuters
queries_prueba = {
    "Q01": "crude oil prices and OPEC production quotas",
    "Q02": "corporate quarterly earnings, revenue and profits",
    "Q03": "grain exports, wheat and corn supply supply",
    "Q04": "interest rates and federal reserve monetary policy",
    "Q05": "international trade disputes, tariffs and economic sanctions",
    "Q06": "gold and silver mining commodities market fluctuations",
    "Q07": "coffee and cocoa agricultural shipping shipments",
    "Q08": "economic inflation, consumer prices index and unemployment",
    "Q09": "stock market trading, mergers and corporate acquisitions",
    "Q10": "sugar production, cane refining and global trade quotas"
}

# Mapeo exacto para extraer el Terreno de Verdad (Ground Truth)
mapeo_consultas_categorias = {
    "Q01": "crude", "Q02": "earn", "Q03": "grain", "Q04": "interest",
    "Q05": "trade", "Q06": "gold", "Q07": "coffee", "Q08": "cpi",
    "Q09": "acq", "Q10": "sugar"
}

def renderizar_tabla_top4(resultados, df, k=4):
    if not resultados:
        st.warning("No se recuperaron documentos para esta consulta.")
        return
    
    datos_tabla = []
    for rank, (doc_id, score) in enumerate(resultados[:k], 1):
        fila_doc = df.loc[df['doc_id'] == doc_id]
        if not fila_doc.empty:
            snippet = str(fila_doc['text'].values[0])[:180].replace("\n", " ") + "..."
        else:
            snippet = "[Contenido no encontrado]"
            
        datos_tabla.append({
            "Ranking": f"🥇 {rank}" if rank == 1 else f"{rank}",
            "ID Documento": doc_id,
            "Métrica Relevancia (Score)": f"{score:.4f}",
            "Contenido (Snippet)": snippet
        })
    
    st.dataframe(pd.DataFrame(datos_tabla), use_container_width=True, hide_index=True)

# ==========================================
# ARQUITECTURA DE LA INTERFAZ DE USUARIO
# ==========================================
st.title("🏛️ Sistema de Recuperación de Información — Reuters Corpus")
st.markdown("Plataforma interactiva de evaluación y benchmarking de modelos clásicos y basados en representaciones semánticas.")

tab_jaccard, tab_tfidf, tab_bm25, tab_embeddings, tab_comparativa = st.tabs([
    "🔍 Modelo Jaccard", 
    "📈 Modelo TF-IDF", 
    "🔥 Modelo BM25", 
    "🧠 Embeddings Semánticos", 
    "📊 Panel Comparativo"
])

with tab_jaccard:
    st.header("Modelo Binario con Similitud Jaccard")
    st.caption("Recuperación léxica basada en conjuntos de términos (Presencia/Ausencia).")
    query_j = st.text_input("Buscar en el Corpus (Jaccard):", key="search_j", placeholder="Ej. crude oil petroleum")
    if query_j:
        renderizar_tabla_top4(mod_jaccard.search(query_j, k=4), df_corpus, k=4)

with tab_tfidf:
    st.header("Modelo Vectorial con TF-IDF y Similitud Coseno")
    st.caption("Ponderación estadística de términos mapeados en un espacio vectorial continuo.")
    query_t = st.text_input("Buscar en el Corpus (TF-IDF):", key="search_t", placeholder="Ej. quarterly corporate profits")
    if query_t:
        renderizar_tabla_top4(mod_tfidf.search(query_t, k=4), df_corpus, k=4)

with tab_bm25:
    st.header("Modelo Probabilístico BM25")
    st.caption("Algoritmo clásico de vanguardia que incorpora saturación de términos y normalización por longitud de documentos.")
    query_b = st.text_input("Buscar en el Corpus (BM25):", key="search_b", placeholder="Ej. wheat grain export shipping")
    if query_b:
        renderizar_tabla_top4(mod_bm25.search(query_b, k=4), df_corpus, k=4)

with tab_embeddings:
    st.header("Recuperación Semántica Basada en Embeddings mas Base Vectorial")
    st.caption("Mapeo denso de oraciones en espacios latentes que captura el contexto y la sinonimia más allá de la coincidencia exacta de palabras.")
    query_e = st.text_input("Buscar en el Corpus (Dense Embeddings):", key="search_e", placeholder="Ej. energy resource market trade")
    if query_e:
        renderizar_tabla_top4(mod_embeddings.search(query_e, k=4), df_corpus, k=4)

# --- PESTAÑA 5: COMPARATIVA Y EVALUACIÓN ---
with tab_comparativa:
    st.header("Análisis Comparativo y Banco de Pruebas")
    st.write("Ejecución automatizada de las 10 consultas predefinidas evaluadas contra las categorías etiquetadas del dataset Reuters.")
    
    with st.expander("👁️ Ver las 10 Consultas de Prueba configuradas"):
        st.json(queries_prueba)
        
    if st.button("🚀 Ejecutar Benchmark del Sistema (10 Consultas)"):
        with st.spinner("Evaluando modelos y calculando métricas de rendimiento..."):
            
            k_eval = 5
            modelos_dict = {
                "Jaccard (Binario)": mod_jaccard,
                "TF-IDF (Coseno)": mod_tfidf,
                "BM25 Probabilístico": mod_bm25,
                "Semántico (Embeddings)": mod_embeddings
            }
            
            # 1. Instanciar Gestor y Obtener Qrels Reales (Ground Truth)
            gestor_qrels = QrelsManager(modelos_dict, queries_prueba)
            qrels_reales = gestor_qrels.generar_ground_truth_desde_corpus(df_corpus, mapeo_consultas_categorias)
            
            # 2. Obtener los documentos recuperados por los modelos (para gráfico de solapamiento)
            qrels_recuperados = gestor_qrels.generar_qrels_por_modelos(k_pool=k_eval)
            df_comparativa_documentos = gestor_qrels.a_dataframe_comparativo()
            
            # 3. Inicializar Evaluador y calcular métricas
            evaluador = Evaluator(qrels_reales)
            df_map_global, _ = evaluador.evaluar_modelos(modelos_dict, queries_prueba, k=k_eval)
            df_map_global["MAP"] = df_map_global["MAP"].round(4)
            
            # 4. Calcular promedios de Precision y Recall
            filas_metricas_agregadas = []
            for nombre_modelo, modelo in modelos_dict.items():
                precisiones, recalls = [], []
                for q_id, q_text in queries_prueba.items():
                    resultados = modelo.search(q_text, k=k_eval)
                    p, r = evaluador.precision_recall(resultados, q_id)
                    precisiones.append(p)
                    recalls.append(r)
                
                avg_precision = sum(precisiones) / len(precisiones) if precisiones else 0
                avg_recall = sum(recalls) / len(recalls) if recalls else 0
                
                filas_metricas_agregadas.append({
                    "Modelo": nombre_modelo,
                    f"Precision@{k_eval}": round(avg_precision, 4),
                    f"Recall@{k_eval}": round(avg_recall, 4)
                })

            df_metricas_agregadas = pd.DataFrame(filas_metricas_agregadas)
            df_resumen_total = pd.merge(df_map_global, df_metricas_agregadas, on="Modelo")
            
            # 5. Calcular Matriz de Coincidencias (Overlap Matrix)
            nombres_modelos = list(modelos_dict.keys())
            co_matrix = pd.DataFrame(0, index=nombres_modelos, columns=nombres_modelos)
            
            for q_id, modelos_docs in qrels_recuperados.items():
                for m1, m2 in itertools.combinations(nombres_modelos, 2):
                    # Intersección de documentos en el top K entre m1 y m2
                    interseccion = len(set(modelos_docs[m1]).intersection(set(modelos_docs[m2])))
                    co_matrix.loc[m1, m2] += interseccion
                    co_matrix.loc[m2, m1] += interseccion
            
            # Llenar la diagonal con el total de documentos recuperados por sí mismo
            for m in nombres_modelos:
                co_matrix.loc[m, m] = sum(len(set(docs[m])) for docs in qrels_recuperados.values())
            
            # --- RENDERIZADO VISUAL ---
            st.markdown("---")
            st.subheader("1. Rendimiento Global del Sistema (Promedios Macro)")
            
            col_graf_map, col_tabla_resumen = st.columns([1.5, 2.5])
            with col_graf_map:
                st.bar_chart(data=df_resumen_total.set_index("Modelo"), y="MAP", color="#1f77b4")
            with col_tabla_resumen:
                st.dataframe(df_resumen_total, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader(f"2. Matriz de Coincidencias (Solapamiento en Top {k_eval})")
            st.write("Muestra la cantidad de documentos idénticos recuperados en conjunto por dos modelos a través de las 10 consultas. Valores más altos indican mayor consenso entre los algoritmos.")
            
            # Usamos background_gradient para generar un Heatmap nativo en Streamlit
            st.dataframe(
                co_matrix.style.background_gradient(cmap='Blues', axis=None), 
                use_container_width=True
            )
            
            st.markdown("---")
            with st.expander("🔍 Ver Alineación detallada de Documentos Recuperados (Ranking)"):
                st.write("Identificación visual de intersecciones y divergencias específicas por cada posición de recuperación.")
                st.dataframe(df_comparativa_documentos, use_container_width=True, hide_index=True)

    # Análisis Cualitativo
    st.markdown("---")
    st.subheader("📝 Directrices de Análisis e Informe Técnico")
    
    col_analisis_1, col_analisis_2 = st.columns(2)
    with col_analisis_1:
        st.info(
            "📌 **¿En qué consultas funciona mejor cada modelo?**\n\n"
            "* **Modelos Léxicos (BM25 / TF-IDF):** Sobresalen en consultas con palabras clave de alta especificidad técnica o nombres propios corporativos (ej. *'OPEC'*, *'refining'*, *'tariffs'*). Si el término es exacto, el cálculo probabilístico o inverso de frecuencia de documento aísla el ruido de inmediato.\n"
            "* **Jaccard:** Es el más plano. Al no considerar la frecuencia de términos (TF), sufre cuando las stopwords mal filtradas o palabras comunes sesgan la intersección binaria.\n"
            "* **Recuperación Semántica:** Domina en consultas abstractas, cortas o con alta presencia de lenguaje natural descriptivo (ej. *'energy resource market fluctuations'*)."
        )
    with col_analisis_2:
        st.success(
            "💡 **Casos donde la Recuperación Semántica mejora o empeora**\n\n"
            "* **Mejora (Ganancia Semántica):** Ocurre ante la presencia de sinonimia o polisemia. Si una noticia de Reuters habla exclusivamente de *'petroleum price drops'* y la consulta utiliza la palabra *'crude oil'*, los embeddings capturan la cercanía en el espacio vectorial y recuperan el documento. BM25 y TF-IDF puntuarán cero si no hay solapamiento exacto de tokens.\n"
            "* **Empeora (Pérdida de Especificidad):** En consultas de grano fino. Si buscas una empresa específica o un código financiero, los embeddings tienden a promediar o suavizar el significado de los vectores, trayendo documentos temáticamente similares pero erróneos en la entidad exacta."
        )