import pandas as pd

class Evaluator:
    def __init__(self, qrels: dict):
        self.qrels = qrels

    def precision_recall(self, resultados, query_id):
        relevantes = set(self.qrels.get(query_id, []))
        obtenidos = set(doc_id for doc_id, _ in resultados)
        if not relevantes or not obtenidos:
            return 0.0, 0.0
        inter = obtenidos & relevantes
        return len(inter) / len(obtenidos), len(inter) / len(relevantes)

    def map_score(self, resultados_por_query):
        aps = []
        for q_id, resultados in resultados_por_query.items():
            relevantes = self.qrels.get(q_id, [])
            if not relevantes:
                continue
            hits, suma = 0, 0.0
            for i, (doc_id, _) in enumerate(resultados):
                if doc_id in relevantes:
                    hits += 1
                    suma += hits / (i + 1)
            aps.append(suma / len(relevantes))
        return sum(aps) / len(aps) if aps else 0.0

    def evaluar_modelos(self, modelos_dict, queries_prueba, k=5):
        resumen = []
        for nombre, modelo in modelos_dict.items():
            resultados_globales = {}
            for q_id, q_text in queries_prueba.items():
                resultados = modelo.search(q_text, k=k)
                resultados_globales[q_id] = resultados
                p, r = self.precision_recall(resultados, q_id)
                print(f"[{nombre}] '{q_text}': P={p:.2f} R={r:.2f}")
            map_val = self.map_score(resultados_globales)
            print(f"MAP {nombre}: {map_val:.4f}\n")
            resumen.append({"Modelo": nombre, "MAP": map_val})
        return pd.DataFrame(resumen).sort_values("MAP", ascending=False).reset_index(drop=True)
