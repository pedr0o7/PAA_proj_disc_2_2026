"""Avalia o sistema RAG com RAGAS (juiz = LLM local) e métricas próprias de recuperação, citação e tempo.

Métricas RAGAS: faithfulness, answer_relevancy, context_precision, context_recall e semantic_similarity
(resposta x referência). Métricas próprias: hit@k e MRR da fonte esperada, citação correta, tempos.
Uso: avaliar.py [--q 30] [--k 3] [--juiz HF_ID]  ->  avaliacao_resultados.json
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from rag import LLM, SistemaRAG

sys.stdout.reconfigure(encoding="utf-8")

DATASET = Path(__file__).parent / "avaliacao_dataset.json"
SAIDA = Path(__file__).parent / "avaliacao_resultados.json"


def metricas_proprias(itens, resultados):
    hits, rr, cit_ok, citou = [], [], [], []
    for it, r in zip(itens, resultados):
        titulos = [t["titulo"] for t in r["trechos"]]
        pos = titulos.index(it["titulo_esperado"]) + 1 if it["titulo_esperado"] in titulos else None
        hits.append(pos is not None)
        rr.append(1 / pos if pos else 0.0)
        citou.append(bool(r["citados"]))
        cit_ok.append(any(titulos[i - 1] == it["titulo_esperado"] for i in r["citados"]))
    return {
        "hit_at_k": float(np.mean(hits)),
        "mrr": float(np.mean(rr)),
        "respostas_com_citacao": float(np.mean(citou)),
        "citacao_da_fonte_correta": float(np.mean(cit_ok)),
    }


def metricas_tempo(resultados):
    def resumo(chave):
        v = sorted(r[chave] for r in resultados)
        return {"media": statistics.mean(v), "mediana": statistics.median(v), "p95": v[int(0.95 * (len(v) - 1))]}
    return {c: resumo(c) for c in ("tempo_recuperacao", "tempo_geracao", "tempo_total")}


def avaliar_ragas(sistema, itens, resultados, juiz):
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall,
                               ResponseRelevancy, SemanticSimilarity)
    from ragas.run_config import RunConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    if juiz == LLM:
        tok, lm = sistema.tok, sistema.lm
    else:
        tok = AutoTokenizer.from_pretrained(juiz)
        lm = AutoModelForCausalLM.from_pretrained(juiz, dtype=torch.float32).eval()
    pipe = pipeline("text-generation", model=lm, tokenizer=tok, max_new_tokens=350, do_sample=False,
                    return_full_text=False)
    llm = LangchainLLMWrapper(HuggingFacePipeline(pipeline=pipe))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

    amostras = [SingleTurnSample(user_input=it["pergunta"], response=r["resposta"], reference=it["referencia"],
                                 retrieved_contexts=[t["texto"] for t in r["trechos"]])
                for it, r in zip(itens, resultados)]
    res = evaluate(EvaluationDataset(amostras),
                   metrics=[Faithfulness(), ResponseRelevancy(strictness=1), LLMContextPrecisionWithReference(),
                            LLMContextRecall(), SemanticSimilarity()],
                   llm=llm, embeddings=emb, run_config=RunConfig(timeout=900, max_workers=1),
                   raise_exceptions=False)
    df = res.to_pandas()
    colunas = [c for c in df.columns if c not in ("user_input", "retrieved_contexts", "response", "reference")]
    return {c: (None if df[c].dropna().empty else float(df[c].mean())) for c in colunas}, \
           {c: int(df[c].isna().sum()) for c in colunas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=30, help="perguntas avaliadas")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--llm", default=LLM)
    ap.add_argument("--juiz", default=None, help="LLM juiz do RAGAS (padrão: o mesmo --llm)")
    ap.add_argument("--sem-ragas", action="store_true")
    a = ap.parse_args()

    itens = json.loads(DATASET.read_text(encoding="utf-8"))[: a.q]
    sistema = SistemaRAG(n=a.n, llm=a.llm, k=a.k)
    sistema.perguntar(itens[0]["pergunta"])  # aquecimento, fora das medições
    resultados = []
    for i, it in enumerate(itens, 1):
        resultados.append(sistema.perguntar(it["pergunta"]))
        print(f"{i}/{len(itens)} {resultados[-1]['tempo_total']:.1f}s", flush=True)

    saida = {"llm": a.llm, "k": a.k, "perguntas": len(itens), "proprias": metricas_proprias(itens, resultados),
             "tempo_s": metricas_tempo(resultados)}
    if not a.sem_ragas:
        saida["ragas"], saida["ragas_nan"] = avaliar_ragas(sistema, itens, resultados, a.juiz or a.llm)
        saida["ragas_juiz"] = a.juiz or a.llm
    saida["detalhe"] = [{"pergunta": it["pergunta"], "referencia": it["referencia"], "resposta": r["resposta"],
                         "fonte_esperada": it["titulo_esperado"], "fontes": [t["titulo"] for t in r["trechos"]],
                         "citados": r["citados"], "tempo_total": r["tempo_total"]}
                        for it, r in zip(itens, resultados)]
    SAIDA.write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in saida.items() if k != "detalhe"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
