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

from configuracao_hf import configurar_huggingface

configurar_huggingface()

import numpy as np
import torch
from acesso_hf import carregar_modelo
from indice_lexical import INDICE
from rag import LLM, MODELO_EMB, REVISAO_EMB, SEM_CONTEXTO, SistemaRAG

sys.stdout.reconfigure(encoding="utf-8")

DATASET = Path(__file__).parent / "avaliacao_dataset.json"
SAIDA = Path(__file__).parent / "avaliacao_resultados.json"


def _remover_cercas_markdown(texto):
    """Remove cercas ```json / ``` do início e do fim, sem regex (evita backtracking)."""
    texto = texto.strip()
    if texto.startswith("```"):
        texto = texto[3:]
        if texto[:4].lower() == "json":
            texto = texto[4:]
        texto = texto.lstrip()
    if texto.endswith("```"):
        texto = texto[:-3].rstrip()
    return texto.strip()


def metricas_proprias(itens, resultados):
    hits, rr, cit_ok, citou, abstencoes = [], [], [], [], []
    for it, r in zip(itens, resultados):
        ids = [t["id"] for t in r["trechos"]]
        pos = ids.index(it["id_esperado"]) + 1 if it["id_esperado"] in ids else None
        hits.append(pos is not None)
        rr.append(1 / pos if pos else 0.0)
        citou.append(bool(r["citados"]))
        cit_ok.append(any(ids[i - 1] == it["id_esperado"] for i in r["citados"]))
        abstencoes.append(r["resposta"] == SEM_CONTEXTO)
    return {
        "hit_at_k": float(np.mean(hits)),
        "mrr": float(np.mean(rr)),
        "respostas_com_citacao": float(np.mean(citou)),
        "citacao_da_fonte_correta": float(np.mean(cit_ok)),
        "respostas_abstidas": float(np.mean(abstencoes)),
    }


def metricas_tempo(resultados):
    def resumo(chave):
        v = sorted(r[chave] for r in resultados)
        return {"media": statistics.mean(v), "mediana": statistics.median(v), "p95": v[int(0.95 * (len(v) - 1))]}
    return {c: resumo(c) for c in ("tempo_recuperacao", "tempo_geracao", "tempo_total")}


def _carregar_embeddings_ragas(modelo, revisao, local_files_only=False):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=modelo,
        model_kwargs={"revision": revisao, "local_files_only": local_files_only},
    )


def _criar_llm_langchain_openvino(tok, lm, max_new_tokens=512):
    from langchain_core.language_models.llms import LLM as LangchainBaseLLM

    class OpenVINOGenAILangchainLLM(LangchainBaseLLM):
        @property
        def _llm_type(self):
            return "openvino_genai"

        def _call(self, prompt, stop=None, run_manager=None, **kwargs):
            msgs = [
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluation judge. Return ONLY valid JSON matching the "
                        "requested schema, with no markdown fences or extra commentary."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            ids = tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
            with torch.no_grad():
                saida = lm.generate(
                    **ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.05,
                )
            texto = tok.decode(saida[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            texto = _remover_cercas_markdown(texto)
            if stop:
                for token_parada in stop:
                    if token_parada in texto:
                        texto = texto.split(token_parada, 1)[0]
            return texto.strip()

    return OpenVINOGenAILangchainLLM()


def avaliar_ragas(sistema, itens, resultados, juiz):
    from langchain_huggingface import HuggingFacePipeline
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall,
                               ResponseRelevancy, SemanticSimilarity)
    from ragas.run_config import RunConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from infraestrutura.aceleracao import ModeloLLMOpenVINOGenAI

    modelo_sistema = getattr(sistema, "config_execucao", {}).get("llm_modelo")
    if juiz == LLM or (modelo_sistema and juiz == modelo_sistema):
        tok, lm = sistema.tok, sistema.lm
    else:
        tok, sistema.offline = carregar_modelo(
            f"o tokenizer do juiz {juiz}", AutoTokenizer.from_pretrained, juiz,
            local_files_only=sistema.offline, retornar_offline=True,
        )
        lm, sistema.offline = carregar_modelo(
            f"o modelo juiz {juiz}", AutoModelForCausalLM.from_pretrained, juiz,
            dtype=torch.float32, local_files_only=sistema.offline, retornar_offline=True,
        )
        lm = lm.eval()
    if isinstance(lm, ModeloLLMOpenVINOGenAI):
        llm = LangchainLLMWrapper(_criar_llm_langchain_openvino(tok, lm))
    else:
        pipe = pipeline("text-generation", model=lm, tokenizer=tok, max_new_tokens=350, do_sample=False,
                        return_full_text=False)
        llm = LangchainLLMWrapper(HuggingFacePipeline(pipeline=pipe))
    emb_lc, sistema.offline = carregar_modelo(
        f"o modelo de embeddings {MODELO_EMB}",
        _carregar_embeddings_ragas,
        MODELO_EMB,
        REVISAO_EMB,
        local_files_only=sistema.offline,
        retornar_offline=True,
    )
    emb = LangchainEmbeddingsWrapper(emb_lc)

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
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--llm", default=LLM)
    ap.add_argument("--juiz", default=None, help="LLM juiz do RAGAS (padrão: o mesmo --llm)")
    ap.add_argument("--sem-ragas", action="store_true")
    ap.add_argument("--min-sim", type=float, default=0.0, help="limiar de similaridade cosseno")
    ap.add_argument("--indice-completo", type=Path, nargs="?", const=INDICE,
                    help="usa o índice lexical completo já construído")
    ap.add_argument("--saida", type=Path, default=SAIDA, help="caminho do JSON de saída")
    a = ap.parse_args()

    itens = json.loads(DATASET.read_text(encoding="utf-8"))[: a.q]
    sistema = SistemaRAG(n=a.n, llm=a.llm, k=a.k, min_sim=a.min_sim,
                         indice_completo=a.indice_completo)
    sistema.perguntar(itens[0]["pergunta"])  # aquecimento, fora das medições
    resultados = []
    for i, it in enumerate(itens, 1):
        resultados.append(sistema.perguntar(it["pergunta"]))
        print(f"{i}/{len(itens)} {resultados[-1]['tempo_total']:.1f}s", flush=True)

    saida = {"llm": a.llm, "k": a.k, "min_sim": a.min_sim,
             "indice_completo": str(a.indice_completo) if a.indice_completo else None,
             "execucao": getattr(sistema, "config_execucao", None),
             "perguntas": len(itens), "proprias": metricas_proprias(itens, resultados),
             "tempo_s": metricas_tempo(resultados)}
    if not a.sem_ragas:
        saida["ragas"], saida["ragas_nan"] = avaliar_ragas(sistema, itens, resultados, a.juiz or a.llm)
        saida["ragas_juiz"] = a.juiz or a.llm
    saida["detalhe"] = [{"pergunta": it["pergunta"], "referencia": it["referencia"], "resposta": r["resposta"],
                         "fonte_esperada": it["titulo_esperado"], "fontes": [t["titulo"] for t in r["trechos"]],
                         "citados": r["citados"], "tempo_total": r["tempo_total"]}
                        for it, r in zip(itens, resultados)]
    a.saida.write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in saida.items() if k != "detalhe"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
