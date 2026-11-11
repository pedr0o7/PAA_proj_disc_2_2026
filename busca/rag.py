"""Sistema RAG: busca semântica (embeddings + HNSW) + LLM local que responde citando as fontes."""
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from usearch.index import Index

from corpus import carregar_passagens

CACHE = Path(__file__).parent / "cache_indice"
MODELO_EMB = "sentence-transformers/all-MiniLM-L6-v2"
LLM = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

PROMPT_SISTEMA = (
    "You are a helpful assistant that answers questions ONLY using the provided numbered context passages. "
    "You MUST cite which passage supports each claim using square brackets, e.g. [1] or [2]. "
    "Reply in a short, fluent paragraph of plain prose (no lists, no markdown). "
    "⚠️ CRITICAL: If the context does NOT contain relevant information to answer the question, "
    "you MUST respond EXACTLY: 'The provided context is insufficient to answer this question.' "
    "Do NOT make up or infer answers. Do NOT use your training data. Only use the context provided."
)

# Tentar importar BM25 para hybrid search; usar fallback se não disponível
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


class SistemaRAG:
    def __init__(self, n=5000, semente=42, n_shards=1, llm=LLM, k=3, max_tokens=200):
        self.k, self.max_tokens = k, max_tokens
        self.modelo_emb = SentenceTransformer(MODELO_EMB, device="cpu")
        CACHE.mkdir(exist_ok=True)
        self.passagens = carregar_passagens(CACHE, n, semente, n_shards)
        emb = self._embeddings(n, semente, n_shards)
        self.indice = Index(ndim=emb.shape[1], metric="cos", connectivity=16, expansion_add=128, expansion_search=64)
        self.indice.add(np.arange(len(emb)), emb)
        
        # Inicializar BM25 para hybrid search
        self.bm25 = None
        if HAS_BM25:
            textos_tokenizados = [p["texto"].lower().split() for p in self.passagens]
            self.bm25 = BM25Okapi(textos_tokenizados)
        
        self.tok = AutoTokenizer.from_pretrained(llm) if llm else None
        self.lm = AutoModelForCausalLM.from_pretrained(llm, dtype=torch.float32).eval() if llm else None

    def _embeddings(self, n, semente, n_shards):
        arq = CACHE / f"emb_titulo_{n}_{semente}_{n_shards}.npy"
        if arq.exists():
            return np.load(arq)
        print(f"Primeira execução: codificando {len(self.passagens)} passagens (alguns minutos em CPU)...")
        self.modelo_emb.max_seq_length = 128
        emb = self.modelo_emb.encode([p["texto"] for p in self.passagens], batch_size=64,
                                     normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        np.save(arq, emb)
        return emb

    def recuperar(self, pergunta):
        """Recupera passagens usando hybrid search (BM25 + semântico) ou apenas semântico."""
        if self.bm25 is not None:
            return self._recuperar_hibrido(pergunta)
        else:
            return self._recuperar_semantico(pergunta)
    
    def _recuperar_semantico(self, pergunta):
        """Busca apenas por similaridade semântica."""
        q = self.modelo_emb.encode([pergunta], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)
        achados = self.indice.search(q, self.k)
        return [{**self.passagens[int(c)], "similaridade": float(1 - d)} for c, d in zip(achados.keys, achados.distances)]
    
    def _recuperar_hibrido(self, pergunta):
        """Busca híbrida: combina BM25 (keyword) com semântico."""
        # BM25: busca por palavras-chave
        tokens = pergunta.lower().split()
        scores_bm25 = self.bm25.get_scores(tokens)
        
        # Normalizar scores BM25 para [0, 1]
        max_bm25 = max(scores_bm25) if max(scores_bm25) > 0 else 1
        scores_bm25_norm = scores_bm25 / max_bm25
        
        # Semântico: busca por similaridade
        q = self.modelo_emb.encode([pergunta], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)
        achados = self.indice.search(q, self.k * 3)  # Buscar 3*k para melhor combinação
        indices_semantico = np.array(achados.keys, dtype=int)
        distances_semantico = np.array(achados.distances, dtype=np.float32)
        scores_semantico = 1 - distances_semantico  # Converter distância em similaridade
        
        # Combinar: score final = média ponderada (semântico=0.7, BM25=0.3)
        scores_finais = {}
        
        # Adicionar scores semânticos
        for idx, score in zip(indices_semantico, scores_semantico):
            idx = int(idx)
            scores_finais[idx] = 0.7 * score
        
        # Adicionar/combinar com scores BM25
        for idx, score_bm25 in enumerate(scores_bm25_norm):
            if score_bm25 > 0:
                if idx in scores_finais:
                    scores_finais[idx] = 0.7 * scores_finais[idx] / 0.7 + 0.3 * score_bm25  # Normalizar
                else:
                    scores_finais[idx] = 0.3 * score_bm25
        
        # Ordenar por score final e pegar top k
        indices_ordenados = sorted(scores_finais.keys(), key=lambda x: scores_finais[x], reverse=True)[:self.k]
        
        # Retornar com score final normalizado
        return [{**self.passagens[int(idx)], "similaridade": float(scores_finais[idx])} for idx in indices_ordenados]

    def completar(self, msgs, max_tokens):
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            saida = self.lm.generate(**ids, max_new_tokens=max_tokens, do_sample=False, repetition_penalty=1.1)
        return self.tok.decode(saida[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def gerar(self, pergunta, trechos):
        contexto = "\n\n".join(f"[{i}] ({t['titulo']}) {t['texto']}" for i, t in enumerate(trechos, 1))
        msgs = [{"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user", "content": f"Context:\n{contexto}\n\nQuestion: {pergunta}"}]
        return self.completar(msgs, self.max_tokens)

    def perguntar(self, pergunta):
        t0 = time.perf_counter()
        trechos = self.recuperar(pergunta)
        t1 = time.perf_counter()
        resposta = self.gerar(pergunta, trechos) if self.lm else ""
        t2 = time.perf_counter()
        citados = sorted({int(m) for m in re.findall(r"\[(\d+)\]", resposta) if 1 <= int(m) <= len(trechos)})
        return {"pergunta": pergunta, "resposta": resposta, "trechos": trechos, "citados": citados,
                "tempo_recuperacao": t1 - t0, "tempo_geracao": t2 - t1, "tempo_total": t2 - t0}
