"""Sistema RAG: busca híbrida (embeddings + HNSW + BM25) em livros do Project Gutenberg + LLM local que responde citando as fontes."""
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from usearch.index import Index

from corpus import carregar_passagens, texto_indexado
from indice_lexical import IndiceLexical

CACHE = Path(__file__).parent / "cache_indice"
MODELO_EMB = "sentence-transformers/all-MiniLM-L6-v2"
LLM = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
SEM_CONTEXTO = "The provided context is insufficient to answer this question."

PROMPT_SISTEMA = (
    "You are a helpful assistant that answers questions about books ONLY using the provided numbered "
    "passages from public-domain books. "
    "You MUST cite which passage supports each claim using square brackets, e.g. [1] or [2]. "
    "Reply in a short, fluent paragraph of plain prose (no lists, no markdown). "
    f"If the context does NOT contain relevant information to answer the question, you MUST respond EXACTLY: "
    f"'{SEM_CONTEXTO}' "
    "Do NOT make up or infer answers. Do NOT use your training data. Only use the context provided."
)

# BM25 é opcional: sem ele a recuperação é apenas semântica
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


def tokenizar(texto):
    return re.findall(r"\w+", texto.lower())


def frase_pergunta(pergunta):
    match = re.fullmatch(r"\s*where\s+does\s+the\s+(?:phrase|quote|quotation)\s+"
                         r"(.+?)\s+come\s+from\s*\??\s*", pergunta, re.IGNORECASE)
    if not match:
        match = re.fullmatch(r"\s*(?:what|which)\s+(?:book|novel|work)\s+(?:contains|includes)\s+"
                             r"the\s+(?:phrase|quote|quotation)\s+(.+?)\s*\??\s*", pergunta, re.IGNORECASE)
    if not match:
        return None
    frase = match.group(1).strip()
    if len(frase) > 1 and (frase[0], frase[-1]) in {
        ('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")
    }:
        frase = frase[1:-1].strip()
    return frase or None


def correspondencia_aproximada(frase, texto):
    palavras = tokenizar(frase)
    if len(palavras) < 8:
        return 0.0
    blocos = [b for b in SequenceMatcher(None, palavras, tokenizar(texto), autojunk=False).get_matching_blocks()
              if b.size]
    if not blocos or max(b.size for b in blocos) < 4:
        return 0.0
    iguais = sum(b.size for b in blocos)
    extensao = blocos[-1].b + blocos[-1].size - blocos[0].b
    return 2 * iguais / (len(palavras) + extensao)


def janelas_embedding(modelo, texto):
    offsets = modelo.tokenizer(texto, add_special_tokens=False, truncation=False,
                              return_offsets_mapping=True, verbose=False)["offset_mapping"]
    if not offsets:
        return [texto]
    limite = modelo.max_seq_length - modelo.tokenizer.num_special_tokens_to_add(pair=False)
    if limite < 2:
        raise ValueError("O limite de tokens do modelo de embeddings é insuficiente.")
    janelas = []
    for inicio in range(0, len(offsets), limite // 2):
        fim = min(inicio + limite, len(offsets))
        janelas.append(texto[offsets[inicio][0]:offsets[fim - 1][1]])
        if fim == len(offsets):
            break
    return janelas


def validar_citacoes(resposta, quantidade):
    """Mantém apenas citações que referenciam trechos existentes e retorna a resposta segura."""
    validas = sorted({int(m) for m in re.findall(r"\[(\d+)\]", resposta) if 1 <= int(m) <= quantidade})
    resposta = re.sub(
        r"\[(\d+)\]",
        lambda m: m.group(0) if 1 <= int(m.group(1)) <= quantidade else "",
        resposta,
    ).strip()
    if resposta != SEM_CONTEXTO and not validas:
        return SEM_CONTEXTO, []
    return resposta, validas


def titulo_pergunta_autoria(pergunta):
    match = re.fullmatch(r"\s*who\s+wrote\s+(.+?)\s*\??\s*", pergunta, re.IGNORECASE)
    if not match:
        return None
    titulo = match.group(1).strip()
    if len(titulo) > 1 and (titulo[0], titulo[-1]) in {
        ('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")
    }:
        titulo = titulo[1:-1].strip()
    return titulo


def autoria_inequivoca(livros):
    autores = []
    for livro in livros:
        titulo = livro["titulo"]
        autor = livro["autores"].strip()
        if (not autor or "\ufffd" in autor or "\ufffd" in titulo
                or re.search(r"[;\[\]]|\band\b|\b(?:anonymous|unknown|various)\b", autor, re.IGNORECASE)
                or re.search(r"\b(?:adapted|adaptation|dramatization|dramatisation|based (?:on|upon)|"
                             r"founded (?:on|upon)|translated|translation|retold|abridged|"
                             r"illustrated|edited|selected|compiled|arranged)\b",
                             titulo, re.IGNORECASE)):
            return None
        autores.append(" ".join(autor.casefold().split()))
    return livros[0] if livros and len(set(autores)) == 1 else None


class SistemaRAG:
    def __init__(self, n=20000, semente=42, llm=LLM, k=3, max_tokens=200, min_sim=0.0,
                 indice_completo=None):
        self.k, self.max_tokens, self.min_sim = k, max_tokens, min_sim
        self.indice_lexical = IndiceLexical(indice_completo) if indice_completo else None
        print(f"Carregando modelo de embeddings: {MODELO_EMB}", flush=True)
        self.modelo_emb = SentenceTransformer(MODELO_EMB, device="cpu")
        print("Modelo de embeddings pronto.", flush=True)
        if self.indice_lexical is None:
            CACHE.mkdir(exist_ok=True)
            self.passagens = carregar_passagens(CACHE, n, semente)
            self.emb = self._embeddings(n, semente)
            self.indice = Index(ndim=self.emb.shape[1], metric="cos", connectivity=16, expansion_add=128,
                                expansion_search=64)
            self.indice.add(np.arange(len(self.emb)), self.emb)
            self.bm25 = BM25Okapi([tokenizar(texto_indexado(p)) for p in self.passagens]) if HAS_BM25 else None
        else:
            self.modelo_emb.max_seq_length = 128

        self.tok = AutoTokenizer.from_pretrained(llm) if llm else None
        self.lm = AutoModelForCausalLM.from_pretrained(llm, dtype=torch.float32).eval() if llm else None

    def _embeddings(self, n, semente):
        arq = CACHE / f"emb_gutenberg_v3_{n}_{semente}.npy"
        if arq.exists():
            return np.load(arq)
        print(f"Primeira execução: codificando {len(self.passagens)} passagens (pode levar vários minutos em CPU)...",
              flush=True)
        self.modelo_emb.max_seq_length = 128
        emb = self.modelo_emb.encode([texto_indexado(p) for p in self.passagens], batch_size=64,
                                     normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        np.save(arq, emb)
        return emb

    def _vetor(self, pergunta):
        return self.modelo_emb.encode([pergunta], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)

    def recuperar(self, pergunta):
        """Top-k passagens (híbrido BM25 + semântico, ou só semântico sem BM25). A similaridade é o cosseno real."""
        if self.indice_lexical is not None:
            candidatos = self.indice_lexical.recuperar(pergunta, max(50, self.k * 10))
            if not candidatos:
                return []
            q = self._vetor(pergunta)
            janelas = [janelas_embedding(self.modelo_emb, texto_indexado(p)) for p in candidatos]
            vetores = self.modelo_emb.encode(
                [janela for grupo in janelas for janela in grupo], batch_size=64,
                normalize_embeddings=True, show_progress_bar=False,
            )
            scores = np.asarray(vetores) @ q
            similaridades = []
            posicao = 0
            for grupo in janelas:
                similaridades.append(float(np.max(scores[posicao:posicao + len(grupo)])))
                posicao += len(grupo)
            ordem_semantica = sorted(range(len(candidatos)), key=lambda i: -similaridades[i])
            rank_semantico = {idx: pos for pos, idx in enumerate(ordem_semantica, 1)}
            ordem = sorted(range(len(candidatos)),
                           key=lambda i: 1 / (60 + i + 1) + 1 / (60 + rank_semantico[i]),
                           reverse=True)
            return [{**candidatos[i], "similaridade": float(similaridades[i])} for i in ordem[:self.k]]
        q = self._vetor(pergunta)
        if self.bm25 is None:
            achados = self.indice.search(q, self.k)
            return [{**self.passagens[int(c)], "similaridade": float(1 - d)}
                    for c, d in zip(achados.keys, achados.distances)]

        scores_bm25 = self.bm25.get_scores(tokenizar(pergunta))
        bm25_norm = scores_bm25 / scores_bm25.max() if scores_bm25.max() > 0 else scores_bm25
        candidatos = {int(c) for c in self.indice.search(q, self.k * 3).keys}
        candidatos.update(int(i) for i in np.argsort(bm25_norm)[::-1][:self.k * 3] if bm25_norm[i] > 0)

        # Ranking = 0.7*cosseno + 0.3*BM25 normalizado; a similaridade exibida é sempre o cosseno
        cos = {i: float(self.emb[i] @ q) for i in candidatos}
        ordenados = sorted(candidatos, key=lambda i: 0.7 * cos[i] + 0.3 * float(bm25_norm[i]), reverse=True)[:self.k]
        return [{**self.passagens[i], "similaridade": cos[i]} for i in ordenados]

    def completar(self, msgs, max_tokens):
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            saida = self.lm.generate(**ids, max_new_tokens=max_tokens, do_sample=False, repetition_penalty=1.1)
        return self.tok.decode(saida[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def gerar(self, pergunta, trechos):
        def rotulo(t):
            return f"{t['titulo']} by {t['autores']}" if t["autores"] else t["titulo"]
        contexto = "\n\n".join(f"[{i}] ({rotulo(t)}) {t['texto']}" for i, t in enumerate(trechos, 1))
        msgs = [{"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user", "content": f"Context:\n{contexto}\n\nQuestion: {pergunta}"}]
        return self.completar(msgs, self.max_tokens)

    def perguntar(self, pergunta):
        t0 = time.perf_counter()
        titulo = titulo_pergunta_autoria(pergunta) if self.lm and self.indice_lexical is not None else None
        livros = self.indice_lexical.buscar_titulo(titulo) if titulo else []
        if titulo and not livros:
            t1 = time.perf_counter()
            return {"pergunta": pergunta, "resposta": SEM_CONTEXTO, "trechos": [], "citados": [],
                    "descartados": [], "aviso": "O título exato não foi encontrado no catálogo.",
                    "tempo_recuperacao": t1 - t0, "tempo_geracao": 0.0, "tempo_total": t1 - t0}
        if livros:
            livro = autoria_inequivoca(livros)
            t1 = time.perf_counter()
            if livro is None:
                resposta, trechos, citados = SEM_CONTEXTO, [], []
                aviso = "Os metadados do título são ambíguos ou não identificam um único autor."
            else:
                fonte = {chave: livro[chave] for chave in ("id", "titulo", "autores", "url")}
                fonte["tipo_fonte"] = "metadados"
                fonte["texto"] = f"Title: {livro['titulo']}\nAuthor: {livro['autores']}"
                resposta = f"According to the catalog, the author of {livro['titulo']} is {livro['autores']}. [1]"
                trechos, citados, aviso = [fonte], [1], None
            t2 = time.perf_counter()
            return {"pergunta": pergunta, "resposta": resposta, "trechos": trechos, "citados": citados,
                    "descartados": [], "aviso": aviso, "tempo_recuperacao": t1 - t0,
                    "tempo_geracao": t2 - t1, "tempo_total": t2 - t0}
        frase = frase_pergunta(pergunta) if self.indice_lexical is not None else None
        if frase:
            curta = len(tokenizar(frase)) < 5
            literais = self.indice_lexical.buscar_frase(frase, self.k) if not curta else []
            if literais:
                fontes = literais
                tipo = "correspondencia_literal"
            else:
                candidatos = self.indice_lexical.recuperar(pergunta, max(50, self.k * 10)) if not curta else []
                similares = [(correspondencia_aproximada(frase, t["texto"]), t) for t in candidatos]
                similares = sorted((item for item in similares if item[0] >= 0.8),
                                   key=lambda item: -item[0])
                fontes = [t for _, t in similares[:self.k]]
                tipo = "correspondencia_aproximada" if fontes else "candidato_lexical"
                if not fontes:
                    fontes = candidatos[:self.k]
            t1 = time.perf_counter()
            trechos = [{**fonte, "tipo_fonte": tipo} for fonte in fontes]
            citados = [1] if self.lm and fontes and tipo != "candidato_lexical" else []
            aviso = None
            if citados:
                descricao = "The wording appears" if tipo == "correspondencia_literal" else \
                            "A passage with similar wording appears"
                resposta = (f"{descricao} in {fontes[0]['titulo']} [1]. "
                            "This does not establish the origin or meaning of the requested wording.")
            else:
                resposta = SEM_CONTEXTO if self.lm else ""
                if tipo == "candidato_lexical":
                    aviso = ("A frase é curta demais para identificar uma fonte com segurança." if curta else
                             "Não foi encontrada correspondência textual suficiente para a frase.")
            t2 = time.perf_counter()
            return {"pergunta": pergunta, "resposta": resposta, "trechos": trechos, "citados": citados,
                    "descartados": [], "aviso": aviso, "tempo_recuperacao": t1 - t0,
                    "tempo_geracao": t2 - t1, "tempo_total": t2 - t0}
        candidatos = self.recuperar(pergunta)
        trechos = [t for t in candidatos if t["similaridade"] >= self.min_sim]
        t1 = time.perf_counter()
        if self.lm and not trechos:
            resposta = SEM_CONTEXTO  # nada relevante no corpus: evita o LLM responder de memória
        else:
            resposta = self.gerar(pergunta, trechos) if self.lm else ""
        t2 = time.perf_counter()
        if self.lm:
            resposta, citados = validar_citacoes(resposta, len(trechos))
        else:
            citados = []
        return {"pergunta": pergunta, "resposta": resposta, "trechos": trechos, "citados": citados,
                "descartados": [t for t in candidatos if t not in trechos],
                "tempo_recuperacao": t1 - t0, "tempo_geracao": t2 - t1, "tempo_total": t2 - t0}
