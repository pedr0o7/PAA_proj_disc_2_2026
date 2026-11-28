"""Sistema RAG: busca híbrida (embeddings + HNSW + BM25) em livros do Project Gutenberg + LLM local que responde citando as fontes."""
import json
import logging
import os
import re
import struct
import time
from difflib import SequenceMatcher
import zipfile
import zlib
from importlib.metadata import version
from pathlib import Path

from configuracao_hf import configurar_huggingface

configurar_huggingface()

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from usearch.index import Index

from corpus import carregar_passagens, texto_indexado
from indice_lexical import IndiceLexical
from acesso_hf import carregar_modelo, preparar_acesso_hub
from cache import gravar_atomico, identidade
from infraestrutura.aceleracao import (
    carregar_modelo_embeddings,
    carregar_modelo_llm,
    normalizar_vetores,
    resolver_config_execucao,
    resumo_execucao,
)

CACHE = Path(__file__).parent / "cache_indice"
MODELO_EMB = "sentence-transformers/all-MiniLM-L6-v2"
REVISAO_EMB = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MAX_SEQ_EMB = 128
LLM_PADRAO = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
LLM = os.environ.get("LLM_MODEL", "").strip() or LLM_PADRAO
SEM_CONTEXTO = "The provided context is insufficient to answer this question."
MIN_SUPORTE_CITACAO = 0.30
LOG = logging.getLogger("busca.rag")


PROMPT_SISTEMA = (
    "You are a literary QA assistant for Project Gutenberg public-domain books.\n"
    "Follow these rules strictly:\n"
    "1. Answer ONLY using facts explicitly stated in the numbered passages below. Never use outside knowledge.\n"
    "2. Answer in English in 1 to 3 plain prose sentences (no lists, no markdown).\n"
    "3. End EVERY factual sentence with the supporting passage number in square brackets, like [1] or [2]. "
    'Example: "The lighthouse keeper lit the beacon at dusk [1]."\n'
    "4. Never cite a passage that does not directly support the claim.\n"
    f"5. If the passages do not contain the answer, reply EXACTLY: '{SEM_CONTEXTO}'"
)

EXEMPLOS_GERACAO = [
    {
        "role": "user",
        "content": (
            "Context:\n"
            "[1] (The Beacon by Clara Reed) Jonas tended the coastal lighthouse for thirty years "
            "and warned ships during winter storms.\n\n"
            "[2] (The Beacon by Clara Reed) Sailors called him a stern but faithful captain of the shore "
            "who lost his left hand in a gale.\n\n"
            "Question: Who is Jonas in The Beacon?\n"
            "Answer in 1-3 sentences using only the passages above, ending every sentence with the supporting passage number ([1], [2])."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Jonas is a coastal lighthouse keeper who warned ships during winter storms [1]. "
            "Sailors described him as stern but faithful after he lost his left hand in a gale [2]."
        ),
    },
    {
        "role": "user",
        "content": (
            "Context:\n"
            "[1] (The Beacon by Clara Reed) Jonas tended the coastal lighthouse for thirty years.\n\n"
            "Question: What year was the smartphone invented?\n"
            "Answer in 1-3 sentences using only the passages above, ending every sentence with the supporting passage number ([1])."
        ),
    },
    {"role": "assistant", "content": SEM_CONTEXTO},
]

# BM25 é opcional: sem ele a recuperação é apenas semântica
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


def tokenizar(texto):
    return re.findall(r"\w+", texto.lower())


def _shingles_texto(texto, n=4):
    """Extrai n-gramas de palavras para detectar passagens quase idênticas entre edições."""
    palavras = tokenizar(texto)
    if len(palavras) < n:
        return set()
    return {tuple(palavras[i:i + n]) for i in range(len(palavras) - n + 1)}


def _trecho_duplicado(shingles, selecionados, limiar=0.25):
    if not shingles:
        return False
    return any((len(shingles & prev) / min(len(shingles), len(prev))) >= limiar for prev in selecionados if prev)


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
    resposta = re.sub(r"\[(?:Passage|Source|Context)\s*(\d+)\]", r"[\1]", resposta, flags=re.IGNORECASE)
    resposta = re.sub(r"\b(?:Passage|Source|Context)\s*\[(\d+)\]", r"[\1]", resposta, flags=re.IGNORECASE)
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
        self.config_execucao = resolver_config_execucao(usar_llm=bool(llm))
        self.config_execucao["llm_modelo"] = llm if llm else None
        self.offline = preparar_acesso_hub()
        LOG.info("Carregando modelo de embeddings (%s/%s); o Hub pode usar arquivos em cache.",
                 self.config_execucao["emb_backend"], self.config_execucao["emb_dispositivo"])
        self.modelo_emb, self.offline = carregar_modelo_embeddings(
            MODELO_EMB, REVISAO_EMB, self.offline, self.config_execucao,
            carregar_fn=carregar_modelo, st_cls=SentenceTransformer,
        )
        self.modelo_emb.max_seq_length = MAX_SEQ_EMB
        LOG.info("Modelo de embeddings carregado (%s/%s).",
                 self.config_execucao["emb_backend"], self.config_execucao["emb_dispositivo"])
        if self.indice_lexical is None:
            CACHE.mkdir(parents=True, exist_ok=True)
            self.passagens = carregar_passagens(CACHE, n, semente)
            self.emb = self._embeddings()
            self.indice = Index(ndim=self.emb.shape[1], metric="cos", connectivity=16, expansion_add=128,
                                expansion_search=64)
            self.indice.add(np.arange(len(self.emb)), self.emb)
            self.bm25 = BM25Okapi([tokenizar(texto_indexado(p)) for p in self.passagens]) if HAS_BM25 else None
        else:
            self.passagens = []
            self.bm25 = None

        if llm:
            LOG.info("Carregando modelo de geração (%s/%s, %s); o Hub pode usar arquivos em cache.",
                     self.config_execucao["llm_backend"], self.config_execucao["llm_dispositivo"],
                     self.config_execucao["llm_quantizacao"])
            self.tok, self.offline = carregar_modelo(
                f"o tokenizer {llm}", AutoTokenizer.from_pretrained, llm,
                local_files_only=self.offline, retornar_offline=True,
            )
            self.lm, self.offline = carregar_modelo_llm(
                llm, self.offline, self.config_execucao,
                carregar_fn=carregar_modelo, auto_model_cls=AutoModelForCausalLM,
            )
            LOG.info("Modelo de geração carregado (%s).", resumo_execucao(self.config_execucao))
        else:
            self.tok, self.lm = None, None
            LOG.info("Execução configurada: %s", resumo_execucao(self.config_execucao))

    def _embeddings(self):
        dimensao_modelo = self.modelo_emb.get_sentence_embedding_dimension()
        if not isinstance(dimensao_modelo, int) or dimensao_modelo <= 0:
            raise ValueError("O modelo de embeddings não informou uma dimensão válida.")
        emb_backend = getattr(self, "config_execucao", {}).get("emb_backend", "torch")
        metadados = {
            "versao": 1, "passagens_sha256": identidade(self.passagens),
            "modelo": MODELO_EMB, "revisao": REVISAO_EMB, "max_seq_length": MAX_SEQ_EMB,
            "normalizado": True, "dtype": "float32",
            "sentence_transformers": version("sentence-transformers"),
            "transformers": version("transformers"),
            "torch": version("torch"),
        }
        if emb_backend != "torch":
            metadados["backend"] = emb_backend
        arq = CACHE / f"embeddings_v1_{identidade(metadados)}.npz"
        if arq.exists():
            try:
                with arq.open("rb") as arquivo, np.load(arquivo, allow_pickle=False) as registro:
                    bruto_meta = registro["metadados"]
                    if not isinstance(bruto_meta, np.ndarray) or bruto_meta.ndim != 0:
                        raise ValueError(f"Cache de embeddings inválido ou incompatível: {arq}")
                    meta_item = bruto_meta.item()
                    if isinstance(meta_item, bytes):
                        meta_item = meta_item.decode("utf-8")
                    if not isinstance(meta_item, str):
                        raise ValueError(f"Cache de embeddings inválido ou incompatível: {arq}")
                    meta_lida = json.loads(meta_item)
                    emb = registro["embeddings"]
                    if not isinstance(emb, np.ndarray):
                        raise ValueError(f"Cache de embeddings inválido ou incompatível: {arq}")
            except (
                OSError,
                KeyError,
                ValueError,
                TypeError,
                EOFError,
                zipfile.BadZipFile,
                zlib.error,
                struct.error,
                NotImplementedError,
                RuntimeError,
            ) as exc:
                raise ValueError(f"Cache de embeddings inválido ou incompatível: {arq}") from exc
            if meta_lida != metadados:
                raise ValueError(f"Metadados de embeddings incompatíveis: {arq}")
            if (
                emb.ndim != 2
                or len(emb) != len(self.passagens)
                or emb.shape[1] != dimensao_modelo
                or emb.dtype != np.float32
            ):
                raise ValueError(f"Cache de embeddings inválido ou incompatível: {arq}")
            LOG.info("Usando embeddings em cache.")
            return emb
        print(f"Primeira execução: codificando {len(self.passagens)} passagens (alguns minutos em CPU)...")
        emb = normalizar_vetores(self.modelo_emb.encode(
            [texto_indexado(p) for p in self.passagens], batch_size=64,
            normalize_embeddings=True, show_progress_bar=True,
        ))
        if emb.ndim != 2 or len(emb) != len(self.passagens) or emb.shape[1] != dimensao_modelo:
            raise ValueError("O modelo retornou embeddings incompatíveis com as passagens.")
        gravar_atomico(arq, lambda arquivo: np.savez(
            arquivo, metadados=json.dumps(metadados, sort_keys=True), embeddings=emb,
        ))
        return emb

    def _vetor(self, pergunta):
        return normalizar_vetores(
            self.modelo_emb.encode([pergunta], normalize_embeddings=True, show_progress_bar=False)[0]
        )

    def recuperar(self, pergunta):
        """Top-k passagens (híbrido BM25 + semântico, ou só semântico sem BM25). A similaridade é o cosseno real."""
        if self.indice_lexical is not None:
            candidatos = self.indice_lexical.recuperar(pergunta, max(50, self.k * 10))
            if not candidatos:
                return []
            q = self._vetor(pergunta)
            janelas = [janelas_embedding(self.modelo_emb, texto_indexado(p)) for p in candidatos]
            vetores = normalizar_vetores(self.modelo_emb.encode(
                [janela for grupo in janelas for janela in grupo], batch_size=64,
                normalize_embeddings=True, show_progress_bar=False,
            ))
            scores = vetores @ q
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
            escolhidos, vistos = [], []
            for i in ordem:
                sh = _shingles_texto(candidatos[i]["texto"])
                if _trecho_duplicado(sh, vistos):
                    continue
                escolhidos.append({**candidatos[i], "similaridade": float(similaridades[i])})
                if sh:
                    vistos.append(sh)
                if len(escolhidos) == self.k:
                    break
            return escolhidos
        q = self._vetor(pergunta)
        if self.bm25 is None:
            achados = self.indice.search(q, self.k)
            return [{**self.passagens[int(c)], "similaridade": float(1 - d)}
                    for c, d in zip(achados.keys, achados.distances)]

        scores_bm25 = self.bm25.get_scores(tokenizar(pergunta))
        max_bm25 = float(scores_bm25.max()) if len(scores_bm25) else 0.0
        bm25_norm = scores_bm25 / max_bm25 if max_bm25 > 0 else scores_bm25
        candidatos = {int(c) for c in self.indice.search(q, self.k * 3).keys}
        limite_bm25 = min(self.k * 3, len(bm25_norm))
        if limite_bm25 > 0 and max_bm25 > 0:
            if limite_bm25 == len(bm25_norm):
                topo_bm25 = np.argsort(-bm25_norm)
            else:
                idx_part = np.argpartition(-bm25_norm, limite_bm25 - 1)[:limite_bm25]
                topo_bm25 = idx_part[np.argsort(-bm25_norm[idx_part])]
            candidatos.update(int(i) for i in topo_bm25 if bm25_norm[i] > 0)

        # Ranking = 0.7*cosseno + 0.3*BM25 normalizado; a similaridade exibida é sempre o cosseno
        cos = {i: float(self.emb[i] @ q) for i in candidatos}
        ordenados = sorted(candidatos, key=lambda i: 0.7 * cos[i] + 0.3 * float(bm25_norm[i]), reverse=True)[:self.k]
        return [{**self.passagens[i], "similaridade": cos[i]} for i in ordenados]

    def completar(self, msgs, max_tokens):
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        alvo_device = getattr(self.lm, "device", None)
        if isinstance(alvo_device, torch.device):
            ids = {chave: valor.to(alvo_device) for chave, valor in ids.items()}
        with torch.no_grad():
            saida = self.lm.generate(**ids, max_new_tokens=max_tokens, do_sample=False, repetition_penalty=1.1)
        return self.tok.decode(saida[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def _validar_suporte_semantico(self, resposta, trechos, citados, min_suporte=MIN_SUPORTE_CITACAO):
        """Rejeita citações cujo trecho não tem suporte semântico mínimo para a resposta gerada."""
        modelo = getattr(self, "modelo_emb", None)
        if not citados or modelo is None or resposta == SEM_CONTEXTO:
            return resposta, citados
        texto_limpo = re.sub(r"\[\d+\]", "", resposta).strip()
        if not texto_limpo:
            return SEM_CONTEXTO, []
        vet_resp = normalizar_vetores(
            modelo.encode([texto_limpo], normalize_embeddings=True, show_progress_bar=False)[0]
        )
        suportadas = []
        for idx in citados:
            if not (1 <= idx <= len(trechos)):
                continue
            conteudo = texto_indexado(trechos[idx - 1])
            janelas = janelas_embedding(modelo, conteudo) if hasattr(modelo, "tokenizer") else [conteudo]
            vet_trecho = normalizar_vetores(
                modelo.encode(janelas, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            )
            sim = float(np.max(vet_trecho @ vet_resp))
            if sim >= min_suporte:
                suportadas.append(idx)
        if not suportadas or len(suportadas) < len(citados):
            return SEM_CONTEXTO, []
        return resposta, suportadas

    def gerar(self, pergunta, trechos):
        def rotulo(t):
            return f"{t['titulo']} by {t['autores']}" if t["autores"] else t["titulo"]
        contexto = "\n\n".join(f"[{i}] ({rotulo(t)}) {t['texto']}" for i, t in enumerate(trechos, 1))
        nums = ", ".join(f"[{i}]" for i in range(1, len(trechos) + 1))
        instrucao = (
            f"Answer in 1-3 sentences using only the passages above, "
            f"ending every sentence with the supporting passage number ({nums})."
        )
        msgs = [
            {"role": "system", "content": PROMPT_SISTEMA},
            *EXEMPLOS_GERACAO,
            {"role": "user", "content": f"Context:\n{contexto}\n\nQuestion: {pergunta}\n{instrucao}"},
        ]
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
            resposta, citados = self._validar_suporte_semantico(resposta, trechos, citados)
        else:
            citados = []
        t3 = time.perf_counter()
        return {"pergunta": pergunta, "resposta": resposta, "trechos": trechos, "citados": citados,
                "descartados": [t for t in candidatos if t["similaridade"] < self.min_sim],
                "tempo_recuperacao": t1 - t0, "tempo_geracao": t2 - t1, "tempo_total": t3 - t0}
