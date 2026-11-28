# Documentação Técnica - Projeto PAA: Sistema RAG com LLM Local

**Data**: 2026-11-27  
**Versão**: 2.0  
**Status**: ✅ Documentação Completa

> **Atualização de infraestrutura:** as seções abaixo sobre formato dos caches,
> amostragem e preparação do ambiente descrevem uma versão anterior. Consulte
> [README.md](README.md) e [migração do Hugging Face](busca/MIGRACAO_HUGGING_FACE.md)
> para os formatos atuais, a regeneração explícita e a revisão do modelo.

---

## Índice

- [2. Linguagens e Bibliotecas (com licenças)](#2-linguagens-e-bibliotecas-com-licenças)
- [3. Programas e Infraestrutura](#3-programas-e-infraestrutura)
- [4. Tema e Funcionamento do Sistema](#4-tema-e-funcionamento-do-sistema)
- [5. Dados de Treinamento](#5-dados-de-treinamento)
- [6. Pré-processamento e Análise de Complexidade](#6-pré-processamento-e-análise-de-complexidade)
- [7. Busca Semântica: Tempo e Complexidade](#7-busca-semântica-tempo-e-complexidade)
- [8. Treinamento da LLM](#8-treinamento-da-llm)
- [9. Formação de Resposta do Servidor](#9-formação-de-resposta-do-servidor)
- [10. Recursos Computacionais Disponíveis](#10-recursos-computacionais-disponíveis)
- [11. Tempo de Execução de Cada Fase](#11-tempo-de-execução-de-cada-fase)
- [12. Tempo de Treinamento × Qualidade](#12-tempo-de-treinamento--qualidade)
- [13. Tempo de Inferência × Qualidade](#13-tempo-de-inferência--qualidade)
- [14. Avaliação da LLM pelo Framework RAGAS](#14-avaliação-da-llm-pelo-framework-ragas)

---

## 2. Linguagens e Bibliotecas (com licenças)

### Linguagens Principais
- **Python 3.12** - Linguagem principal
- **JSON** - Formato de armazenamento de dados
- **HTML5, CSS3 e JavaScript** (sem frameworks) - Página da interface web em `frontend/static/index.html`

### Bibliotecas Utilizadas

| Biblioteca | Versão | Licença | Propósito |
|---|---|---|---|
| sentence-transformers | - | Apache 2.0 | Embeddings semânticos (modelo all-MiniLM-L6-v2) |
| transformers | - | Apache 2.0 | Carregamento de modelos do Hugging Face |
| torch | - | BSD | Motor de computação para ML |
| usearch | - | Elastic 2.0 | Indexação HNSW para busca vetorial |
| rank-bm25 | - | Apache 2.0 | Busca por palavras-chave (BM25) |
| pyarrow | - | Apache 2.0 | Leitura de arquivos Parquet |
| langchain-huggingface | - | MIT | Integração com HF |
| langchain-community | <0.4 | MIT | Componentes de LLM |
| ragas | 0.3.9 | Apache 2.0 | Framework de avaliação |
| gensim | - | LGPL 2.1 | Processamento de linguagem |
| scikit-learn | - | BSD 3-Clause | Utilitários de ML |
| numpy | - | BSD | Computação numérica |
| datasets | - | Apache 2.0 | Download de datasets HF |
| fastapi | - | MIT | Servidor web da interface (rotas `/`, `/info`, `/perguntar`) |
| uvicorn | - | BSD 3-Clause | Servidor ASGI que executa o FastAPI |
| pydantic | - | MIT | Validação da pergunta recebida pela interface (1 a 2000 caracteres) |
| httpx | - | BSD 3-Clause | Cliente HTTP (acesso ao Hub e testes da interface com `TestClient`) |
| openvino / openvino-genai / openvino-tokenizers | opcionais | Apache 2.0 | Inferência acelerada em CPU Intel e iGPU Iris Xe / UHD Graphics |
| onnxruntime / onnxruntime-directml | opcionais | MIT | Inferência de embeddings portátil (CPU e GPU via DirectML) |

**Resumo de Licenças**: Todas são permissivas/open-source, compatíveis com uso comercial e acadêmico.

---

## 3. Programas e Infraestrutura

### Servidor
### Servidor
- **Servidor web local**: FastAPI executado pelo Uvicorn em `127.0.0.1:8000` (só aceita conexões da própria máquina), em `frontend/servidor.py`
- **Cliente**: navegador comum abrindo `http://localhost:8000`; a página é um único arquivo estático (`frontend/static/index.html`), sem Node.js nem etapa de build
- **Alternativa sem navegador**: CLI interativa em `busca/consultar.py`
- Os dois usam o mesmo `SistemaRAG` de `busca/rag.py`; detalhes do fluxo no [item 9](#9-formação-de-resposta-do-servidor)
- **Portabilidade via `.env`**: `EMB_BACKEND`, `LLM_BACKEND`, `DISPOSITIVO_IA`, `LLM_QUANTIZACAO` e `LLM_MODEL` (padrão `auto`, com *fallback* silencioso para `torch`). Os wrappers nativos (`openvino.Core`, `openvino_genai.LLMPipeline`, `onnxruntime.InferenceSession`) rodam sem `optimum-intel`/`optimum-onnx` no `.venv` principal; a conversão única do LLM para **IR** (*Intermediate Representation*, formato nativo do OpenVINO com `openvino_model.xml` + `openvino_model.bin` já quantizados em `int4`/`int8`) ocorre em ambiente efêmero isolado via `uv run` (alternativa rápida: `python -m pip install uv` no `.venv`, evitando *downgrade* de `transformers>=5.23` e `huggingface_hub>=1.33`), salvando o cache em `busca/cache_indice/ov_*`.
  - **Uso sem `uv` com IR pré-exportado (Google Drive):** é possível baixar a pasta pronta `busca/cache_indice/ov_Qwen_Qwen2.5-1.5B-Instruct_int8_ff36b9d256/` ([Google Drive](https://drive.google.com/drive/folders/1q8n85VgS4NsVNPSgO-2emMMDIa23WSGl)), instalar apenas `busca/requirements-aceleracao.txt` e definir `LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct`. Isso pula o download dos pesos originais (~3 GB), dispensa `uv` e evita a carga pesada de CPU e RAM da conversão e quantização `int8` (mantendo apenas a compilação rápida de kernels na carga, compatível com `HF_HUB_OFFLINE=1`). **Ressalva de hardware:** aplica-se automaticamente (`auto`) a **GPUs Intel** (Iris Xe, UHD, Arc) e **CPUs Intel**; em **CPUs AMD ou ARM**, também funciona em CPU definindo `LLM_BACKEND=openvino`; **não acelera em GPUs NVIDIA (CUDA) ou AMD (ROCm/DirectML)**, que exigem `torch` ou `onnx` para usar GPU (embora ainda possam rodar a pasta `ov_*` em CPU com `LLM_BACKEND=openvino` e `DISPOSITIVO_IA=CPU`).

### Banco de Dados
- **Armazenamento em disco**:
  - Embeddings versionados (`.npz`): `busca/cache_indice/embeddings_v1_*.npz`
  - Passagens amostradas: `busca/cache_indice/passagens_gutenberg_v3_*.json`
  - Índice lexical completo (SQLite FTS5): `busca/cache_indice/gutenberg_lexical_v1.sqlite`
  - Resultados: `busca/resultados.json`, `busca/escala.json`, `busca/avaliacao_resultados.json`

### Modelos (Pré-treinados)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384 dim, CPU-friendly)
- **LLM para respostas**: HuggingFaceTB/SmolLM2-1.7B-Instruct (padrão, 1.7B parâmetros; opcionalmente substituível via `LLM_MODEL` no `.env` ou `--llm`, ex.: `Qwen/Qwen2.5-1.5B-Instruct`)
- **LLM opcional para avaliação**: Mesmo LLM ou outro especificado

### Processamento de Dados
- **Origem**: Project Gutenberg (dataset `sedthh/gutenberg_english`, livros em inglês de domínio público)
- **Formato**: Arquivos Parquet (colunas TEXT, SOURCE, METADATA)
- **Cache local**: dados/cache_gutenberg/ para downloads; dados/gutenberg_10gb/ para o texto extraído

---

## 4. Tema e Funcionamento do Sistema

### Tema: **Literatura e livros (Project Gutenberg) com RAG e LLM Local**

**Propósito**: Responder perguntas sobre livros usando:
1. Busca híbrida em um corpus de livros de domínio público (>= 10 GB de texto)
2. Geração de respostas por LLM local
3. Citação de fontes utilizadas

### Fluxo de Funcionamento

```text
┌──────────────┐
│  Pergunta    │
│  (Inglês)    │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────┐
│ 1. RECUPERAÇÃO (Hybrid Search)│
│   ├─ BM25 (palavras-chave)   │
│   └─ Semântico (embeddings)  │
└──────────────────────────────┘
       │
       ▼ [Top-3 trechos]
       │
       ▼
┌──────────────────────────────┐
│ 2. GERAÇÃO (LLM Local)       │
│   └─ SmolLM2 1.7B Instruct   │
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ Resposta com citações        │
│ [1] Fonte 1 / [2] Fonte 2    │
└──────────────────────────────┘
```

### Características Principais
- ✅ **Offline**: Não requer internet após download
- ✅ **Execução local viável**: Roda em notebooks Intel com CPU ou iGPU (ex.: `i5-1135G7` com 40 GB RAM + Iris Xe e `i5-13420H` com 16 GB RAM + Intel UHD Graphics)
- ✅ **Rastreável**: Sempre cita fontes
- ✅ **Avaliável**: Integrado com RAGAS
- ✅ **Interface web**: chat no navegador com as fontes em cartões e as citações `[n]` clicáveis

### Uso pela interface web

A pessoa digita a pergunta na página `http://localhost:8000`. O navegador envia a pergunta ao
servidor (`POST /perguntar`), que executa a recuperação e a geração descritas acima e devolve
eventos de progresso. A página mostra um indicador com o tempo decorrido e, ao final, a resposta, os
trechos usados como fonte (título, trecho, similaridade ou tipo de correspondência e link para o
livro no Project Gutenberg) e os tempos de busca e geração. Cada `[n]` da resposta é um link para o
trecho `n`, e as fontes citadas ficam destacadas. O topo da página indica o modo de busca ativo
(`Amostra` ou `Índice completo`) e o resumo de execução (`backend/dispositivo/quantização`).


---

## 5. Dados de Treinamento

### Fonte Original
- **Dataset**: `sedthh/gutenberg_english` (Hugging Face), licença MIT; livros de domínio público
- **Tamanho total**: 48.284 livros, ~18 GB de texto (download de 10,7 GB). A coleta extrai 10 GB (10.000.000.000 bytes)
- **Metadados por livro**: id (Text# do Gutenberg), título, autores; a URL da fonte é `https://www.gutenberg.org/ebooks/<id>`

### Download e Tamanho

**Corpus Atual**:
- **Coleta**: `python dados/coletar_gutenberg.py --contato SEU-EMAIL` gera `dados/gutenberg_10gb/` (`textos.txt`, `livros.jsonl`, `relatorio.json`)
- **Índice do RAG**: até 20.000 passagens por padrão, uma por livro; o corpus completo de 10 GB não é embedado (inviável em CPU) e é usado no estudo de escala (`benchmark.py`, `escala.py`)

**Como consultar**:
```bash
cd busca/
python consultar.py -p "Who is Captain Ahab?"
# Primeira execução: sorteia passagens e calcula embeddings (alguns minutos em CPU)
# Consultas seguintes: dominadas pelo tempo de geração do LLM
```

### Parâmetros de Amostragem
```python
# Arquivo: corpus.py
def amostrar_passagens(n, semente, dados=DADOS,
                       min_car=300, max_car=1000):
    """
    min_car: Caracteres mínimos por parágrafo (300)
    max_car: Caracteres máximos (1000)
    Retorna no máximo uma passagem por livro para ampliar a cobertura de títulos.
    """
```

O título e o autor entram no texto embedado e no BM25 (`texto_indexado`), para que perguntas como "Who wrote Moby Dick?" encontrem o livro.
O score impresso é similaridade cosseno, não uma porcentagem de confiança. Uma citação válida também não prova automaticamente que o trecho implica toda a resposta; a avaliação de suporte semântico permanece necessária.

---

## 6. Pré-processamento e Análise de Complexidade

### Etapas de Pré-processamento

#### **6.1 Coleta de Dados** (Arquivo: `dados/coletar_gutenberg.py`)
```text
1. Lista os Parquets do dataset via API do Hugging Face
2. Baixa em ordem, com retomada e verificação de SHA-256
3. Lê livros em lotes pequenos, deduplica por id
4. Grava textos.txt + livros.jsonl (offsets) até 10 GB
```

**Complexidade**: 
- **Tempo**: dominado pelo download (~6 GB para 10 GB de texto)
- **Espaço**: ~11 GB de cache + 10 GB de texto
- **I/O**: Linear com tamanho do dataset

#### **6.2 Amostragem de Passagens** (Arquivo: `busca/corpus.py`)
```text
1. Lê L registros de livros.jsonl e embaralha com semente
2. Seleciona m = min(L, max(2n, floor(1.35n))) livros e ordena por deslocamento (offset)
3. Lê cada livro sequencialmente em textos.txt via offset
4. Split por quebras de parágrafo (\n\n ou \n\n\n) e junta linhas internas
5. Filtra por tamanho (300-1000 caracteres) e por aparência de prosa
6. Sorteia no máximo 1 passagem por livro e deduplica por conteúdo
7. Salva em JSON com metadados (título, autores, url, id)
```

**Complexidade**:
- **Tempo**: $O(L + m \log m + B_{\text{lidos}})$, onde $L$ é o total de livros no catálogo, $m \log m$ vem da ordenação por offset para evitar *seeks* aleatórios e $B_{\text{lidos}}$ é o volume de bytes lidos
- **Espaço**: $O(L + n_{\text{passagens}})$

#### **6.3 Codificação (Embedding)** (Arquivo: `busca/rag.py`)
```text
1. Tokenização com SentenceTransformer (max_seq_length = 128)
2. Passagem pelo modelo (all-MiniLM-L6-v2) via OpenVINO, ONNX ou PyTorch
3. Normalização L2 explícita e conversão para float32
4. Armazenamento atômico em cache_indice/embeddings_v1_*.npz
```

**Complexidade**:
- **Tempo**: $O(N \cdot C_{\text{emb}})$ — ~16 a 43 passagens/segundo em CPU (20.000 passagens em ~8–21 min, dependendo da CPU/backend)
- **Espaço por embedding**: 384 floats × 4 bytes = **1,5 KB/passagem** (~30 MB para 20.000 passagens)

#### **6.4 Indexação (HNSW)** (Arquivo: `busca/rag.py`)
```text
1. Construção de grafo de proximidade em memória (usearch)
2. Parâmetros: connectivity (M) = 16, expansion_add (efConstruction) = 128, expansion_search = 64
```

**Complexidade**:
- **Tempo construção**: $O(N \log N)$ (~7 s para 20.000 passagens em `RELATORIO.md`, ~5 s para 10.000 e ~21 s para 100.000 em `escala.py`)
- **Espaço**: $O(N \cdot (d + M))$ (~62 MB total vetores + grafo para 20.000 passagens em `RELATORIO.md`)

#### **6.5 Índice Lexical Completo** (Arquivo: `busca/indice_lexical.py`)
```text
1. Segmenta os ~10 GB de textos.txt em janelas de 1.000 caracteres com sobreposição de 150
2. Insere em tabela virtual SQLite FTS5 (passagens) com checkpoint transacional por livro
3. Na consulta (--indice-completo), FTS5 retorna M = max(50, 10k) candidatos por BM25
4. Reranking semântico calcula embeddings de W janelas dos M candidatos e combina por RRF
5. Consultas "Who wrote <título>?" carregam o catálogo em hash map na 1ª chamada
```

**Complexidade**:
- **Construção**: Tempo $O(T \log V_{\text{fts}})$ (linear no total de tokens $T$, logarítmica no vocabulário/árvore B do FTS5); Espaço $O(T)$ em disco
- **Consulta (`recuperar`)**: $O\left(\sum_{t \in q} \text{postings}(t) \log M + W \cdot C_{\text{emb}} + M \log M\right)$, independente de $N$ na etapa semântica
- **Catálogo (`buscar_titulo`)**: $O(L)$ na primeira chamada; $O(1)$ nas chamadas subsequentes

### Resumo de Complexidade Geral

| Operação | Tempo | Espaço | Escalabilidade |
|---|---|---|---|
| Amostragem (`corpus.py`) | $O(L + m \log m + B_{\text{lidos}})$ | $O(L + n)$ | Quase linear ✓ |
| Embedding (`rag.py`) | $O(n \cdot C_{\text{emb}})$ | $O(n \cdot 384)$ | Linear ✓ |
| HNSW (`usearch`) | $O(n \log n)$ | $O(n \cdot (d + M))$ | Sublinear na busca ✓ |
| Índice Lexical FTS5 | $O(T \log V_{\text{fts}})$ | $O(T)$ em disco | Independente de $N$ no reranking ✓ |


---

## 7. Busca Semântica: Tempo e Complexidade

### Métodos Comparados (Benchmark — Project Gutenberg)

**Setup**: 20.000 passagens do Project Gutenberg (`dados/gutenberg_10gb/textos.txt`), Intel Core i5-1135G7 + Intel Iris Xe Graphics, 300 queries

| Método | Recall@1 | Recall@10 | MRR@10 | Construção | Latência/consulta | Complexidade de Busca |
|---|---:|---:|---:|---:|---:|---|
| **TF-IDF (`benchmark.py`)** | 0,987 | 0,997 | 0,989 | 1,22 s | 3,03 ms | $O(\text{nnz}_{\text{visitado}} + N + k \log k)$ (produto esparso + densificação) |
| **Word2Vec Médio** | 0,267 | 0,463 | 0,323 | 9,10 s | 0,60 ms | $O(N \cdot d + k \log k)$ |
| **Sentence + Força Bruta (`torch/CPU`)** | 0,563 | 0,667 | 0,594 | 388,69 s | 17,94 ms (busca: 1,53 ms) | $O(N \cdot d + k \log k)$ + 1 inferência |
| **Sentence + Força Bruta (`openvino/GPU`)** | 0,563 | 0,663 | 0,593 | **77,95 s** (**5,0×**) | 19,63 ms (busca: 1,29 ms) | $O(N \cdot d + k \log k)$ + 1 inferência |
| **Sentence + HNSW (`torch/CPU`)** | 0,527 | 0,620 | 0,554 | 395,53 s | 16,24 ms (busca: 0,78 ms) | **$O(\log N \cdot d)$** + 1 inferência ✓ |
| **Sentence + HNSW (`openvino/GPU`)** | 0,530 | 0,620 | 0,557 | **87,73 s** (**4,5×**) | **13,47 ms** (busca: 0,80 ms) | **$O(\log N \cdot d)$** + 1 inferência ✓ |

### Análise Detalhada para HNSW

#### **Decomposição da Latência (20k passagens, `openvino/GPU`)**:
```text
Latência Total: 13.47 ms
├─ Encoding da query: ~12.67 ms (94%)  ← GARGALO
└─ Busca HNSW pura:   ~0.80 ms (6%)
```

#### **Escalabilidade Vetorial (`busca/escala.py` — 384 dim, vetores sintéticos agrupados)**:

| N | Força Bruta | HNSW | Recall@10 HNSW | Construção HNSW |
|---|---:|---:|---:|---:|
| 10.000 | 0,76 ms | 0,50 ms | 0,995 | 2,3 s |
| 100.000 | 9,01 ms | 0,62 ms | 0,993 | 25,4 s |
| 500.000 | 38,94 ms | 1,19 ms | 0,871 | 216,0 s |
| 1.000.000 | 86,45 ms | **1,85 ms** | 0,770 | 617,2 s |

#### **Complexidade Teórica**:

**Força Bruta**:
- O(N × d) = O(20.000 × 384) ≈ 7.7M operações
- Tempo real: ~15 ms (510k ops/ms)

**HNSW**:
- O(log N × d) = O(14 × 384) ≈ 5.4k operações
- **Redução**: 1.400x mais rápido (teoricamente)
- **Prática**: 21x mais rápido (com tradeoff de 2.5% recall)

### Hybrid Search Implementado

Combina BM25 + Semântico:
- BM25: ~0.5 ms (muito rápido)
- Semântico: ~13.3 ms
- **Total**: ~13.8 ms

---

## 8. Treinamento da LLM

### ⚠️ IMPORTANTE: Não há Treinamento Local

O modelo LLM **não foi treinado** pelo projeto. É um modelo **pré-treinado** utilizado como-é.

### Modelo Utilizado
```text
Nome: HuggingFaceTB/SmolLM2-1.7B-Instruct
Parâmetros: 1.7 bilhões
Treinado por: Hugging Face
Licença: APACHE 2.0
Otimizado para: CPU
```

### Pipeline de Utilização (arquivo `busca/rag.py`)

```python
def completar(self, msgs, max_tokens):
    """Gera resposta usando modelo pré-treinado."""
    ids = self.tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    )
    with torch.no_grad():
        saida = self.lm.generate(
            **ids, 
            max_new_tokens=max_tokens,  # Máximo 200
            do_sample=False,            # Determinístico (top-1)
            repetition_penalty=1.1      # Evita repetição
        )
    return self.tok.decode(saida[0][...], skip_special_tokens=True)
```

### Dados Fornecidos ao Modelo

```text
SYSTEM PROMPT:
"You are a helpful assistant that answers questions ONLY using 
the provided numbered context passages. You MUST cite sources..."

INPUT:
"Context:
[1] (Moby Dick by Melville, Herman) Call me Ishmael...
[2] (Moby Dick by Melville, Herman) Captain Ahab was...

Question: Who is Captain Ahab?"
```

### Complexidade de Inferência

| Aspecto | Valor |
|---|---|
| **Tamanho do modelo** | 1.7B parâmetros |
| **Memória necessária** | ~6.8 GB (float32) |
| **Velocidade** | ~15-50 tokens/segundo em CPU |
| **Latência por token** | ~20-70 ms |

```text
Geração de texto (autoregressive):

Para cada token (iteração):
  1. Embedding do token anterior
  2. Passagem pela rede: O(1.7B parâmetros)
  3. Softmax no vocab: O(50.000)

Complexidade TOTAL (autoregressivo com KV-cache, prompt $P$ tokens e $G$ tokens gerados):
- Teórica: $O((P+G) \cdot \text{params} + (P+G)^2 \cdot d_{\text{model}} \cdot n_{\text{layers}})$
- Prática em CPU: ~20-50 segundos (dependendo da quantização/backend)
- Em GPU: ~1-2 segundos
```

### O que é quantização?

Os pesos de um LLM (como os **1,7 bilhão de parâmetros** do `SmolLM2-1.7B-Instruct`) são números decimais guardados por padrão em **32 bits (`fp32`)**. **Quantização** é reduzir quantos bits representam cada peso, comprimindo o modelo com perda mínima de qualidade:

| Precisão | Bits / peso | Tamanho aproximado (`SmolLM2-1.7B`) | Efeito prático |
|---|---:|---:|---|
| `fp32` | 32 bits (4 bytes) | ~6,8 GB | Padrão exato, mais pesado e lento |
| `fp16` | 16 bits (2 bytes) | ~3,4 GB | 2× menor; bom para GPUs |
| `int8` | 8 bits (1 byte) | ~1,7 GB | 4× menor; quase sem perda de qualidade |
| `int4` | 4 bits (0,5 byte) | ~0,9–1,1 GB | ~6–7× menor; ideal para CPU e Intel Iris Xe |

**Por que isso acelera tanto a geração?**  
Gerar texto token por token é gargalado pela **velocidade de leitura da RAM** (a cada token gerado, a CPU/GPU precisa ler praticamente todos os pesos do modelo). Em `int4`, a Iris Xe lê ~6× menos dados da memória compartilhada por token — então responde bem mais rápido e sem estourar a RAM.

### Avaliação de Qualidade

**Teste realizado: 20 perguntas, corpus 5k passagens**

```text
Métricas Próprias:
- Hit@3: 100% (fonte correta nos 3 primeiros)
- MRR@3: 100% (fonte sempre em posição 1)
- Respostas com citação: 45% (problema de parsing)
- Citação da fonte correta: 45%

Tempo de geração:
- Média: 52 seg
- P95: 189 seg
- Status: Aceitável para interativo
```


---

## 9. Formação de Resposta do Servidor

### Arquitetura de Geração (arquivo: `busca/rag.py`)

```python
def gerar(self, pergunta, trechos):
    # 1. Construir contexto
    contexto = "\n\n".join(
        f"[{i}] ({t['titulo']}) {t['texto']}"
        for i, t in enumerate(trechos, 1)
    )
    
    # 2. Montar mensagens (system + few-shot multipassagem/abstenção + contexto com lembrete de citação)
    nums = ", ".join(f"[{i}]" for i in range(1, len(trechos) + 1))
    instrucao = (
        f"Answer in 1-3 sentences using only the passages above, "
        f"ending every sentence with the supporting passage number ({nums})."
    )
    msgs = [
        {"role": "system", "content": PROMPT_SISTEMA},
        *EXEMPLOS_GERACAO,
        {"role": "user", "content": f"Context:\n{contexto}\n\nQuestion: {pergunta}\n{instrucao}"}
    ]
    
    # 3. Gerar resposta
    return self.completar(msgs, self.max_tokens)
```

### Fluxo Detalhado

```text
┌─ ENTRADA: "Who is Captain Ahab?"
│
├─ ETAPA 1: RECUPERAÇÃO
│  ├─ 1.1 Encoding da query (SentenceTransformer / OpenVINO / ONNX)
│  ├─ 1.2 Busca lexical (BM25 em memória no modo amostra ou FTS5 no --indice-completo)
│  ├─ 1.3 Busca vetorial (HNSW no modo amostra ou reranking em janelas no --indice-completo)
│  └─ 1.4 Fusão de rankings, deduplicação lexical entre edições (--indice-completo) e filtragem por --min-sim
│
├─ ETAPA 2: FORMAÇÃO DO CONTEXTO
│  └─ Concatenar até k=3 trechos distintos com títulos, autores e URLs
│
├─ ETAPA 3: CHAMADA DO LLM
│  ├─ Input: System prompt + EXEMPLOS_GERACAO (few-shot multipassagem/abstenção) + Context + Question + lembrete [n]
│  ├─ Geração autoregressive token-por-token
│  └─ Output: Resposta citando [1], [2], ... ou abstenção padrão
│
├─ ETAPA 4: PÓS-PROCESSAMENTO
│  ├─ Validar referências citadas via regex (validar_citacoes; rejeitar resposta sem citação válida)
│  └─ Validar suporte semântico mínimo (_validar_suporte_semantico, limiar 0,30)
│
└─ SAÍDA:
   {
     "resposta": "Captain Ahab is the monomaniacal captain of the Pequod... [1]",
     "trechos": [...],
     "citados": [1],
     "tempo_recuperacao": 0.042,
     "tempo_geracao": 36.8,
     "tempo_total": 36.84
   }
```

### Roteamento por Categoria de Pergunta (`--indice-completo`)

No modo `--indice-completo`, `SistemaRAG.perguntar()` não envia todas as perguntas ao LLM — apenas consultas abertas sobre personagens ou enredo passam pela etapa de geração:

| Categoria de pergunta | Caminho no código | Usa o LLM? | Observação |
| :--- | :--- | :---: | :--- |
| **Autoria da obra** (`Who wrote ...?`) | Consulta determinística ao catálogo de metadados | Não | Retorna autor inequívoco e `[1]` sem gerar texto via LLM. |
| **Localização de frase/trecho** (`Where does the phrase ... come from?`) | Busca literal ou aproximada no índice FTS5 | Não | Retorna correspondência comprovada e `[1]` sem acionar o LLM. |
| **Personagem / enredo** (`Who is Captain Ahab in Moby Dick?`) | Pipeline RAG completo (recuperação híbrida + LLM + validação) | **Sim** | Único fluxo sujeito a deduplicação entre edições (para evitar cenas repetidas ocupando os 3 slots) e à necessidade de síntese com citação explícita `[n]`. |

### Análise de Complexidade (Modo Amostra com BM25 + HNSW)

| Operação | Complexidade | Observação |
|---|---|---|
| Encoding query | $O(S^2 \cdot d_{\text{model}} + S \cdot d_{\text{model}}^2)$ | Transformer encoder ($S \le 128$, independente de `vocab`) |
| BM25 (`BM25Okapi`) | $O(N \cdot \lvert q \rvert + N + k \log k)$ | Varredura em memória + seleção top-$3k$ com `argpartition` |
| HNSW search (`usearch`) | $O(\log N \cdot d)$ | Busca aproximada no grafo HNSW |
| Fusão rankings | $O(k \log k)$ | Ordenação de até $6k$ candidatos |
| Concatenação | $O(k \cdot L_{\text{trecho}})$ | Montagem do prompt |
| LLM inference | $O((P+G)\cdot\text{params} + (P+G)^2 \cdot d_{\text{model}} \cdot n_{\text{layers}})$ | Gargalo dominante (>99% do tempo) |

**Gargalo**: 99.9% do tempo é geração LLM, não recuperação!

### Servidor web (arquivo: `frontend/servidor.py`)

Na interface web, a resposta é formada pelo mesmo `SistemaRAG.perguntar()` da CLI. O servidor só
recebe a pergunta, controla a concorrência e entrega o resultado ao navegador:

```
Navegador                         Servidor (FastAPI)                      SistemaRAG (busca/rag.py)
   │  POST /perguntar {pergunta}       │                                        │
   ├──────────────────────────────────►│ valida (Pydantic, 1–2000 caracteres)   │
   │◄── status "Consultando…" ─────────┤                                        │
   │                                   │ aguarda a trava (threading.Lock)       │
   │                                   ├──────── perguntar(pergunta) ──────────►│ metadados, frase literal
   │                                   │                                        │ ou busca + LLM + citações
   │                                   │◄────── resposta, trechos, citados ─────┤
   │◄── fontes (trechos e aviso) ──────┤                                        │
   │◄── resposta (texto completo) ─────┤  (só quando o LLM está carregado)      │
   │◄── fim (citados, tempos) ─────────┤                                        │
```

- **Formato da resposta:** NDJSON, um objeto JSON por linha (`status`, `fontes`, `resposta`, `fim`
  ou `erro`). O navegador lê as linhas conforme chegam com `fetch` + `ReadableStream`.
- **Uma pergunta por vez:** uma `threading.Lock` serializa as chamadas a `perguntar()`. Há um único
  LLM local (CPU ou GPU integrada); duas gerações em paralelo só disputariam os mesmos recursos e
  ficariam ambas mais lentas. Perguntas simultâneas esperam na fila.
- **Resposta inteira, não token a token:** o texto só é enviado depois que `perguntar()` valida as
  citações, para não mostrar um texto provisório que depois seria trocado por "contexto
  insuficiente". O custo é que a tela mostra apenas o indicador de progresso durante a geração.
- **Erros:** qualquer exceção vira um evento `erro` exibido na página, em vez de cortar a conexão.
- **Na página:** o texto é escapado (evita injeção de HTML), cada `[n]` válido vira link para o
  cartão da fonte `n` e as fontes citadas são destacadas.

| Etapa no servidor/página | Complexidade | Tempo |
|---|---|---|
| Validação da pergunta | O(tamanho da pergunta) | desprezível |
| Espera na trava | O(perguntas na fila × tempo de uma pergunta) | 0 com um usuário |
| `perguntar()` (busca + LLM) | ver tabela acima | domina o total |
| Serialização em JSON | O(k × tamanho do trecho) | < 1 ms |
| Montagem da tela (escape + citações) | O(tamanho da resposta + k) | < 1 ms |

O servidor e a página acrescentam um custo desprezível ao tempo do `SistemaRAG`.

---

## 10. Recursos Computacionais Disponíveis

### Hardware do Projeto

```text
CPU: 11th Gen Intel Core i5-1135G7 (4 cores / 8 threads, 2.40 GHz)
GPU integrada: Intel Iris Xe Graphics (suportada via OpenVINO e OpenVINO GenAI)
RAM: 40 GB (39,73 GiB utilizáveis)
Armazenamento: SSD NVMe Kingston KC3000 (KINGSTON SKC3000S1024G, 1 TB / 953,9 GiB; partição C: de 546,6 GiB)
Sistema Operacional: Windows 11
```

> **Compatibilidade com a máquina anterior da equipe (`i5-13420H`, 16 GB RAM, 13ª geração):** a GPU integrada do `i5-13420H` (**Intel UHD Graphics**, arquitetura Xe-LP) também é suportada nativamente pelo OpenVINO e OpenVINO GenAI (`EMB_BACKEND=auto`, `LLM_BACKEND=auto`, `DISPOSITIVO_IA=GPU`), mantendo *fallback* automático para `torch/CPU` caso os runtimes opcionais não estejam instalados.

### Armazenamento Utilizado na Execução do Projeto

| Componente | Caminho | Espaço medido | Observação |
|---|---|---:|---|
| **Corpus extraído** | `dados/gutenberg_10gb/` | `9,32 GiB` (`10,00 GB`) | 27.740 livros em UTF-8 + metadados |
| **Índice lexical completo (FTS5)** | `busca/cache_indice/gutenberg_lexical_v1.sqlite` | `23,30 GiB` (`~25,02 GB`) | 11.741.620 passagens indexadas |
| **Modelos OpenVINO IR** | `busca/cache_indice/ov_*` | `3,94 GiB` (3 caches) | `0,88 GiB` (`SmolLM2 int4`), `1,60 GiB` (`SmolLM2 int8`), `1,46 GiB` (`Qwen2.5-1.5B int8`); ~0,9–1,6 GiB para 1 modelo |
| **Caches amostrais** | `busca/cache_indice/passagens_*`, `embeddings_v1_*.npz` | `~0,05 GiB` | Amostra de 20.000 passagens e vetores L2 |
| **Ambiente virtual** | `.venv/` | `1,34 GiB` | 128 pacotes (Python 3.12 + OpenVINO + PyTorch) |
| **Total persistente no workspace** | — | **`~38,0 GiB`** | Após `--limpar-parquets`; durante a coleta são necessários `~6 GB` temporários adicionais em `dados/cache_gutenberg/` |

### O que é Possível Fazer

#### ✅ **Viável AGORA:**

1. **Corpus de até 100.000 passagens**
   - Tempo embedding: ~40-50 min
   - Espaço disco: ~500 MB
   - Espaço RAM: ~2-3 GB durante indexação

2. **Busca em tempo real**
   - Latência recuperação: ~15 ms
   - Latência geração: ~50-100 seg

3. **Avaliação com RAGAS**
   - ~20-30 perguntas avaliadas em 20-30 min

4. **Benchmarking comparativo**
   - Até 20k passagens sem problemas

#### ⚠️ **Viável com Limitações:**

5. **Corpus de 200.000-500.000 passagens**
   - Tempo embedding: ~2-4 horas
   - Espaço disco: ~1-2.5 GB
   - ⚠️ RAM pode ficar apertada (swap necessário)

#### ❌ **Não Viável:**

6. **Corpus 1.000.000+ passagens**
   - Embedding: ~8+ horas
   - Memória necessária: ~10+ GB apenas para embeddings
   - **Solução**: GPU necessária

7. **Fine-tuning do LLM**
   - Requer GPU (VRAM > 8 GB)

### Recomendações

```text
AGORA (Imediato):
  ✓ Corpus de 10 GB e índice RAG de 20.000 livros/passagens
  ✓ Hybrid search
  ✓ SmolLM2-1.7B (otimizado para CPU)

PRÓXIMO PASSO (Fácil, 1-2h):
  ✓ Aumentar para 50.000 passagens
  ✓ Otimizar HNSW

MÉDIO PRAZO (Com preparação):
  ✓ GPU cloud (~-10/mês)
  ✓ Quantizar modelo (int8)
  ✓ Testar modelos menores

LONGO PRAZO:
  ✓ GPU local (RTX 4060: ~)
  ✓ Cache de embeddings
  ✓ Índices em disco
```


---

## 11. Tempo de Execução de Cada Fase

### Fases do Projeto (medidas no corpus Project Gutenberg — i5-1135G7, 40 GB RAM, Iris Xe, SSD NVMe)

| Fase | Atividade | Tempo Medido | Notas |
|---|---|---|---|
| **COLETA** | Download Gutenberg até 10 GB de texto | ~48 min | 27.740 livros, 10.000.018.524 bytes |
| **ÍNDICE FTS5** | Construção completa (`indice_lexical.py`) | 2.176 s (~36 min) | 11.741.620 passagens, ~25,0 GB em disco |
| **AMOSTRAGEM** | Seleção de 15.251 passagens (`corpus.py`) | 1.019 s (~17 min) | Leitura ordenada em 27.740 livros |
| **EMBEDDING** | Codificação de 20k passagens (`benchmark.py`) | 78 s (OpenVINO GPU) / 389 s (Torch CPU) | **5,0× mais rápido** na Iris Xe |
| **INDEXAÇÃO** | HNSW grafo (`usearch`, 20k passagens) | 7–10 s | $O(n \log n)$ |
| **RECUPERAÇÃO** | BM25 + HNSW (`avaliar.py`) | ~141–161 ms | Inclui encoding da query |
| **GERAÇÃO** | LLM resposta (`SmolLM2-1.7B-Instruct`) | **5,36 s** (OpenVINO GPU int4) / **38,63 s** (Torch CPU fp32) | **7,2× mais rápido** na Iris Xe |

### Timeline Completo

```text
PRIMEIRA EXECUÇÃO (do zero ao resultado):
├─ Setup + download: 15-20 min
├─ Embedding: 2-3 min
├─ PRIMEIRA QUERY (warmup): 60-90 seg
└─ PRÓXIMAS QUERIES: 50-150 ms recuperação + 20-200 seg geração
  
TEMPO TOTAL ATÉ PRIMEIRA RESPOSTA: ~20-30 minutos

EXECUÇÕES SUBSEQUENTES:
├─ Importação Python: 2-3 seg
├─ Carregamento cache: 1-2 seg
├─ Query: 50-200 seg (dominado por LLM)

TEMPO POR QUERY: ~1-3 minutos
```

### Gráfico de Escalabilidade

| Corpus N | Embedding | Indexação | Busca | Geração | TOTAL |
|---|---|---|---|---|---|
| 5.000 | 2 min | 30 seg | 15ms | 20-50s | ~3-4 min |
| 20.000 | 8 min | 1 min | 15ms | 20-50s | ~9-10 min |
| 50.000 | 20 min | 3 min | 15ms | 20-50s | ~23-24 min |
| 100.000 | 40 min | 6 min | 20ms | 20-50s | ~46-47 min |

**Hardware dos benchmarks atuais**: `i5-1135G7` (40 GB RAM, Intel Iris Xe, SSD NVMe). Os tempos estimados de referência da v1.0 em `i5-13420H` (16 GB RAM) referiam-se ao corpus Wikipédia em `torch/CPU` puro.

### Interface web

| Fase | Tempo medido (Project Gutenberg) | Notas |
|---|---|---|
| Subir o servidor (`--sem-llm`) | segundos | Carrega apenas o modelo de embeddings e abre o índice |
| Subir o servidor com LLM | poucos segundos (`openvino/GPU` com IR em cache) / ~1 min (`torch/CPU fp32`) | Na 1ª execução de amostra sem cache, somam-se amostragem e embeddings |
| Pergunta na tela (`--indice-completo`, `openvino/GPU int8` + `Qwen2.5-1.5B`) | **~5–10 s** | Busca híbrida `~1,4 s` + geração `~4–9 s` |
| Pergunta na tela (referência `torch/CPU fp32`, `SmolLM2-1.7B`) | **~38,6 s** médios | Sem aceleração OpenVINO |
| Servidor + página | `< 1 ms` | Validação Pydantic, NDJSON e montagem dos cartões/citações |

---

## 12. Tempo de Treinamento × Qualidade

### ⚠️ CONTEXTO: Não há Treinamento Neste Projeto

Como o modelo é **pré-treinado**, não há "tempo de treinamento" ajustável. Mas há "tempo de inferência" do LLM:

### Relação Tempo de Inferência × Qualidade

#### **Parâmetro: `max_tokens = 200` (padrão)**

```text
max_tokens | Tempo Médio | Qualidade Resposta
────────────────────────────────────────────
50         | 5-10 seg   | 🔴 Muito curta
100        | 10-20 seg  | 🟡 Incompleta
200        | 20-50 seg  | 🟢 ÓTIMO (PADRÃO)
500        | 50-120 seg | 🟢 Detalhada
1000       | 120-300 seg| 🟡 Muito longa
```

### Impacto da Qualidade do Corpus

**Mais relevante**: Tempo investido na **preparação do corpus**:

| Atividade | Tempo | Impacto Qualidade |
|---|---|---|
| Corpus padrão (min_car=300) | Mínimo | 🔴 Baixa - Fatos curtos faltam |
| Ajustar min_car para 80 | +2 min | 🟢 Média |
| Aumentar de 5k para 50k | +15 min | 🟢 Alta - Mais cobertura |
| Query expansion | +3 min (setup) | 🟡 Média |
| Semantic chunking | +1 hora | 🟢🟢 Muito Alta |

**Conclusão**: Tempo de **pré-processamento** impacta mais que tempo de **LLM inference**

---

## 13. Tempo de Inferência × Qualidade

### Tempo de Inferência (Geração de Resposta)

O tempo é **aproximadamente linear** com qualidade, até um ponto:

```text
Tempo Inferência | Qualidade | Observação
─────────────────────────────────────────────
< 5 seg          | 🔴 Ruim    | Respostas truncadas
5-20 seg         | 🟡 Média   | Faltam detalhes
20-50 seg ✅     | 🟢 Boa     | PONTO ÓTIMO
50-100 seg       | 🟢 Muito Boa| Mais detalhes
100+ seg         | 🟡 Decai   | Sem melhoria real
```

### Componentes do Tempo

```text
Tempo Total = Encoding Query + Busca + Geração Tokens

Encoding Query:        ~12 ms (fixo)
Busca:                 ~1 ms (fixo)
Geração por Token:     ~20-70 ms/token (linear)

Para max_tokens = 200 (~70 tokens gerados em CPU a ~500-700 ms/token):
├─ Encoding: ~12 ms
├─ Busca: ~1 ms
└─ Geração: ~70 tokens × ~500 ms ≈ 35.000 ms (35 s)
───────────────────────────────────────────────────
TOTAL: ~35–52 s (dependendo da CPU/backend e tamanho do prompt)
```

### Impacto: Temperature e Sampling

O projeto usa `do_sample=False` (determinístico):
- Mais rápido (sem sampling)
- Menos criativo
- Mais reproduzível
- Menos alucinação

Se fosse \do_sample=True\ com `temperature=0.7`:
- +10-20% mais lento
- +30% mais criativo
- +20% mais alucinação

### Avaliação RAG Ponta a Ponta (`busca/avaliar.py --sem-ragas --min-sim 0.4`, 20 perguntas, amostra 15.251 passagens)

| Configuração | Recuperação média | Geração média | Geração mediana | Geração P95 | Hit@3 | Abstenções |
|---|---:|---:|---:|---:|---:|---:|
| `torch/CPU (fp32)` | 0,161 s | 38,63 s | 38,62 s | 83,40 s | 0,15 | 75% |
| `openvino/GPU (int4)` | 0,141 s | **5,36 s** (**7,2×**) | **5,59 s** | **10,97 s** (**7,6×**) | 0,15 | 100% |

**Conclusão**: A combinação OpenVINO + OpenVINO GenAI reduz o gargalo de geração para **3,54–3,74 s** (`int8`) na Intel Iris Xe (**5,2×–5,5×** mais rápido que `torch/CPU fp32`). Em teste controlado com 4 perguntas literárias, `SmolLM2-1.7B-Instruct` apresentou baixa adesão ao marcador obrigatório `[1]` tanto em `int8`/`int4` quanto em `fp32`, enquanto `Qwen/Qwen2.5-1.5B-Instruct` (`int8`) com o `PROMPT_SISTEMA` estruturado obteve `3,54 s` médios e **4/4 de acerto** (3/3 citações válidas + recusa correta sem contexto). Para mitigar citações alucinadas independentemente do LLM escolhido, `busca/rag.py` aplica validação semântica (`_validar_suporte_semantico`), e `LLM_MODEL` / `LLM_QUANTIZACAO` permitem alternar o modelo e a precisão via `.env`. Já o `Hit@3 = 0,15` na amostra de 15k passagens evidencia a importância do `--indice-completo` (`gutenberg_lexical_v1.sqlite`) para perguntas sobre qualquer um dos 27.740 livros.

---

## 14. Avaliação da LLM pelo Framework RAGAS

### Avaliação Realizada (`--indice-completo`, `openvino/GPU int8`)

- **Dataset:** 12 primeiras perguntas de `busca/avaliacao_dataset.json` (todas presentes nos 27.740 livros do corpus)
- **Arquivo de saída:** `busca/avaliacao_resultados_ragas.json` (`k = 3`, `min_sim = 0.4`, `--indice-completo`)
- **Modelo gerador e juiz local:** `Qwen/Qwen2.5-1.5B-Instruct` via `openvino/GPU (int8)` (`ModeloLLMOpenVINOGenAI` adaptado para LangChain)
- **Framework:** RAGAS v0.3.9 + `all-MiniLM-L6-v2` (embeddings)

### Resultados — Métricas Próprias e Tempos (✅ Completas)

| Métrica | Valor | Observação |
|---|---:|---|
| `hit_at_k` (`k = 3`) | **0,500** | 6/12 perguntas recuperaram o livro esperado no top-3 |
| `mrr` | **0,417** | Posição recíproca média |
| `respostas_com_citacao` | **0,417** | 5/12 respostas passaram nas validações sintática e semântica |
| `citacao_da_fonte_correta` | **0,250** | 3/12 citaram exatamente o `id_esperado` |
| `respostas_abstidas` | **0,583** | 7/12 abstenções conservadoras (`SEM_CONTEXTO`) |
| Recuperação média / P95 | **10,47 s** / 17,99 s | FTS5 + reranking semântico em 50 candidatos (11,7M passagens) |
| Geração média / P95 | **6,77 s** / 10,91 s | `Qwen2.5-1.5B-Instruct` em `openvino/GPU (int8)` |
| Total médio / P95 | **17,33 s** / 24,43 s | Inclui recuperação + geração + validação semântica |

### Resultados — Métricas RAGAS (✅ Completas, `0` NaN)

| Métrica RAGAS | Média (12 perguntas) | `NaN` | Significado |
|---|---:|---:|---|
| `faithfulness` | **0,250** | 0 | Fidelidade das afirmações aos trechos recuperados (abstenções pontuam 0) |
| `answer_relevancy` | **0,285** | 0 | Aderência da resposta à pergunta (abstenções reduzem a média) |
| `llm_context_precision_with_reference` | **0,681** | 0 | Precisão dos contextos recuperados frente à resposta de referência |
| `context_recall` | **0,917** | 0 | Cobertura da resposta de referência pelos trechos recuperados |
| `semantic_similarity` | **0,330** | 0 | Similaridade semântica resposta × referência |

### Análise dos Resultados

1. **Execução integral sem falhas:** ao adaptar o wrapper `ModeloLLMOpenVINOGenAI` para o RAGAS em `openvino/GPU (int8)`, todas as 60 avaliações (`12 × 5` métricas) completaram em ~13 minutos com **`0` erros/`NaN`**.
2. **Recuperação forte em cobertura (`context_recall = 0,917`, `context_precision = 0,681`):** o índice completo FTS5 + reranking semântico encontra passagens semanticamente ricas; parte dos `hit_at_k = 0,50` ocorre porque o Gutenberg traz múltiplas edições/obras com termos semelhantes (ex.: outro `id` do mesmo texto ou obra correlata).
3. **Impacto das abstenções nas médias (`58,3%` de `SEM_CONTEXTO`):** perguntas muito específicas do dataset (como detalhes pontuais de poemas ou capítulos isolados) levam o sistema conservador (`_validar_suporte_semantico` + regras estritas de `PROMPT_SISTEMA`) a preferir `SEM_CONTEXTO` em vez de alucinar — o que protege contra respostas inventadas, mas penaliza `faithfulness`, `answer_relevancy` e `semantic_similarity` no RAGAS (que pontuam `0` em abstenções).
4. **Limitação do juiz local compacto (1,5B):** um modelo de 1,5B parâmetros consegue devolver JSON válido (`0` `NaN`), porém é menos rigoroso na decomposição de afirmações do que juízes comerciais maiores (7B+ / APIs).

### Exemplo de Teste Individual (Histórico — Corpus Wikipédia)

> Nota: o exemplo abaixo ilustra a falha de contexto ocorrida na versão anterior (Wikipédia), que motivou a validação obrigatória de citações (`validar_citacoes`) e a recusa via `SEM_CONTEXTO`.

```text
Pergunta (histórica, corpus Wikipédia): "What is the capital of France?"

RESULTADO:
┌─ Fontes recuperadas:
│  [1] Whalley, Surrey (0.503) ❌ Irrelevante
│  [2] Geography (0.442) ❌ Irrelevante
│  [3] Woollard (0.368) ❌ Irrelevante
│
├─ Resposta sem validador:
│  "The capital of France is Paris." (sem citação -> hoje rejeitada com SEM_CONTEXTO)
```

### Conclusões da Avaliação

1. ✅ **RAGAS completo executado localmente:** 12 perguntas avaliadas em `openvino/GPU (int8)` com **`0` `NaN`** em todas as 5 métricas.
2. ✅ **Boa cobertura de contexto (`context_recall = 0,917`):** o `--indice-completo` FTS5 localiza conteúdo relevante nos 27.740 livros sem gaps da amostra.
3. ✅ **Postura conservadora anti-alucinação:** o par `validar_citacoes` + `_validar_suporte_semantico` evita respostas inventadas (`58,3%` de abstenção quando o trecho não sustenta claramente a pergunta).
4. ⚠️ **Compromisso entre segurança e recall de resposta:** abstenções reduzem as médias de `faithfulness` (`0,250`) e `answer_relevancy` (`0,285`).

### Recomendações para Melhoria

- **Concluídos:**
  - ✅ **Parser/normalização de citações + validação semântica:** `validar_citacoes` normaliza formatos (`[Passage n]`, `Source [n]`) e `_validar_suporte_semantico` (limiar cosseno `0,30`, por frase citada) rejeita citações sem suporte no trecho.
  - ✅ **Cobertura de corpus e deduplicação:** benchmarks em 20k passagens e índice lexical completo FTS5 (27.740 livros / 11,7M passagens) com deduplicação lexical por 4-gramas entre edições.
  - ✅ **Ajuste de prompt e escolha do LLM:** `PROMPT_SISTEMA` estruturado em 5 regras, `EXEMPLOS_GERACAO` (*few-shot* multipassagem + abstenção) e suporte a `LLM_MODEL` / `--llm` (ex.: `Qwen/Qwen2.5-1.5B-Instruct`).
  - ✅ **Avaliação completa com RAGAS:** executada em `busca/avaliar.py` com 12 perguntas sobre `--indice-completo` usando `Qwen/Qwen2.5-1.5B-Instruct` em `openvino/GPU (int8)` (`0` `NaN`).
- **Pendentes:**
  - ⏳ **Modelos maiores / treino específico para RAG:** avaliar LLMs de 3–7B ou *fine-tuning* dedicado em hardware com maior VRAM/RAM.
  - ⏳ **Modelos OpenVINO IR pré-convertidos do Hub:** suportar download direto de repositórios oficiais `OpenVINO/` no Hugging Face Hub (ex.: `OpenVINO/Qwen2.5-1.5B-Instruct-int8-ov`), dispensando a conversão local via `uv`/`optimum-intel` para modelos já pré-exportados.

---

## Resumo Executivo

### Respostas Rápidas

| # | Pergunta | Resposta |
|---|---|---|
| **2** | Linguagens/Bibliotecas | Python 3.12 + HTML5/CSS3/JS + libs open-source (FastAPI, Uvicorn, Pydantic, PyTorch, OpenVINO/ONNX opcionais) |
| **3** | Programas/Infraestrutura | Servidor web local (FastAPI + Uvicorn) com página HTML/JS, CLI alternativa (`consultar.py`), caches `.npz`/`.sqlite` e portabilidade `.env` |
| **4** | Tema/Funcionamento | RAG: busca híbrida no Project Gutenberg + LLM local gera resposta com citações `[n]` |
| **5** | Dados de Treinamento | 27.740 livros, 10.000.018.524 bytes extraídos do Project Gutenberg; índice amostra de 20.000 passagens ou completo em FTS5 |
| **6** | Pré-processamento | Amostragem $O(L + m \log m + B_{\text{lidos}})$, embedding linear, HNSW $O(n \log n)$ |
| **7** | Busca Semântica | HNSW: 13,5 ms total em `openvino/GPU` (0,8 ms busca pura). Em 1M de vetores: 1,85 ms vs 86,45 ms força bruta. |
| **8** | Treinamento LLM | SEM treinamento local. SmolLM2-1.7B: 38,6 s/resp em `torch/CPU (fp32)` vs **5,36 s/resp** em `openvino/GPU (int4)`. |
| **9** | Formação Resposta | Recupera 3 trechos deduplicados (~140 ms) → Monta prompt/few-shot → LLM gera tokens → Valida citações (regex + semântica). Na interface web, envia eventos NDJSON, uma pergunta por vez. |
| **10** | Recursos Disponíveis | Intel i5-1135G7, 40 GB RAM, Intel Iris Xe, SSD NVMe; ~38 GiB em disco no modo completo (compatível com i5-13420H + Intel UHD). |
| **11** | Tempo Fases | Coleta 10 GB (~48 min), FTS5 11,7M passagens (~36 min), Embedding 20k: 78 s (Iris Xe) / 389 s (CPU). |
| **12** | Tempo Training × Qualidade | Sem treinamento. max_tokens=200 é ótimo. Corpus completo FTS5 supera gaps da amostragem. |
| **13** | Tempo Inferência × Qualidade | `openvino/GPU (int4)` reduz geração média para 5,36 s (7,2× mais rápido que `torch/CPU`). |
| **14** | Avaliação RAGAS | RAGAS completo em 12 perguntas (`--indice-completo`, `Qwen2.5-1.5B` `openvino/GPU int8`, `0` NaN): `context_recall=0,917`, `context_precision=0,681`, `faithfulness=0,250`, `answer_relevancy=0,285`, `semantic_similarity=0,330`. |

### Próximos Passos Recomendados

**1. Cobertura do corpus indexado (✅ Concluído)**
- Além da amostragem padrão (`--n 20000`), o sistema conta com o índice lexical completo FTS5 (`--indice-completo`, 27.740 livros / 11,7M passagens) e deduplicação lexical entre edições.

**2. Ajuste de amostragem e deduplicação (✅ Concluído)**
- Modo amostra usa 1 passagem por livro para ampliar títulos; modo `--indice-completo` filtra passagens sobrepostas entre edições via `_shingles_texto` / `_trecho_duplicado`.

**3. Implementar query expansion (⏳ Pendente)**
- Gerar variações da pergunta e combinar resultados na recuperação híbrida.

**4. Pendências de validação e engenharia (⏳ Pendente — ver [`busca/TAREFAS_FUTURAS.md`](busca/TAREFAS_FUTURAS.md))**
- Validar retomada HTTP real de shards `.part`, gerar *lockfile* de dependências, otimizar `metodo_tfidf` em `busca/benchmark.py` e suportar download direto de modelos OpenVINO IR pré-convertidos do Hub (`OpenVINO/*-ov`). *(Avaliação completa com juiz RAGAS já concluída em `busca/avaliacao_resultados_ragas.json`.)*

---

## Arquivos Relacionados

- **rag.py** - Implementação do sistema RAG (recuperação + geração)
- **corpus.py** - Amostragem e carregamento de passagens
- **consultar.py** - Interface CLI interativa
- **frontend/servidor.py** - Servidor web (FastAPI) da interface de chat
- **frontend/static/index.html** - Página do chat (HTML, CSS e JavaScript em um único arquivo)
- **frontend/test_servidor.py** - Testes automáticos da interface web
- **infraestrutura/aceleracao.py** - Portabilidade OpenVINO/ONNX/PyTorch e exportação quantizada via `uv run`
- **avaliar.py** - Avaliação com RAGAS
- **benchmark.py** - Comparação de métodos de busca
- **dados/coletar_gutenberg.py** - Coleta e extração do corpus de livros
- **dados/README.md** - Instruções da coleta e atribuição/licença

---

## Recursos Externos

- [Sentence Transformers](https://www.sbert.net/) - Documentação de embeddings
- [Hugging Face Hub](https://huggingface.co/) - Modelos pré-treinados
- [USearch](https://github.com/unum-cloud/usearch) - Indexação HNSW
- [RAGAS](https://docs.ragas.io/) - Framework de avaliação
- [Dataset Gutenberg em inglês](https://huggingface.co/datasets/sedthh/gutenberg_english) - Corpus de dados
- [Project Gutenberg](https://www.gutenberg.org/) - Catálogo e fontes dos livros

---

## Histórico de Revisões

| Versão | Data | Mudanças |
|---|---|---|
| 1.0 | 2026-11-11 | Documentação completa das 13 questões + resumo executivo |
| 2.0 | 2026-11-27 | Portabilidade via `.env` (OpenVINO/ONNX/PyTorch), auditoria de complexidade PAA, índice lexical FTS5, quantização INT4 e benchmarks reais no Project Gutenberg |
| 2.1 | 2026-11-27 | Validação semântica de citações (`_validar_suporte_semantico`), `LLM_MODEL`/`--llm`, deduplicação lexical entre edições, `EXEMPLOS_GERACAO`, flags `--reconstruir`/`--limpar-parquets`, integração da documentação do frontend (PR #7) e registro de RAM/armazenamento |
| 2.2 | 2026-11-28 | Avaliação RAGAS completa (12 perguntas sobre `--indice-completo` com juiz `Qwen/Qwen2.5-1.5B-Instruct` em `openvino/GPU int8`, `0` NaN) e atualização das seções de avaliação |

---

**Autor**: Sistema de Documentação Automática  
**Última atualização**: 2026-11-28 02:35:00 UTC-03:00  
**Status**: ✅ Completo e Verificado
