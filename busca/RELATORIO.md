# Busca semântica: comparação de métodos

> Resultados medidos sobre o corpus atual do **Project Gutenberg** (`dados\gutenberg_10gb\textos.txt`: 27.740 livros, 10.000.018.524 bytes) em **Intel Core i5-1135G7 (40 GB RAM, SSD NVMe) + Intel Iris Xe Graphics** (`torch/CPU` vs. `openvino/GPU`).

Scripts: [benchmark.py](./benchmark.py) (qualidade e latência em passagens reais),
[escala.py](./escala.py) (busca vetorial pura em N até 1 milhão) e
[avaliar.py](./avaliar.py) (avaliação ponta a ponta do RAG). Resultados brutos em
`resultados.json`, `resultados_openvino.json`, `resultados_torch.json`, `escala.json`,
`avaliacao_resultados.json`, `avaliacao_resultados_openvino.json` e `avaliacao_resultados_torch.json`.

Reprodução (PowerShell, na raiz):

```powershell
.\.venv\Scripts\python.exe busca\benchmark.py
.\.venv\Scripts\python.exe busca\escala.py
.\.venv\Scripts\python.exe busca\avaliar.py --sem-ragas --min-sim 0.4
```

## Metodologia

- Corpus: passagens de `dados\gutenberg_10gb\textos.txt`; N = 2.000, 8.000 e 20.000.
- Consultas (`benchmark.py`): 300 frases (>= 8 palavras) extraídas da passagem-alvo; a passagem de origem é a única relevante. Métricas: recall@1, recall@10, MRR@10 (k = 10).
- Métodos (todos com similaridade de cosseno, vetores normalizados):
  1. **TF-IDF + cosseno** (linha de base lexical, esparso).
  2. **Word2Vec médio** (gensim, 100 dim, treinado na própria amostra, média ponderada por IDF).
  3. **Sentence embeddings** (`all-MiniLM-L6-v2`, 384 dim) com cosseno exato (força bruta), medido em `torch/CPU` e `openvino/GPU`.
  4. **Sentence embeddings + HNSW** (usearch, M = 16, efConstruction = 128, efSearch = 64), medido em `torch/CPU` e `openvino/GPU`.
- Hardware: Intel Core i5-1135G7 (4C/8T, 40 GB RAM, SSD NVMe) + Intel Iris Xe Graphics, Windows 11.

## Resultados de Recuperação (N = 20.000 — Project Gutenberg)

| Método | recall@1 | recall@10 | MRR@10 | Construção | Latência/consulta | Índice |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + cosseno | 0,987 | 0,997 | 0,989 | 1,22 s | 3,03 ms | 6,1 MB |
| Word2Vec médio | 0,267 | 0,463 | 0,323 | 9,10 s | 0,60 ms | 7,6 MB |
| Sentence + cosseno exato (`torch/CPU`) | 0,563 | 0,667 | 0,594 | 388,69 s | 17,94 ms (busca: 1,53 ms) | 29,3 MB |
| Sentence + cosseno exato (`openvino/GPU`) | 0,563 | 0,663 | 0,593 | 77,95 s | 19,63 ms (busca: 1,29 ms) | 29,3 MB |
| Sentence + HNSW (`torch/CPU`) | 0,527 | 0,620 | 0,554 | 395,53 s | 16,24 ms (busca: 0,78 ms) | 61,5 MB |
| Sentence + HNSW (`openvino/GPU`) | 0,530 | 0,620 | 0,557 | 87,73 s | 13,47 ms (busca: 0,80 ms) | 61,5 MB |

**Comparativo de embeddings (`torch/CPU` vs. `openvino/GPU` em N = 20.000):**

- Codificação de 20.000 passagens caiu de **388,69 s** (~6,5 min) para **77,95 s** (**5,0× mais rápido** na Iris Xe), preservando praticamente os mesmos `recall@10` e `MRR@10` (paridade cosseno $> 0{,}99999$).
- Na busca HNSW, a latência total caiu de **16,24 ms** para **13,47 ms**; o `recall_vs_exato@10` do HNSW ficou em **0,948** (`torch`) / **0,947** (`openvino`).

## Escalabilidade da busca vetorial (`escala.py` — 384 dim, vetores sintéticos agrupados)

| N | Força bruta | HNSW | Recall@10 HNSW | Construção HNSW |
|---|---:|---:|---:|---:|
| 10 mil | 0,76 ms | 0,50 ms | 0,995 | 2,3 s |
| 100 mil | 9,01 ms | 0,62 ms | 0,993 | 25,4 s |
| 500 mil | 38,94 ms | 1,19 ms | 0,871 | 216,0 s |
| 1 milhão | 86,45 ms | 1,85 ms | 0,770 | 617,2 s |

A força bruta cresce linearmente; o HNSW, aproximadamente logarítmico (**46,7× mais rápido** em 1 milhão de vetores). O recall do HNSW cai em N grande com parâmetros fixos; seria preciso aumentar `efSearch`/`M` (mais latência e memória).

## Avaliação RAG Ponta a Ponta (`avaliar.py`: `torch/CPU fp32` vs. `openvino/GPU int4`)

Setup: 20 perguntas (`avaliacao_dataset.json`), `k = 3`, `min_sim = 0.4`, amostra de 15.251 passagens (1 por livro), modelo `HuggingFaceTB/SmolLM2-1.7B-Instruct`.

| Configuração | Recuperação média | Geração média | Geração mediana | Geração P95 | Hit@3 | Abstenções |
|---|---:|---:|---:|---:|---:|---:|
| `torch/CPU (fp32)` | 0,161 s | 38,63 s | 38,62 s | 83,40 s | 0,15 | 75% |
| `openvino/GPU (int4)` | 0,141 s | **5,36 s** | **5,59 s** | **10,97 s** | 0,15 | 100% |

- **Speedup de geração:** **`7,2×`** na média (`38,63 s → 5,36 s`) e **`7,6×`** no P95 (`83,40 s → 10,97 s`).
- **Nota sobre citações e modelos (`SmolLM2-1.7B` vs. `Qwen2.5-1.5B`):** em teste exploratório anterior às melhorias de prompt e validação (`max_tokens=80`, 4 perguntas), `SmolLM2-1.7B-Instruct` em `openvino/GPU (int8)` atingiu `3,74 s` médios (`5,2×` sobre `fp32` em CPU, `19,32 s`), mas omitiu `[1]` nas respostas válidas e alucinou `[1]` sem contexto. Em **reteste atualizado sobre o código final** (`--indice-completo`, `min_sim=0.4`, `max_tokens=200`, `openvino/GPU int8`, 4 perguntas + teste direto com passagem distratora), `SmolLM2-1.7B-Instruct` obteve **2/4** (`10,13 s` médios — acertou a pergunta factual com `[1]` e recusou corretamente fora do corpus, mas absteve em 2 perguntas abertas de personagem), enquanto `Qwen/Qwen2.5-1.5B-Instruct` atingiu **4/4** (`7,58 s` médios, com `[1]`–`[3]` e recusa correta). Além disso, `rag.py` aplica validação semântica das citações (`_validar_suporte_semantico`, limiar cosseno `0,30`) para bloquear citações alucinadas.
- **Nota sobre Hit@3 (0,15):** `avaliacao_dataset.json` foi gerado sobre uma amostra anterior; na amostra atual de 15.251 passagens, apenas 15% das fontes esperadas estavam presentes no índice amostral. Para cobertura total dos 27.740 livros, recomenda-se usar `--indice-completo` (`gutenberg_lexical_v1.sqlite`, 11.741.620 passagens).

### Avaliação RAGAS Completa (`--indice-completo`, `openvino/GPU int8`)

Execução oficial com juiz local (`avaliar.py --q 12 --indice-completo --min-sim 0.4 --saida busca/avaliacao_resultados_ragas.json`), utilizando `Qwen/Qwen2.5-1.5B-Instruct` em `openvino/GPU (int8)` tanto na geração quanto como juiz RAGAS (via adaptador `ModeloLLMOpenVINOGenAI` para LangChain):

| Métrica | Valor | Observação |
|---|---:|---|
| `hit_at_k` (`k = 3`) / `mrr` | **0,500** / **0,417** | 12 perguntas presentes nos 27.740 livros |
| `respostas_com_citacao` / `citacao_da_fonte_correta` | **0,417** / **0,250** | 58,3% de abstenções conservadoras (`SEM_CONTEXTO`) |
| Tempo médio: recuperação / geração / total | 10,47 s / 6,77 s / **17,33 s** | Inclui reranking semântico e validação |
| `context_recall` (RAGAS, `0` NaN) | **0,917** | Alta cobertura pelo juiz local 1,5B (`Hit@3` real = `0,500`) |
| `llm_context_precision_with_reference` (RAGAS, `0` NaN) | **0,681** | Precisão dos contextos frente à referência |
| `faithfulness` / `answer_relevancy` / `semantic_similarity` (`0` NaN) | 0,250 / 0,285 / 0,330 | Médias reduzidas pelo alto índice de abstenções anti-alucinação |

### Comportamento por categoria de pergunta e lições aprendidas (`--indice-completo`)

No modo `--indice-completo`, [`SistemaRAG.perguntar()`](./rag.py) roteia as perguntas do usuário por três caminhos distintos:

| Categoria de pergunta | Caminho no código | Usa o LLM? | Comportamento |
| :--- | :--- | :---: | :--- |
| **Autoria da obra** (`Who wrote ...?`) | Consulta direta aos metadados do catálogo | Não | Resposta e `[1]` montados deterministicamente (`tempo_geracao ≈ 0 s`) quando a atribuição é inequívoca. |
| **Localização de frase/trecho** (`Where does the phrase ... come from?` / `Which book contains the quote ...?`) | Busca literal/aproximada no índice FTS5 | Não | Localiza a frase exata ou aproximada e retorna a fonte `[1]` sem acionar o LLM. |
| **Pergunta aberta sobre personagem/enredo** (`Who is Captain Ahab in Moby Dick?`) | Pipeline RAG completo | **Sim** | Único fluxo que depende de busca híbrida, síntese pelo LLM e validação sintática/semântica de citações `[n]`. |

**Lições aprendidas em consultas literárias abertas:**

- **Síntese sem verbete enciclopédico:** romances não trazem definições biográficas diretas de personagens; a resposta precisa ser sintetizada a partir de múltiplas cenas recuperadas.
- **Duplicação entre edições do Project Gutenberg:** sem deduplicação, janelas sobrepostas da mesma cena em edições distintas (ex.: *Moby-Dick* `#2489`, `#15`, `#2701`) ocupavam todos os 3 slots de `k = 3`. A inclusão do filtro por sobreposição de 4-gramas (`_shingles_texto` / `_trecho_duplicado`, limiar `0,25`) em `SistemaRAG.recuperar()` garantiu 3 cenas distintas (`#2489`, `#2489`, `#2701`).
- **Omissão de `[n]` em contextos longos:** modelos compactos (1,5B–1,7B), diante de ~3.000 caracteres de contexto e perguntas famosas do tipo `"Who is X in Y?"`, tendiam a responder em "modo enciclopédia" omitindo `[n]`, levando `validar_citacoes()` a descartar a resposta como `SEM_CONTEXTO`. Alinhar `EXEMPLOS_GERACAO` (*few-shot* multipassagem + exemplo de abstenção no mesmo cabeçalho `(Title by Author)`) e adicionar o lembrete final de citação em `SistemaRAG.gerar()` estabilizou a saída (`citados = [1, 2, 3]` consistentes em `"Who is Captain Ahab in Moby Dick?"` e abstenção segura em `"What is the current population of Tokyo in 2026?"`).

## Complexidade

| Método | Construção | Consulta | Memória |
|---|---|---|---|
| TF-IDF | O(tokens) | O(nnz_visitado + N + k log k) (produto esparso com densificação da consulta) | O(tokens únicos por doc) |
| Word2Vec médio | treino O(épocas·tokens·d) + O(N·d) | O(N·d) | O(N·d) |
| Sentence + cosseno exato | N inferências do modelo | O(N·d) + 1 inferência | O(N·d) |
| Sentence + HNSW | N inferências + O(N·log N) | ~O(log N) + 1 inferência | O(N·(d + M)) |

## Conclusões

- **Qualidade:** em passagens literárias do Gutenberg, o TF-IDF venceu no teste sintético porque a consulta é uma frase literal da própria passagem (forte sobreposição lexical). Sentence embeddings (`all-MiniLM-L6-v2`) permitem busca semântica quando há paráfrase, mas com 1 passagem por livro em amostra de 20k há perda de cobertura — superada pelo índice lexical completo FTS5 + reranking semântico (`--indice-completo`).
- **Desempenho:** até ~100 mil passagens a força bruta basta e é exata. Acima disso, HNSW reduz a latência em dezenas de vezes (86,45 ms → 1,85 ms em 1 M). Na inferência local em notebook Intel (i5-1135G7 + Iris Xe), o runtime nativo **OpenVINO + OpenVINO GenAI (`int4`)** reduz em **5,0×** o tempo de embedding e em **7,2×** o tempo de geração do LLM.
