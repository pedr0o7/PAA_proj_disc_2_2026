# Roteiro da Apresentação, Falas Prontas e Demonstração Prática (10 minutos)

**Disciplina:** Projeto e Análise de Algoritmos (CIC/UnB — 117536, 2/2026)
**Tema:** Sistema RAG com LLM Local sobre o Project Gutenberg (10 GB)
**Data:** 30/11/2026

---

## 1. Cronograma Sugerido (10 minutos | 5 integrantes)

|                   Tempo | Bloco                                      | Slides             | Tópicos do Enunciado                                                                                                                                                                          | Responsável Sugerido |
| ----------------------: | ------------------------------------------ | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
|  **0:00 – 1:45** | **Abertura, Stack e Funcionamento**  | Slides 1–4        | Integrantes, linguagens/licenças, programas (FastAPI + CLI + FTS5 + `.env`), tema e roteamento por categoria                                                                                | Integrante 1          |
|  **1:45 – 3:30** | **Dados e Pré-processamento (PAA)** | Slides 5–6        | Coleta Gutenberg (10 GB, 27.740 livros, SHA-256, `hf_xet`), segmentação FTS5 e análise de complexidade (`argpartition` $O(N + k \log k)$)                                             | Integrante 2          |
|  **3:30 – 5:30** | **Busca Semântica, LLM e Servidor** | Slides 7–9        | Benchmark N=20k + escala 1M (`46,7×`), quantização OpenVINO IR (`int8`/`int4`), NDJSON, deduplicação 4-gramas e validação semântica                                              | Integrante 3          |
|  **5:30 – 7:45** | **Hardware, Tempos e RAGAS**         | Slides 10–12      | Recursos (`i5-1135G7`, 40 GB RAM, Iris Xe, NVMe, ~38 GiB; compatível `i5-13420H`), speedup `5,2×`–`7,6×`, reteste `SmolLM2` vs `Qwen2.5`, RAGAS (`0` NaN, `Hit@3 = 0,500`) | Integrante 4          |
|  **7:45 – 9:45** | **Demonstração Prática ao Vivo**  | Slide 13 + Browser | Execução das 4 consultas na interface web (`http://localhost:8000`) mostrando fontes e citações `[n]` clicáveis                                                                       | Integrante 5          |
| **9:45 – 10:00** | **Encerramento**                     | Slide 13           | Síntese, 118 testes unitários e entrega                                                                                                                                                      | Todos                 |

---

## 2. Script Falado — Slide a Slide (7 min 45 s)

> **Dica de ritmo:** ~40–45 s por bloco. As *Speaker Notes* do Google Slides trazem a versão resumida em bullets; abaixo está o texto completo para ensaio.

### Bloco 1 — Abertura, Stack e Funcionamento (`0:00 – 1:45`, Integrante 1)

- **Slide 1 (Capa e Integrantes — ~15 s):**
  > *"Bom dia a todos. Nosso grupo desenvolveu um sistema RAG com LLM local para responder perguntas sobre literatura em domínio público do Project Gutenberg, operando sobre 10 GB de texto puro e sempre citando a obra e a passagem de origem."*
  >
- **Slide 2 (Linguagens, Bibliotecas e Licenças — ~30 s):**
  > *"Toda a base foi construída em Python 3.12, com interface web em HTML, CSS e JavaScript puros — sem etapa de build — e persistência lexical em SQLite FTS5. Nas bibliotecas, usamos `torch`, `transformers`, `sentence-transformers` com `all-MiniLM-L6-v2`, `usearch` para HNSW, `FastAPI`/`Uvicorn`, `ragas` para avaliação e, de forma opcional, `openvino` e `onnxruntime` para acelerar em CPU e GPU integrada, todas sob licenças permissivas como MIT, BSD e Apache 2.0."*
  >
- **Slide 3 (Programas e Infraestrutura — ~30 s):**
  > *"A arquitetura tem três frentes: o servidor FastAPI com streaming NDJSON e fila segura por `Lock` — compartilhando o mesmo motor `SistemaRAG` com a CLI; uma camada de portabilidade via `.env` que detecta hardware Intel ou AMD automaticamente com fallback seguro para `torch/CPU`; e os arquivos em disco com gravação atômica, corpus UTF-8 e índice FTS5."*
  >
- **Slide 4 (Tema e Funcionamento — ~30 s):**
  > *"Quando o usuário faz uma pergunta em inglês, o sistema aplica um roteamento inteligente no índice completo: perguntas de autoria vão direto ao catálogo em ~0 s sem acionar o LLM; localização de frases literais usa busca exata/aproximada no FTS5 também em ~0 s; e apenas perguntas abertas sobre personagens ou enredo percorrem o pipeline RAG completo, com validação sintática e semântica das citações."*
  >

### Bloco 2 — Dados e Pré-processamento (`1:45 – 3:30`, Integrante 2)

- **Slide 5 (Dados: Origem, Download e Tamanho — ~45 s):**
  > *"Os dados vêm do dataset `sedthh/gutenberg_english` no Hugging Face. Coletamos de forma incremental em arquivos Parquet com validação de tamanho e SHA-256, retomada segura via `.part` e deduplicação por ID. O corpus final atingiu exatos 10.000.018.524 bytes (10,00 GB) e 27.740 livros únicos em `textos.txt`, indexados por offset e tamanho em `livros.jsonl` para leitura direta em $O(1)$."*
  >
- **Slide 6 (Pré-processamento e Complexidade — ~60 s):**
  > *"Aqui resumimos as complexidades de tempo e espaço de cada etapa: a coleta é linear nos bytes $O(B)$; a amostragem com leitura ordenada por offset custa $O(L + m \log m + B_{\text{lidos}})$; o índice FTS5 segmenta os livros em 11,7 milhões de janelas de 1.000 caracteres com 150 de sobreposição; e o grafo HNSW indexa em $O(n \log n)$. Um destaque de PAA foi a auditoria no `rag.py`: trocamos a ordenação completa do BM25 de $O(N \log N)$ para $O(N + k \log k)$ usando `np.argpartition`."*
  >

### Bloco 3 — Busca Semântica, LLM e Servidor (`3:30 – 5:30`, Integrante 3)

- **Slide 7 (Busca Semântica e Escalabilidade — ~45 s):**
  > *"No benchmark real com 20.000 passagens, o TF-IDF obteve 0,997 de recall@10 porque a consulta sintética é extraída literalmente da passagem. Já os Sentence Embeddings (`all-MiniLM-L6-v2`) capturam paráfrases e, ao migrar o cálculo de `torch/CPU` para `openvino/GPU`, o tempo de codificação caiu 5,0× (de 388,7 s para 77,95 s). No teste de escala até 1 milhão de vetores (`escala.py`), o HNSW caiu de 86,45 ms da força bruta para 1,85 ms — um ganho de 46,7×."*
  >
- **Slide 8 (LLM Local, Complexidade e Quantização — ~35 s):**
  > *"Não fizemos fine-tuning local: usamos modelos instruct pré-treinados (`SmolLM2-1.7B` por padrão e `Qwen2.5-1.5B` opcional) com decodificação determinística (`do_sample=False`). A quantização para OpenVINO IR `int8` e `int4` reduz o modelo em até 4× e acelera a inferência em até 7,2×, convertendo uma única vez via `uv run` isolado — ou usando a pasta IR pré-exportada sem `uv`."*
  >
- **Slide 9 (Formação de Resposta do Servidor e Citações — ~40 s):**
  > *"No servidor FastAPI, cada requisição entra numa fila com `threading.Lock` e devolve status, fontes e resposta em streaming NDJSON com menos de 1 ms de overhead. Para garantir citações confiáveis, aplicamos três defesas: deduplicação de 4-gramas entre edições distintas do Gutenberg, exemplos few-shot multipassagem/abstenção e validação semântica por frase citada com limiar de cosseno $\ge 0,30$."*
  >

### Bloco 4 — Hardware, Tempos e RAGAS (`5:30 – 7:45`, Integrante 4)

- **Slide 10 (Recursos Computacionais e Tempo das Fases — ~40 s):**
  > *"Os experimentos rodaram em um notebook Intel Core i5-1135G7 com GPU integrada Iris Xe, 40 GB de RAM e SSD NVMe de 1 TB — consumindo ~38 GiB em disco entre corpus, índice FTS5 e caches — e a mesma esteira é compatível com outros chips Intel como o i5-13420H. A coleta levou ~48 min, o FTS5 ~36 min, os embeddings 78 s na iGPU e uma consulta completa na web leva ~5 a 10 s."*
  >
- **Slide 11 (Tempo de Inferência × Qualidade — ~45 s):**
  > *"Na geração token-a-token, o gargalo é a largura de banda de memória, por isso a quantização OpenVINO na GPU reduziu a média de 38,63 s em CPU `fp32` para 5,36 s em `int4`. Em reteste atualizado sobre o código final com `int8` e 4 perguntas no índice completo, o `SmolLM2-1.7B` acertou 2/4 em 10,13 s médios (acertou factual com `[1]` e recusou fora do corpus, mas absteve em síntese de personagem), enquanto o `Qwen2.5-1.5B` obteve 4/4 em 7,58 s médios."*
  >
- **Slide 12 (Avaliação pelo Framework RAGAS — ~50 s):**
  > *"Por fim, rodamos a avaliação completa com RAGAS v0.3.9 sobre 12 perguntas no `--indice-completo` usando `Qwen2.5-1.5B` (`int8`) como gerador e juiz local, sem nenhum `NaN` nas 60 avaliações. Uma leitura crítica importante: nosso `Hit@3` real foi 0,500 (6/12 livros exatos), e o `context_recall` de 0,917 também reflete certo otimismo do juiz local de 1,5B. Já `faithfulness` (0,250) e `answer_relevancy` (0,285) caem porque o sistema se absteve em 58,3% das perguntas — 4 onde o livro não veio no top-3 e 3 por bloqueio conservador do validador — o que pontua 0 no RAGAS, mas evita alucinações."*
  >

---

## 3. Preparação Pré-Apresentação (Checklist Técnico)

1. Abrir terminal PowerShell na raiz do projeto com a `.venv` ativada e subir o servidor:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   cd busca
   python ..\frontend\servidor.py --indice-completo
   ```
2. Aguardar `Uvicorn running on http://127.0.0.1:8000` e abrir **`http://localhost:8000`** no navegador.
3. Conferir no topo da página o indicador:

   - `Índice completo (27.740 livros)`
   - Badge de execução: `EMB=openvino/GPU | LLM=openvino/GPU(int8)` (com `Qwen/Qwen2.5-1.5B-Instruct` configurado no `.env`).
4. **Plano B (sem internet na sala):** definir `HF_HUB_OFFLINE=1` no `.env` ou terminal antes de subir o servidor; todos os modelos e índices já estão em cache local.

---

## 4. Demonstração Prática ao Vivo (`7:45 – 9:45`, Integrante 5 — Mauro)

> **Fala de abertura da demo:** *"Agora vamos demonstrar o sistema ao vivo em `http://localhost:8000` conectado ao índice completo de 27.740 livros, percorrendo os quatro cenários do roteamento."*

1. **Categoria 1 — Autoria da obra (Metadados, `~0 s`, sem LLM):**

   - **Digitar:** `Who wrote Pride and Prejudice?`
   - **O que apontar na tela:** resposta instantânea com `[1]` apontando para os metadados do catálogo do Project Gutenberg (sem acionar o LLM).
   - **Fala:** *"Para perguntas de autoria, o roteador consulta direto o catálogo de metadados e responde em milissegundos, sem gastar inferência do LLM."*
2. **Categoria 2 — Localização de frase literal (Índice FTS5, `~0 s`, sem LLM):**

   - **Digitar:** `Where does the phrase tranquillise the mind as a steady purpose come from?`
   - **O que apontar na tela:** identificação exata em *Frankenstein* (`ebook/84`) com link clicável e ressalva de correspondência textual.
   - **Fala:** *"Quando buscamos uma citação literal, o índice FTS5 localiza a obra exata — aqui Frankenstein — também sem acionar o modelo generativo."*
3. **Categoria 3 — Pergunta aberta de personagem (Pipeline RAG completo + Deduplicação + Citação `[n]`):**

   - **Digitar:** `Who is Captain Ahab in Moby Dick?`
   - **O que apontar na tela:** busca híbrida (~1,4 s) + geração local na Intel Iris Xe (~5–10 s), retornando 3 trechos **deduplicados** entre edições de *Moby-Dick* (`#2489`, `#2701`) e citações clicáveis `[1]`, `[2]`, `[3]` validadas semanticamente.
   - **Fala:** *"Já em uma pergunta aberta sobre personagem, entra o pipeline completo: o FTS5 + reranking recupera 3 cenas distintas — evitando edições duplicadas por shingles de 4 palavras — o Qwen2.5 sintetiza em GPU integrada e o validador confirma cada citação `[1]`, `[2]`, `[3]`."*
4. **Guarda Anti-Alucinação (Pergunta fora do corpus):**

   - **Digitar:** `What is the current population of Tokyo in 2026?`
   - **O que apontar na tela:** recusa determinística `The provided context is insufficient to answer this question.` (`nenhuma fonte citada`).
   - **Fala:** *"Por fim, se perguntarmos algo fora do corpus — como a população de Tóquio em 2026 — mesmo havendo livros sobre o Japão, o sistema não responde de memória e devolve a recusa padrão."*

- **Encerramento (`9:45 – 10:00`):**
  > *"Em resumo, entregamos o pipeline completo de 10 GB validado por 118 testes unitários, com aceleração portátil para hardware comum e fallback seguro. Obrigado, e ficamos à disposição para perguntas."*
  >

---

## 5. Fluxo de Revisão e Arquivos de Referência

- **Edição da apresentação (fonte única):** feita pelo grupo exclusivamente no **Google Slides** dentro da pasta privada `PAA_2026_Revisao_Grupo` no Google Drive. Na versão final congelada (até 30/11), serão exportados o arquivo editável original e o PDF para compor o `.zip` submetido no Aprender3.
- **Perguntas prováveis de arguição:** [apresentacao/PERGUNTAS_ARGUICAO.md](PERGUNTAS_ARGUICAO.md)
- **Banco de dados completo** [(Drive público):](`https://drive.google.com/drive/folders/1Wm744_mm06RF32B8KLE8GTGZzVtvFyMr`) e detalhes em [apresentacao/LINK_BANCO_COMPLETO.md](LINK_BANCO_COMPLETO.md)
