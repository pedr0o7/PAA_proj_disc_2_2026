# Roteiro da Apresentação e Demonstração Prática (10 minutos)

**Disciplina:** Projeto e Análise de Algoritmos (CIC/UnB — 117536, 2/2026)  
**Tema:** Sistema RAG com LLM Local sobre o Project Gutenberg (10 GB)  
**Data:** 30/11/2026  

---

## 1. Cronograma Sugerido (10 minutos | 5 integrantes)

| Tempo | Bloco | Slides | Tópicos do Enunciado | Responsável Sugerido |
|---:|---|---|---|---|
| **0:00 – 1:45** | **Abertura, Stack e Funcionamento** | Slides 1–4 | Integrantes, linguagens/licenças, programas (FastAPI + CLI + FTS5 + `.env`), tema e roteamento por categoria | Integrante 1 |
| **1:45 – 3:30** | **Dados e Pré-processamento (PAA)** | Slides 5–6 | Coleta Gutenberg (10 GB, 27.740 livros, SHA-256, `hf_xet`), segmentação FTS5 e análise de complexidade (`argpartition` $O(N + k \log k)$) | Integrante 2 |
| **3:30 – 5:30** | **Busca Semântica, LLM e Servidor** | Slides 7–9 | Benchmark N=20k + escala 1M (`46,7×`), quantização OpenVINO IR (`int8`/`int4`), NDJSON, deduplicação 4-gramas e validação semântica | Integrante 3 |
| **5:30 – 7:45** | **Hardware, Tempos e RAGAS** | Slides 10–12 | Recursos (`i5-1135G7`, 40 GB RAM, Iris Xe, NVMe, ~38 GiB; compatível `i5-13420H`), speedup `5,0×`/`7,2×`, avaliação RAGAS (`0` NaN, `0,917` recall) | Integrante 4 |
| **7:45 – 9:45** | **Demonstração Prática ao Vivo** | Slide 13 + Browser | Execução das 4 consultas na interface web (`http://localhost:8000`) mostrando fontes e citações `[n]` clicáveis | Integrante 5 (Mauro) |
| **9:45 – 10:00** | **Encerramento** | Slide 13 | Síntese, 118 testes unitários e links de entrega | Todos |

---

## 2. Preparação Pré-Apresentação (Checklist)

1. Abrir terminal PowerShell na pasta `busca/` com a `.venv` ativada:
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

## 3. Roteiro da Demonstração Prática ao Vivo (~2 minutos)

Executar as 4 perguntas em sequência no chat (`http://localhost:8000`), destacando os diferentes caminhos do sistema:

1. **Categoria 1 — Autoria da obra (Metadados, `~0 s`, sem LLM):**
   - **Pergunta:** `Who wrote Pride and Prejudice?`
   - **O que mostrar:** resposta instantânea com `[1]` apontando para os metadados do catálogo do Project Gutenberg (sem acionar o LLM).

2. **Categoria 2 — Localização de frase literal (Índice FTS5, `~0 s`, sem LLM):**
   - **Pergunta:** `Where does the phrase tranquillise the mind as a steady purpose come from?`
   - **O que mostrar:** identificação exata em *Frankenstein* (`ebook/84`) com link clicável e ressalva de correspondência textual.

3. **Categoria 3 — Pergunta aberta de personagem (Pipeline RAG completo + Deduplicação + Citação `[n]`):**
   - **Pergunta:** `Who is Captain Ahab in Moby Dick?`
   - **O que mostrar:** busca híbrida (~1,4 s) + geração local na Intel Iris Xe (~5–9 s), retornando 3 trechos **deduplicados** entre edições de *Moby-Dick* (`#2489`, `#2701`) e citações clicáveis `[1]`, `[2]`, `[3]` validadas semanticamente.

4. **Guarda Anti-Alucinação (Pergunta fora do corpus):**
   - **Pergunta:** `What is the current population of Tokyo in 2026?`
   - **O que mostrar:** o sistema se recusa a responder de memória e devolve exatamente `The provided context is insufficient to answer this question.` (`nenhuma fonte citada`).

---

## 4. Arquivos e Links de Referência

- **Apresentação editável (PPTX):** `apresentacao/PAA_RAG_Gutenberg.pptx`
- **Apresentação exportada (PDF):** `apresentacao/PAA_RAG_Gutenberg.pdf`
- **Banco de dados completo (Google Drive):** https://drive.google.com/drive/folders/1Wm744_mm06RF32B8KLE8GTGZzVtvFyMr
