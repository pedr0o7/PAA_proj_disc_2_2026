# Perguntas Prováveis de Arguição e Respostas Técnicas

Guia rápido para o grupo responder com precisão a perguntas do professor de **PAA** ou de colegas da turma após a apresentação.

---

## Bloco A — Complexidade e Algoritmos (PAA)

### 1. Por que a troca para `np.argpartition` em `busca/rag.py` reduziu a complexidade?

- **Resposta:** Para selecionar apenas os `3k` melhores documentos do BM25 entre `N` passagens, `np.argsort` ordenava o vetor inteiro em $O(N \log N)$. Com `np.argpartition`, particionamos o vetor em tempo linear $O(N)$ para isolar os `3k` maiores e ordenamos só esse subconjunto em $O(k \log k)$, resultando em **$O(N + k \log k)$**.

### 2. Por que usar SQLite FTS5 + reranking semântico em 50 candidatos em vez de indexar todas as 11,7 milhões de passagens no HNSW?

- **Resposta:** Indexar 11,7M vetores `float32` de 384 dimensões exigiria `~18 GB` só de matriz de embeddings (mais o grafo HNSW em RAM) e muitas horas de GPU/CPU na preparação. O FTS5 em disco ocupa `23,3 GiB`, pré-filtra os 50 melhores candidatos lexicais em segundos com baixo uso de RAM e permite aplicar o modelo neural (`all-MiniLM-L6-v2`) apenas sobre esses 50 trechos.

### 3. Por que o TF-IDF venceu (`recall@10 = 0,997`) os Sentence Embeddings (`0,667`) no benchmark sintético de N = 20.000?

- **Resposta:** Porque no benchmark (`busca/benchmark.py`), cada consulta é uma frase literal de $\ge 8$ palavras extraída da própria passagem-alvo. Quando há sobreposição exata de palavras raras, modelos esparsos (TF-IDF) são praticamente imbatíveis. O ganho dos Sentence Embeddings aparece em **paráfrases e consultas semânticas abertas** (onde os termos exatos da cena podem não coincidir com a pergunta).

### 4. Por que o recall do HNSW caiu para `0,770` em 1 milhão de vetores (`escala.py`)?

- **Resposta:** Porque mantivemos os parâmetros fixos (`M = 16`, `efConstruction = 128`, `efSearch = 64`) enquanto `N` cresceu 100× (de 10 mil para 1 milhão), aumentando a densidade de vizinhos próximos e fazendo a busca gulosa parar antes de visitar todos os candidatos ótimos. Aumentar `efSearch` e `M` recupera o recall, trocando memória e latência por precisão.

---

## Bloco B — LLM, Quantização e Portabilidade

### 5. Por que não houve treinamento ou *fine-tuning* local da LLM?

- **Resposta:** Treinar ou ajustar um modelo de 1,5B–1,7B parâmetros sobre 10 GB de texto exigiria GPUs dedicadas com dezenas de GB de VRAM e dezenas de horas, fora dos recursos de notebooks comuns (Intel Iris Xe / UHD). Adotamos modelos *instruct* pré-treinados e concentramos o esforço na engenharia de recuperação (FTS5 + deduplicação) e na validação das respostas, conforme previsto pelo enunciado (*"fazer o melhor possível com os recursos disponíveis"*).

### 6. Por que a quantização `int8`/`int4` acelera 5×–7× mesmo em GPU integrada compartilhando a mesma RAM da CPU?

- **Resposta:** A geração autoregressiva token-a-token tem baixa intensidade aritmética e é limitada por **largura de banda de memória** (*memory-bandwidth bound*): a cada token gerado, é preciso ler praticamente todos os pesos do modelo. Reduzir os pesos de `fp32` (~6,8 GB) para `int8` (~1,5 GB) ou `int4` (~0,9 GB) diminui na mesma proporção o volume de bytes trafegados por token, além de aproveitar instruções vetoriais da iGPU Intel via OpenVINO.

### 7. Por que `SmolLM2-1.7B` obteve 2/4 no reteste atual enquanto `Qwen2.5-1.5B` obteve 4/4?

- **Resposta:** No reteste sobre o código final (`--indice-completo`, `int8`, `max_tokens=200`), o `SmolLM2-1.7B` seguiu bem a pergunta factual direta (`[1]`) e a recusa fora do corpus (`SEM_CONTEXTO`), mas falhou em sintetizar respostas com citações em perguntas abertas de personagem (*Captain Ahab* e *Elizabeth Bennet*), acionando a abstenção conservadora. O `Qwen2.5-1.5B-Instruct` seguiu consistentemente o formato multi-passagem `[1]`–`[3]` em todos os 4 casos e levou menos tempo (`7,58 s` vs. `10,13 s`).

### 8. Por que usar `uv run` para exportar OpenVINO IR e como rodar sem `uv` em outras máquinas?

- **Resposta:** O `optimum-intel==2.2.0` exige `transformers<5.6` e `huggingface_hub<1.22`, enquanto o projeto usa `transformers>=5.23`. Rodar a conversão PyTorch → OpenVINO IR (`openvino_model.xml` + `.bin`) em um ambiente efêmero via `uv run` evita *downgrade* no `.venv` principal. Além disso, disponibilizamos a pasta IR pré-exportada no Google Drive para Intel (GPU/CPU no modo `auto`) e CPU AMD/ARM (`LLM_BACKEND=openvino`), sem precisar de `uv` nem reprocessar a quantização.

---

## Bloco C — RAGAS, Anti-Alucinação e Reprodutibilidade

### 9. Como `context_recall` no RAGAS deu `0,917` se o `Hit@3` exato foi `0,500`?

- **Resposta:** O `Hit@3 = 0,500` exige que o `id` exato da obra esperada apareça no top-3 (em alguns casos o FTS5 traz outra edição ou obra com tema próximo). Já o `context_recall` de `0,917` (11/12) foi atribuído pelo juiz local `Qwen2.5-1.5B` — que só pontuou `0` quando `fontes = []` — mostrando que um juiz compacto de 1,5B tende a ser otimista/leniente ao avaliar cobertura de contexto frente a juízes maiores.

### 10. `faithfulness = 0,250` e `answer_relevancy = 0,285` significam que o sistema alucina 75% das vezes?

- **Resposta:** **Não.** Em `7` das `12` perguntas (`58,3%`), o sistema retornou `The provided context is insufficient to answer this question.` (`SEM_CONTEXTO`). No RAGAS, abstenções pontuam **`0`** em `faithfulness`, `answer_relevancy` e `semantic_similarity`. Em 4 desses 7 casos o livro esperado não estava no top-3 (recusa correta anti-alucinação); nos outros 3 o validador bloqueou por precaução. Ou seja, a queda reflete **conservadorismo**, não alucinação.

### 11. Como o sistema garante que a LLM não inventa citações `[1]` ou repete a mesma cena?

- **Resposta:** Primeiro, durante a recuperação no `--indice-completo`, um filtro por **shingles de 4 palavras** (`limiar 0,25`) impede que edições distintas da mesma obra do Gutenberg ocupem os 3 slots com o mesmo texto. Depois da geração, `validar_citacoes()` normaliza marcadores e remove índices inválidos, e `_validar_suporte_semantico()` verifica a similaridade de cosseno ($\ge 0,30$) entre cada frase citada e o trecho correspondente — rejeitando a resposta inteira com `SEM_CONTEXTO` se alguma citação não tiver suporte real.

### 12. Como avaliar ou demonstrar o sistema sem baixar 35 GB e sem internet na sala?

- **Resposta:** O ZIP de entrega (`< 100 MB`) traz o código completo, os 118 testes unitários e uma amostra curada de **100 livros** (incluindo os 4 livros da demo e os 20 de `avaliacao_dataset.json`) junto ao cache `torch` de 15.251 passagens e embeddings prontos. Para a demo em sala no `--indice-completo` com `HF_HUB_OFFLINE=1`, todos os índices e o IR OpenVINO já ficam em cache local, sem depender de rede.
