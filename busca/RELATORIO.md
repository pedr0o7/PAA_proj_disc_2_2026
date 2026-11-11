# Busca semântica: comparação de métodos

Scripts: [benchmark.py](./benchmark.py) (qualidade e latência em passagens reais) e
[escala.py](./escala.py) (busca vetorial pura em N até 1 milhão). Resultados brutos em
`resultados.json` e `escala.json`.

Reprodução (PowerShell, na raiz):

```powershell
$py="$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
& $py -m pip install -r busca\requirements.txt
& $py busca\benchmark.py   # ~25 min em CPU
& $py busca\escala.py      # ~20 min
```

## Metodologia

- Corpus: passagens (parágrafos de 300–1200 caracteres) sorteadas em posições aleatórias de
  `dados\wikipedia_en_10gb\textos.txt`; N = 2.000, 8.000 e 20.000.
- Consultas: 300 frases (>= 8 palavras) extraídas da passagem-alvo; a passagem de origem é a
  única relevante. Métricas: recall@1, recall@10, MRR@10 (k = 10).
- Métodos (todos com similaridade de cosseno, vetores normalizados):
  1. **TF-IDF + cosseno** (linha de base lexical, esparso).
  2. **Word2Vec médio** (gensim, 100 dim, treinado na própria amostra, média ponderada por IDF).
  3. **Sentence embeddings** (`all-MiniLM-L6-v2`, 384 dim, CPU) com cosseno exato (força bruta).
  4. **Sentence embeddings + HNSW** (usearch, M = 16, efConstruction = 128, efSearch = 64).
- Hardware: i5-13420H, 16 GB, sem GPU.

## Resultados (N = 20.000)

| Método | recall@1 | recall@10 | MRR@10 | Construção | Latência/consulta | Índice |
|---|---|---|---|---|---|---|
| TF-IDF + cosseno | 0,983 | 1,000 | 0,990 | 5 s | 12 ms | 6 MB |
| Word2Vec médio | 0,267 | 0,453 | 0,318 | 32 s | 1,6 ms | 8 MB |
| Sentence + cosseno exato | 0,747 | 0,877 | 0,788 | 460 s | 15 ms (busca: 1,5 ms) | 29 MB |
| Sentence + HNSW | 0,737 | 0,857 | 0,775 | 467 s | 14 ms (busca: 0,7 ms) | 62 MB |

Em N = 2.000 e 8.000 a ordem de qualidade é a mesma (TF-IDF > sentence > Word2Vec). HNSW
manteve recall frente ao exato de 0,97–0,99. A latência total do sentence embedding é dominada
pela codificação da consulta (~14 ms), não pela busca. A codificação das passagens (~43/s na CPU)
é de longe o maior custo de construção.

## Escalabilidade da busca vetorial (384 dim, vetores sintéticos agrupados)

| N | Força bruta | HNSW | Recall@10 HNSW | Construção HNSW |
|---|---|---|---|---|
| 10 mil | 0,59 ms | 0,50 ms | 0,99 | 5 s |
| 100 mil | 8,8 ms | 0,55 ms | 0,99 | 21 s |
| 500 mil | 38 ms | 1,1 ms | 0,86 | 170 s |
| 1 milhão | 75 ms | 1,4 ms | 0,77 | 468 s |

A força bruta cresce linearmente; o HNSW, aproximadamente logarítmico. O recall do HNSW cai
em N grande com parâmetros fixos; seria preciso aumentar `efSearch`/`M` (mais latência e memória).
Os vetores sintéticos são mais difíceis que embeddings reais, então esse recall é pessimista.

## Complexidade

| Método | Construção | Consulta | Memória |
|---|---|---|---|
| TF-IDF | O(tokens) | O(soma dos postings dos termos); aqui, produto esparso | O(tokens únicos por doc) |
| Word2Vec médio | treino O(épocas·tokens·d) + O(N·d) | O(N·d) | O(N·d) |
| Sentence + cosseno exato | N inferências do modelo | O(N·d) + 1 inferência | O(N·d) |
| Sentence + HNSW | N inferências + O(N·log N) | ~O(log N) + 1 inferência | O(N·(d + M)) |

## Conclusões

- **Qualidade:** sentence embeddings superam muito o Word2Vec médio (que, treinado em só
  20 mil passagens, é fraco e instável). O TF-IDF venceu neste teste, mas há viés: a consulta é uma
  frase da própria passagem, com forte sobreposição lexical. Com perguntas reais parafraseadas,
  espera-se vantagem dos embeddings; isso não foi medido (não há perguntas rotuladas).
- **Desempenho:** até ~100 mil passagens a força bruta basta e é exata. Acima disso, HNSW
  reduz a latência de busca em dezenas de vezes (75 ms -> 1,4 ms em 1 M), ao custo de
  construção do grafo, ~2x de memória e recall aproximado.
- **Recomendação:** sentence embeddings + HNSW para o corpus completo, possivelmente combinado
  com TF-IDF/BM25 (híbrido) para consultas lexicais.
- **Limite prático:** o corpus de 10 GB tem da ordem de 14 milhões de passagens (estimativa a ~700
  caracteres por passagem). A 43 passagens/s em CPU, codificá-las levaria ~90 horas e os vetores
  ocupariam ~21 GB (float32). Indexar tudo exige GPU, quantização (usearch suporta f16/i8) ou
  índice em disco; por isso os testes usam amostras.
