# Análise do Problema: Resposta Incorreta do RAG

## Problema Identificado

Ao perguntar **"What is the capital of France?"**, o sistema retornou:
```
The capital of France is [1], which is Nouméa, located in the South Province of New Caledonia.
```

Quando a resposta correta deveria ser: **Paris**

## Causa Raiz

O problema **não está no modelo LLM ou na arquitetura do RAG**, mas sim **na falta de dados relevantes no corpus**.

### Verificação Realizada

1. **Corpus atual**: 5.000 passagens amostradas aleatoriamente da Wikipedia (com seed=42)
2. **Busca por "Paris"**: Encontrados 79 resultados sobre Paris, mas **NENHUM** menciona que Paris é a capital da França
3. **Busca por "capital of France"**: Nenhuma passagem encontrada
4. **Resultado da busca semântica**: O modelo retornou os 3 documentos mais similares, que casualmente mencionam:
   - South Province, New Caledonia (similaridade 0.408)
   - Bad Bergzabern, Germany (similaridade 0.407)
   - Colomiers, France (similaridade 0.368)

### Por que isso aconteceu?

O corpus foi criado por:
1. Baixar parquets da Wikipédia
2. **Amostrar aleatoriamente** parágrafos entre 300-1.200 caracteres de cada artigo
3. Selecionar até 3 parágrafos por artigo
4. Parar após atingir 5.000 passagens

**Resultado**: O artigo sobre "Paris" pode ter sido incluído, mas os parágrafos que o descrevem (como "Paris is the capital of France and the most populous city of France") provavelmente têm menos de 300 caracteres (são muito curtos para a política de filtro) ou não foram selecionados pela amostragem aleatória.

## Soluções Propostas

### Opção 1: Expandir o Corpus (Mais Efetiva)
**Aumentar o número de passagens** de 5.000 para um valor maior (ex: 50.000 ou 100.000):

```bash
# No consultar.py, alterar:
python consultar.py --n 50000 -p "What is the capital of France?"
```

**Vantagens:**
- Maior cobertura de conhecimento factual
- Melhor para perguntas genéricas e comuns
- Melhor desempenho geral do sistema

**Desvantagens:**
- Tempo de indexação maior (minutos em CPU)
- Mais consumo de RAM
- Tempo de busca pode aumentar (mais documentos para buscar)

### Opção 2: Melhorar a Estratégia de Amostragem
Modificar `corpus.py` para:
- **Reduzir o filtro de tamanho** (ex: 50-500 caracteres em vez de 300-1.200)
- **Aumentar `por_artigo`** de 3 para 5-10 parágrafos por artigo
- **Prioritarizar artigos principais** (Wikipedia mais curtos para tópicos bem-conhecidos)

### Opção 3: Verificar o Modelo de Embeddings
O modelo `sentence-transformers/all-MiniLM-L6-v2` é bom, mas:
- Pode não capturar bem perguntas factuais simples
- Poderia tentar: `all-mpnet-base-v2` (mais preciso, mas 10x maior)

### Opção 4: Hybrid Search
Combinar busca semântica com busca por keywords (BM25):
- Buscar por "capital" E "France" exatamente
- Depois combinar com ranking semântico

---

## Recomendação

**Para corrigir este problema especificamente**: 
- Use `--n 50000` para aumentar o corpus

**Para melhorar o sistema a longo prazo**:
1. Aumentar o corpus para ~50k-100k passagens
2. Reduzir o filtro de tamanho mínimo (permite perguntas factuais curtas)
3. Aumentar `por_artigo` para capturar mais contexto de cada artigo
4. Considerar hybrid search (semântico + BM25)

## Teste Recomendado

```bash
# Teste com corpus maior
cd busca/
python consultar.py --n 50000 -p "What is the capital of France?"

# Se isso resolver o problema, considerar fazer a mesma amostragem por padrão
```
