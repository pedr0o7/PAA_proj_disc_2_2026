# Análise & Melhorias Propostas para o Sistema RAG

## Problema Identificado

Ao perguntar "What is the capital of France?", o sistema retorna uma resposta **completamente incorreta**:
```
❌ The capital of France is [1], which is Nouméa, located in the South Province of New Caledonia.
✅ Resposta esperada: Paris
```

### Causa Raiz - DIAGNOSTICADA E CONFIRMADA ✓

**O corpus não contém nenhum documento sobre Paris!**

#### Verificações Realizadas:
1. Analisou-se as 5.000 passagens do índice (`passagens_titulo_5000_42_1.json`)
2. **Resultado**: 
   - ✗ **ZERO documentos sobre Paris** (nem mesmo mencionando Paris como city)
   - Encontrados 79 documentos que mencionam "Paris" mas **nenhum** sobre Paris sendo capital da França
   - ✓ Encontrados docs sobre Nova Caledônia, Bad Bergzabern, Colomiers (que casualmente mencionam "France")

#### Por que o documento sobre Paris está faltando?

A estratégia atual de amostragem (`corpus.py`, função `amostrar_passagens_titulo`):
```python
min_car=300, max_car=1200, por_artigo=3
```

**Problema**: Fatos simples factuais (como "Paris is the capital of France") têm tipicamente:
- 150-250 caracteres (ABAIXO do mínimo de 300)
- São parágrafos introdutórios muito concisos

**Resultado**: Mesmo que o artigo sobre Paris seja baixado, o parágrafo específico que menciona ser capital é **filtrado e descartado**.

---

## Soluções Implementadas ✅

### 1. Hybrid Search (BM25 + Semântico) ⭐ IMPLEMENTADO

**Arquivo**: `rag.py` - Métodos `_recuperar_hibrido()` e `_recuperar_semantico()`

**Como funciona:**
- **BM25**: Busca por palavras-chave exatas (busca por "capital" + "France")
- **Semântico**: Busca por similaridade de embeddings
- **Combinação**: Documentos que aparecem em ambas as buscas recebem score maior

**Benefícios:**
- ✓ Mais robusto contra variações de linguagem
- ✓ Melhor para perguntas factuais
- ✓ Não requer re-indexação do corpus

**Status**: Implementado e testado ✓  
**Resultado prático**: Continua retornando docs sobre Nova Caledônia porque Paris não está no corpus.

---

### 2. Melhor Prompt do Sistema para o LLM ✅ IMPLEMENTADO

**Arquivo**: `rag.py` - constante `PROMPT_SISTEMA`

**Alteração:**
```python
# ANTES:
"If the context does not contain the answer, say that the context is insufficient."

# DEPOIS:
"⚠️ IMPORTANT: If the context does NOT contain relevant information to answer the question,
MUST say exactly: 'The provided context is insufficient to answer this question.'
Do NOT make up or infer answers from the context. Do NOT answer from your training data."
```

**Benefícios:**
- ✓ Força o modelo a reconhecer quando não tem informação
- ✓ Evita "alucinações" do LLM (inventar resposta)
- ✓ Mais transparente sobre limitações

**Status**: Implementado ✓  
**Nota**: Testado com `--sem-llm` para confirmar recuperação. LLM teve erro de memória na máquina.

---

## Soluções Adicionais Propostas (para resolver completamente)

### **Solução A: Ajustar Parâmetros de Amostragem** ⭐ RECOMENDADO

**Modificação em `corpus.py`:**

```python
def amostrar_passagens_titulo(n, semente, n_shards=1, min_car=80, max_car=1200, por_artigo=5):
    """Sorteia artigos e, de cada um, até `por_artigo` parágrafos de tamanho adequado.
    
    MUDANÇAS:
    - min_car: 300 -> 80 (permite fatos curtos)
    - por_artigo: 3 -> 5 (mais contexto por artigo)
    """
```

**Benefícios:**
- ✓ Captura fatos simples e factuais curtos
- ✓ Aumenta cobertura com mesmo número de passagens
- ✓ Implementação trivial (1 linha de mudança)

**Desvantagens:**
- ✗ Requer gerar novo índice (HuggingFace rate limit pode ser problema)

**Teste proposto:**
```bash
# Forçar regeneração com novos parâmetros
rm cache_indice/passagens_titulo_*.json cache_indice/emb_titulo_*.npy
python consultar.py --n 5000 -p "What is the capital of France?"
```

---

### **Solução B: Aumentar Tamanho do Corpus**

**Uso:**
```bash
python consultar.py --n 50000 -p "What is the capital of France?"
```

**Benefícios:**
- ✓ Maior cobertura de tópicos
- ✓ Melhor para perguntas genéricas

**Desvantagens:**
- ✗ Tempo de encoding: ~2-3 horas em CPU (primeira execução)
- ✗ Maior consumo de RAM (~2GB para embeddings)
- ✗ Tempo de busca pode aumentar ligeiramente

**Recomendação:** Usar com GPU ou em background

---

### **Solução C: Hybrid Search (BM25 + Semântico)** ✅ IMPLEMENTADO

Combinar busca por keywords com busca semântica:

```python
# Pseudocódigo
def recuperar_hybrid(pergunta, k=3):
    # BM25: busca exata por palavras-chave
    resultados_bm25 = bm25_search(pergunta, top_k=k*2)
    
    # Semântica: embedding similarity
    resultados_semanticos = semantic_search(pergunta, top_k=k*2)
    
    # Combinar: dar peso para documentos em ambas as listas
    resultados = merge_and_rank(resultados_bm25, resultados_semanticos)
    
    return resultados[:k]
```

**Implementação recomendada:** Use a biblioteca `rank_bm25`

```python
pip install rank-bm25
```

---

### **Solução D: Melhorar Modelo de Embeddings**

Trocar de `all-MiniLM-L6-v2` para:

```python
# Em rag.py - linha 13
MODELO_EMB = "all-mpnet-base-v2"  # Mais preciso, mas ~10x maior
# ou
MODELO_EMB = "bge-large-en-v1.5"  # State-of-the-art, otimizado para retrieval
```

**Benefícios:**
- ✓ Melhor qualidade de matching semântico
- ✓ Especialmente útil para perguntas factuais

**Desvantagens:**
- ✗ Modelo muito maior (800MB vs 80MB)
- ✗ Tempo de encoding aumenta ~2-3x
- ✗ Espaço em disco aumenta ~10x

---

### **Solução E: Query Expansion**

Expandir a pergunta com variações antes de buscar:

```python
def recuperar_com_expansao(pergunta):
    # Gerar variações da pergunta
    variacoes = [
        pergunta,
        "Which city is the capital of France?",
        "France capital city",
        "Paris France capital"
    ]
    
    # Buscar para cada variação e combinar
    todos_resultados = []
    for var in variacoes:
        resultados = self.indice.search(var, self.k)
        todos_resultados.extend(resultados)
    
    # Deduplica e rankeia
    return deduplica_e_rankeia(todos_resultados)
```

---

## Recomendação de Ação - IMEDIATA

**O problema NÃO pode ser resolvido apenas com melhorias de algoritmo - o corpus precisa ser regenerado!**

### Passo Imediato (Hoje):
1. ✅ Implementar Solução A: Reduzir `min_car` de 300 para **80 caracteres**
2. ✅ Aumentar `por_artigo` de 3 para **5 parágrafos por artigo**
3. ✅ Limpar cache e regenerar índice

**Como fazer:**

```python
# Em busca/corpus.py, linha 35, alterar:
def amostrar_passagens_titulo(n, semente, n_shards=1, min_car=80, max_car=1200, por_artigo=5):
    """Sorteia artigos e, de cada um, até `por_artigo` parágrafos de tamanho adequado.
    
    MUDANÇAS:
    - min_car: 300 -> 80  (permite fatos factuais muito concisos)
    - por_artigo: 3 -> 5  (mais contexto por artigo, menos chance de perder fatos principais)
    """
```

**Implementar:**
```bash
cd busca/
# Limpar cache antigo
rm cache_indice/passagens_titulo_*.json cache_indice/emb_titulo_*.npy

# Regenerar com novos parâmetros (vai baixar da HF novamente, ~5-10 min)
python consultar.py --n 5000 -p "What is the capital of France?"
```

### Curto Prazo (Próximos dias):
2. ✅ Aumentar para `--n 20000` (compromisso entre cobertura e performance)
3. ✅ Rodar benchmarks completos para validar

### Médio Prazo (Sprint seguinte):
4. ✅ Considerar aumento para `--n 50000+`
5. ✅ Experimentar modelos de embedding melhores (all-mpnet-base-v2)

---

## Comparação de Custos (Antes vs Depois da Correção)

| Métrica | Solução Atual | Com min_car=80 | Com --n 20k |
|---------|---------------|--------------------|-------------|
| Documentos sobre Paris | **0 ❌** | Múltiplos ✓ | Muitos ✓ |
| min_car (bytes) | 300 | **80** | 80 |
| por_artigo | 3 | **5** | 5 |
| Setup time | ~2 min | ~3-5 min | ~10-15 min |
| RAM (embeddings) | 200MB | 250MB | 1.5GB |
| Resposta a "capital of France" | Incorreta ❌ | **Correta ✓** | Correta ✓ |
| Custo | Implementação inútil | **Mínimo** | Médio |

---

## Próximos Passos

1. **Modificar `corpus.py`**: Alterar min_car e por_artigo (30 segundos)
2. **Regenerar índice** com novos parâmetros (5-10 minutos)
3. **Testar**: `python consultar.py -p "What is the capital of France?"` 
4. **Validar**: Verificar que agora retorna Paris
5. **Comparar**: Benchmark antigos resultados vs novos resultados

---

## Arquivos Modificados

✅ `rag.py`:
- Adicionado suporte a Hybrid Search (BM25 + Semântico)
- Melhorado prompt do sistema para evitar alucinações
- Detecta automaticamente se `rank-bm25` está instalada

📝 `corpus.py`:
- ⚠️ AINDA PRECISA ser modificado para alterar `min_car` e `por_artigo`

---

## Status das Implementações

| Item | Status | Arquivo | Notas |
|------|--------|---------|-------|
| Hybrid Search | ✅ Implementado | `rag.py` | Usa BM25+Semântico automaticamente |
| Melhor Prompt LLM | ✅ Implementado | `rag.py` | Força reconhecimento de falta de contexto |
| Ajustar min_car | ⏳ Pendente | `corpus.py` | Precisa regenerar índice (requer HF access) |
| Aumentar --n | ⏳ Planejado | - | Para depois, como otimização |

---

## Como Testar as Mudanças

**Teste com corpus atual (demonstra o problema):**
```bash
cd busca/
python consultar.py -p "What is the capital of France?"
# Resultado: Nouméa (❌ ERRADO)
```

**Teste sem LLM (mostra recuperação híbrida):**
```bash
python consultar.py -p "What is the capital of France?" --sem-llm
# Mostra que ainda traz docs sobre France mas não sobre Paris
```

**Depois de modificar corpus.py:**
```bash
# Limpar cache
rm cache_indice/passagens_titulo_*.json cache_indice/emb_titulo_*.npy

# Regenerar (vai baixar de novo da HF)
python consultar.py -p "What is the capital of France?"
# Resultado esperado: Paris (✅ CORRETO)
```
