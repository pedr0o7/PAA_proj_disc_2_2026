# Soluções Recomendadas para Melhorar RAG

## 🎯 Problema Principal: Cobertura do Corpus

O sistema RAG agora funciona tecnicamente bem (recuperação + geração), mas **o dataset de 5000 passagens não contém os fatos necessários**.

**Exemplo**: A pergunta "What is the capital of France?" espera encontrar algo como:
```
"Paris is the capital of France and the capital of the Île-de-France region. ..."
```
Mas essas 5000 passagens não contêm esse parágrafo específico.

## 💡 Solução 1: Aumentar Dataset (RECOMENDADO - Rápido)

### ✅ Teste com 20.000 passagens

```bash
cd busca/

# Limpar cache antigo
rm -f cache_indice/emb_titulo_*
rm -f cache_indice/passagens_titulo_*

# Testar com mais dados (usando mais shards)
python consultar.py --n 20000 --shards 2 -p "What is the capital of France?"
```

**Benefício**: 
- Aumenta em 4x a probabilidade de conter o fato desejado
- Tempo de embedding: ~5-10 min (uma única vez)
- Recuperação continua rápida (0.1-0.3s)

### Resultado esperado
Maior chance de recuperar artigos sobre Paris, França, etc.

## 💡 Solução 2: Melhorar Extração de Passagens

Atualmente, o código split apenas por `\n\n` (duplas quebras). Melhorar isso:

### Modificar `corpus.py`

Adicionar extração inteligente de sentenças:

```python
def amostrar_passagens_sentencas(n, semente, n_shards=1, min_car=150, max_car=500):
    """Extrai sentenças em vez de parágrafos completos."""
    import re
    
    artigos = []
    for arq in baixar_shards(n_shards):
        for lote in pq.ParquetFile(arq).iter_batches(batch_size=2000, columns=["title", "url", "text"]):
            for titulo, url, texto in zip(...):
                # Split por sentença (mais granular)
                sentencas = re.split(r'(?<=[.!?])\s+', texto or "")
                sentencas = [s.strip() for s in sentencas if min_car <= len(s) <= max_car]
                if sentencas:
                    artigos.append((titulo, url, sentencas))
    
    # ... resto similar
    return passagens[:n]
```

**Benefício**:
- ✅ Mais passagens relevantes por artigo
- ✅ Melhor cobertura com mesmo dataset
- ✅ Reduz ruído (parágrafos com múltiplos tópicos)

## 💡 Solução 3: Query Expansion (Médio-Avançado)

Expandir a pergunta antes de buscar:

```python
def expandir_pergunta(pergunta):
    """Gera variações da pergunta para melhor recuperação."""
    # Exemplo: "What is the capital of France?"
    # Variações:
    # - "capital France"
    # - "France capital city"
    # - "Paris France"
    
    # Implementação simplificada:
    tokens = pergunta.lower().split()
    
    # Remover stop words
    stop_words = {"what", "is", "the", "a", "of"}
    palavras_chave = [t for t in tokens if t not in stop_words]
    
    # Queries alternativas
    queries = [
        pergunta,  # Original
        " ".join(palavras_chave),  # Só palavras-chave
        " ".join(reversed(palavras_chave)),  # Reversa
    ]
    return queries
```

Usar em `rag.py`:
```python
def recuperar(self, pergunta):
    """Recupera com query expansion."""
    queries_expandidas = expandir_pergunta(pergunta)
    todos_trechos = set()
    
    for q in queries_expandidas:
        trechos = self._recuperar_hibrido(q)
        todos_trechos.update((t['titulo'], t['url']) for t in trechos)
    
    # Retornar union dos top k
    return list(todos_trechos)[:self.k]
```

## 💡 Solução 4: Caching Estruturado de Fatos (Avançado)

Para perguntas comuns (capitais, datas históricas), usar base de conhecimento estruturada:

```python
# fatos_estruturados.json
{
  "capitals": {
    "france": "Paris",
    "brazil": "Brasília",
    "italy": "Rome",
    ...
  }
}
```

Consultar primeiro:
```python
def perguntar(self, pergunta):
    # Tentar match estruturado
    fato = buscar_fato_estruturado(pergunta)
    if fato:
        return {"resposta": fato, "tipo": "estruturado", ...}
    
    # Fallback para RAG
    return self._perguntar_rag(pergunta)
```

## 📊 Matriz de Soluções

| Solução | Dificuldade | Tempo Implementação | Impacto | Recomendado |
|---------|------------|-------------------|--------|-----------|
| **Aumentar dataset** | ⭐ Muito Fácil | 30 min | 🟢 Alto | ✅ SIM |
| **Melhorar extração** | ⭐⭐ Fácil | 1h | 🟢 Alto | ✅ SIM (depois) |
| **Query expansion** | ⭐⭐ Fácil | 1h | 🟡 Médio | ⭐ Depois |
| **Fatos estruturados** | ⭐⭐⭐ Difícil | 2-3h | 🟢 Alto | ⭐⭐ Último |
| **Modelo embedding melhor** | ⭐⭐⭐ Difícil | 2h + treino | 🟡 Médio | ⚠️ Talvez |

## 🚀 Plano de Ação Recomendado

### Fase 1: Rápida (Hoje - 30 min)
```bash
# Teste com 20k passagens
python consultar.py --n 20000 --shards 2 -p "What is the capital of France?"
# Se melhorar: sucesso!
```

### Fase 2: Melhoria (Amanhã - 1h)
```python
# Refatorar corpus.py para usar sentenças
# Criar dataset de 50.000 passagens
python consultar.py --n 50000 --shards 3 -p "What is the capital of France?"
```

### Fase 3: Otimização (Próxima semana)
- Implementar query expansion
- Adicionar validação de resposta
- Criar métricas de qualidade (BLEU, F1, etc.)

## ⚡ Quick Start - Aumentar Dataset Já

```bash
cd C:\Users\PedroFerreira\Documents\git\PAA_proj_disc_2_2026\busca

# Backup do cache antigo (opcional)
if (Test-Path cache_indice) { 
    Copy-Item cache_indice cache_indice.backup -Recurse 
}

# Limpar embeddings antigos
Remove-Item cache_indice/emb_titulo_* -Force -ErrorAction SilentlyContinue
Remove-Item cache_indice/passagens_titulo_* -Force -ErrorAction SilentlyContinue

# Testar com dataset aumentado
python consultar.py --n 20000 --shards 2 -p "What is the capital of France?"
```

Tempo esperado: 
- 🕐 Primeira vez: ~15-20 min (download + embedding)
- 🕐 Queries depois: ~30 seg cada (rápido)

---

**Versão**: 1.0  
**Data**: 2026-11-11
