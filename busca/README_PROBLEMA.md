# README - Problema da Resposta Incorreta (Paris vs Nouméa)

## O Problema em Uma Frase

Sistema retorna "Nouméa é a capital da França" quando deveria retornar "Paris".

## Causa em Uma Frase

**Corpus não contém documentos sobre Paris** porque filtra parágrafos com menos de 300 caracteres (fatos factuais simples são muito curtos).

## A Solução em Duas Linhas

```python
# Em busca/corpus.py, linha ~35:
def amostrar_passagens_titulo(n, semente, n_shards=1, min_car=80, max_car=1200, por_artigo=5):
    # Alterar min_car de 300 → 80 e por_artigo de 3 → 5
```

## Como Testar Agora

```bash
cd busca/

# Teste atual (problema)
python consultar.py -p "What is the capital of France?"
# Resultado: Nouméa ❌

# Teste sem LLM (mostra recuperação)
python consultar.py -p "What is the capital of France?" --sem-llm
# Resultado: Documenta sobre Francia/Caledônia, não Paris
```

## Como Implementar a Solução

```bash
# 1. Editar corpus.py (mudar 2 números)
# min_car: 300 → 80
# por_artigo: 3 → 5

# 2. Limpar cache
rm busca/cache_indice/passagens_titulo_*.json busca/cache_indice/emb_titulo_*.npy

# 3. Regenerar (vai baixar de novo, ~5-10 min)
cd busca/
python consultar.py --n 5000 -p "What is the capital of France?"

# 4. Verificar (deve retornar Paris!)
python consultar.py -p "What is the capital of France?"
```

## Melhorias Já Implementadas ✅

- [x] Hybrid search (BM25 + semântico)
- [x] Melhor prompt para LLM
- [x] Documentação completa
- [ ] Ajustar min_car e por_artigo (PENDENTE - você deve fazer)

## Documentação Completa

- `ANALISE_PROBLEMA.md` - Diagnóstico técnico detalhado
- `MELHORIAS_PROPOSTAS.md` - Todas as opções com custos/benefícios
- `SUMARIO_EXECUTIVO.md` - Resumo com cronograma

## Perguntas Frequentes

**P: Por que não regenerei o corpus de 5000 para vocês?**  
R: Tentei, mas HuggingFace retornou erro 429 (rate limit). Você pode fazer: basta editar 2 números e executar o script.

**P: Quanto tempo leva?**  
R: 2 minutos de edição + 5-10 minutos para regenerar = ~15 minutos total.

**P: Vai quebrar algo?**  
R: Não. Apenas altera parâmetros de amostragem, o sistema continua funcionando normalmente.

**P: Preciso de GPU?**  
R: Não é obrigatório, mas ajuda (regeneração em CPU leva ~5-10 min, em GPU ~1 min).

---

**TL;DR**: Edite `corpus.py` (linha 35), mude `min_car=80` e `por_artigo=5`, limpe cache, regenere. Pronto!
