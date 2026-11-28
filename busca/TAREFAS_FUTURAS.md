# Tarefas futuras

Registro de trabalho identificado mas deliberadamente adiado. Cada item diz por
que não entrou na branch em que surgiu e o que precisa acontecer para fechá-lo.

> **Nota da auditoria PAA (2026-11-26):** a complexidade declarada em `RELATORIO.md` e `Documentacao.md` foi alinhada à implementação atual (`O(nnz_visitado + N + k log k)`). O gargalo `O(N log N)` na seleção BM25 de `busca/rag.py` já foi corrigido com `np.argpartition`.
>
> **Atualização da esteira (2026-11-27):** já foram concluídos e validados na branch a portabilidade via `.env` (OpenVINO / ONNX / PyTorch com quantização `int8`/`int4` isolada via `uv run`), a remoção do legado `infraestrutura/wikipedia.py`, o índice lexical completo FTS5 (27.740 livros / 11,7M passagens) com deduplicação lexical entre edições, a validação semântica de citações (`_validar_suporte_semantico`) + `EXEMPLOS_GERACAO`, a seleção de modelo via `LLM_MODEL`/`--llm`, as flags `--reconstruir` e `--limpar-parquets` e a suíte de 107 testes automatizados (`80` em `busca/`, `18` em `dados/`, `9` em `frontend/`). Permanecem pendentes apenas os dois blocos abaixo.

## 1. Otimizar a busca TF-IDF em `benchmark.py`

**Origem:** revisão do commit `2341baf` na branch
`feat/auth-infra-caches-huggingface-xet`. Esse commit trocou, sob a descrição
"ajusta tipagem", o produto esparso × esparso de `metodo_tfidf` por um produto
esparso × vetor denso. A mudança foi revertida para preservar a
reprodutibilidade dos resultados publicados e o escopo da branch (autenticação,
caches e modo offline do Hugging Face). Apenas a tipagem foi mantida, via
`typing.cast(csr_matrix, ...)`.

**Por que importa:** os rankings são idênticos, mas a latência medida muda.
Em um corpus sintético com vocabulário comparável ao real (300 consultas):

| N | Vocabulário | Esparso × esparso (atual) | Esparso × denso | Rankings |
|---|---|---|---|---|
| 2.000 | 47 mil | 1,24 ms | 0,63 ms | idênticos |
| 8.000 | 99 mil | 3,49 ms | 1,36 ms | idênticos |
| 20.000 | 133 mil | 7,56 ms | 3,01 ms | idênticos |

Na medição anterior (corpus Wikipédia, N = 20.000), o TF-IDF aparecia com 12 ms,
próximo do sentence embedding (14–15 ms). Nos resultados atuais de `RELATORIO.md`
(Project Gutenberg, N = 20.000), o TF-IDF está em 3,03 ms contra 13,47–19,63 ms
do sentence embedding; com a otimização de índice invertido ou produto esparso ×
denso, a latência tende a cair ainda mais, alterando os números publicados. Por
isso a mudança não pode entrar sem regenerar `resultados.json` e revisar o texto.

**Opções:**

1. **Esparso × denso** (já medida acima): densifica a consulta, com custo
   O(vocabulário) por consulta mais O(nnz da matriz). É simples, mas o custo de
   densificar cresce com o vocabulário.
2. **Índice invertido** (preferível): guardar a matriz transposta em CSR
   (termos × documentos) na construção e, por consulta, somar apenas as linhas
   dos termos presentes, ponderadas pelos pesos da consulta. O custo passa a ser
   O(soma dos postings dos termos da consulta), que é a complexidade já
   descrita em `RELATORIO.md` (seção de complexidade, linha do TF-IDF).

**Critérios para concluir:**

- [ ] Implementar a opção escolhida em `metodo_tfidf` em PR próprio, separado
      de mudanças de infraestrutura.
- [ ] Teste unitário garantindo rankings iguais aos do produto esparso × esparso
      em um corpus pequeno.
- [ ] Rodar `python busca/benchmark.py` com os parâmetros padrão e substituir
      `busca/resultados.json`.
- [ ] Atualizar a tabela de resultados e a linha de complexidade do TF-IDF em
      `RELATORIO.md`, além de qualquer conclusão sobre latência relativa entre
      métodos em `Documentacao.md`.
- [ ] Mensagem de commit descrevendo a otimização como mudança de comportamento
      do benchmark, não como ajuste de tipagem.

## 2. Limites conhecidos da migração do Hugging Face

Itens ainda não validados de ponta a ponta. Detalhes em
`MIGRACAO_HUGGING_FACE.md`, seção "Validações realizadas e limites":

- [ ] Execução completa de `busca/avaliar.py` com `ragas.evaluate` e LLM juiz
      real.
- [ ] Retomada HTTP real de shards `.part` (~330 MB) com rede de verdade; hoje
      só há testes simulados.
- [ ] Lockfile de dependências (`datasets`, `pyarrow`, `huggingface_hub`,
      `hf-xet`) para tornar a instalação em ambiente limpo reproduzível.
