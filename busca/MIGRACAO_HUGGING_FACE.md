# Migração: autenticação e downloads do Hugging Face

Este documento descreve as alterações da branch em relação à `main`, a visão
executiva das decisões tomadas, os passos para atualizar os ambientes da equipe
e os pontos de validação. A migração não treina modelos nem garante a qualidade
das respostas.

## Visão executiva

### 1. Cumprimento dos objetivos iniciais

- **Melhoria nas taxas de transferência:** o projeto deixou de depender apenas
  de requisições HTTP convencionais nos downloads do dataset e passou a utilizar
  a infraestrutura nativa do Hugging Face Hub integrada ao protocolo **Xet** em
  modo de alto desempenho, aliada ao suporte a autenticação por token ou CLI
  (reduzindo gargalos de limite de requisição de acessos anônimos).
- **Divisão de responsabilidades e fim da duplicação:** foi criada uma camada
  dedicada de infraestrutura compartilhada. Antes, o módulo de busca importava
  diretamente rotinas internas do módulo de coleta de dados, misturando
  responsabilidades. Agora, a separação está organizada em três níveis:
  1. **Infraestrutura:** configuração do ambiente, autenticação,
     obtenção/validação do manifesto e download/verificação dos arquivos da
     Wikipédia.
  2. **Dados:** regras de negócio de coleta, filtragem temática e extração em
     lote.
  3. **Busca:** amostragem de passagens, geração de embeddings, recuperação
     híbrida, geração de respostas e avaliação.

### 2. Questões adicionais incorporadas

Ao unificar o acesso ao Hub e reorganizar os módulos, surgiram necessidades
práticas de confiabilidade, reprodutibilidade e operação que também foram
tratadas na branch:

- **Política previsível de autenticação e segurança:** o sistema passou a
  diferenciar o uso intencionalmente anônimo (sem token configurado) do uso com
  credencial inválida, interrompendo a execução imediatamente neste último caso.
  Também foi garantido que credenciais não vazem em logs nem venham preenchidas
  com valores fictícios em arquivos de exemplo.
- **Modo offline e tolerância a quedas de rede:** foi adicionado suporte
  explícito para operação sem internet e um mecanismo de contingência que,
  diante de indisponibilidade ou instabilidade de rede, passa a utilizar os
  arquivos já presentes no cache local e preserva essa decisão ao longo da
  execução.
- **Rastreabilidade e integridade dos caches:** os caches gerados pela busca
  passaram a registrar sua proveniência (revisão da base, parâmetros de
  amostragem, revisão fixa do modelo e versões das bibliotecas), com gravação
  atômica para evitar arquivos corrompidos em caso de interrupção e rejeição
  explícita de arquivos adulterados ou inválidos. Caches antigos sem esses
  registros são preservados em disco e exigem autorização explícita para serem
  recriados.
- **Padronização de ambiente, higiene do Git e testes:** as dependências dos
  módulos de coleta e busca foram alinhadas para coexistirem no mesmo ambiente
  virtual sem conflitos; arquivos de cache derivados e bytecode que estavam
  versionados foram removidos do controle de versão; a revisão do modelo de
  embeddings foi uniformizada entre consulta, avaliação e benchmark; e uma suíte
  de testes automatizados e documentação de migração foi incorporada.

### 3. Avaliação geral

Embora a branch tenha ido além da meta inicial de velocidade de download e
refatoração modular, as entregas adicionais foram **coerentes e necessárias**.
Em sistemas que lidam com gigabytes de dados e modelos pesados, trocar o
mecanismo de download e compartilhar caches sem tratar autenticação, quedas de
rede, integridade de arquivos e invalidação de caches antigos geraria falhas
difíceis de diagnosticar para a equipe. As mudanças entregam uma arquitetura
modular, reprodutível e resiliente.

---

## Resumo técnico

- `infraestrutura/configuracao_hf.py` carrega o `.env` da raiz antes das
  bibliotecas do Hub (descartando `HF_TOKEN` vazio ou composto apenas por
  espaços); os scripts de busca mantêm uma entrada de compatibilidade.
- O token pode vir de `HF_TOKEN` ou do login local com `hf auth login`.
- `infraestrutura/huggingface.py` verifica credenciais ao iniciar o RAG ou
  carregar modelos no benchmark; sem credencial, o acesso público é anônimo. A
  verificação não ocorre offline, e uma credencial inválida não é trocada por
  acesso anônimo.
- O Hub usa `hf_xet`, com `HF_XET_HIGH_PERFORMANCE=1` por padrão. Se o módulo
  não puder ser importado ou estiver desabilitado via `HF_HUB_DISABLE_XET=1`,
  a configuração emite um aviso; os downloads usam HTTP. Como `hf-xet` também está nos
  requisitos obrigatórios, esse fallback de execução não resolve uma falha
  de instalação do pacote em uma plataforma incompatível.
- `infraestrutura/wikipedia.py` centraliza revisão, manifesto e acesso aos
  shards ingleses; `corpus.py` cuida da amostragem.
- A coleta inglesa e `extrair_10gb_en.py` compartilham o acesso ao Hub e ajustam
  seu `sys.path` de forma independente no topo do script. Dumps da Wikimedia
  continuam no downloader próprio; parciais antigos (`.part`) de shards podem
  ser retomados pelo coletor via HTTP público no modo online, sem revalidação
  SHA-256 duplicada após a conclusão. A coleta portuguesa em `coletar_lote.py`
  mantém seu downloader anterior.

`hf_transfer` não é a tecnologia usada pelo Hub 1.x. A alternativa suportada
para aceleração é o `hf_xet`; não é necessário configurar
`HF_HUB_ENABLE_HF_TRANSFER`. Consulte a
[documentação oficial](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables).

## Mensagens INFO e validação

`consultar.py` mostra mensagens `INFO` sem exigir opções adicionais. Elas
indicam o modo de acesso, a validação da credencial quando realizada, o
carregamento dos modelos, uso de cache de passagens, consulta do manifesto e
validação dos shards. Uma solicitação ao Hub pode ser atendida pelo cache
local; nem ela nem um download de dataset público comprovam autenticação.
Nenhuma mensagem da aplicação imprime o valor do token.

Um teste manual com credencial real (ver "Validações realizadas e limites")
confirmou a aceitação do token e um download novo via Xet. Cada membro da equipe
pode repetir a verificação no próprio ambiente. Evite divulgar o token ou
compartilhar logs de diagnóstico que possam conter dados sensíveis. As mensagens
da aplicação não informam qual transporte foi usado.

## Migração do ambiente

1. **Preserve o ambiente existente.** Prefira experimentar em um ambiente
   virtual Python 3.12 separado antes de atualizar o ambiente de trabalho.
2. Na raiz do projeto, com o ambiente de teste ativo, simule a resolução dos
   dois conjuntos de dependências:

   ```powershell
   python -m pip install --dry-run -r busca\requirements.txt -r dados\requirements.txt
   ```

   Revise as versões que seriam instaladas ou substituídas. A busca limita
   `datasets` à série 5.0 e `pyarrow` a `>=21,<24`: `datasets` 5.1 e 5.2
   exigem `pyarrow>=24`, enquanto `dados` exige `pyarrow>=18,<24`.
   A migração também exige `transformers>=5.23,<6` e
   `sentence-transformers>=6.3,<7`; ambientes antigos podem precisar
   atualizar esses pacotes e suas dependências. Algumas versões antigas de
   `transformers` exigem `huggingface_hub<1`, incompatível com esta migração.
   `pip check` não verifica por si só se um ambiente satisfaz as restrições de
   arquivos de requisitos que não foram instalados em conjunto.
3. Após revisar a resolução, instale os requisitos necessários no ambiente de
   teste e execute `python -m pip check`. Se usar os dois componentes no mesmo
   ambiente, instale os dois arquivos na mesma operação para que o resolvedor
   considere ambas as restrições:

   ```powershell
   python -m pip install -r busca\requirements.txt -r dados\requirements.txt
   python -m pip check
   ```

   A resolução e a execução devem ser verificadas nesse ambiente antes de
   atualizar outro ambiente da equipe. Os limites restringem as dependências
   diretamente relacionadas à integração, mas não constituem um lockfile:
   versões dentro dessas faixas e outras dependências podem mudar.
4. Autentique-se por uma das opções:
   - Copie `.env.example` para `.env` na raiz e preencha `HF_TOKEN` com um
     token de leitura criado no Hugging Face.
   - Ou use `hf auth login`; nesse caso, mantenha `HF_TOKEN=` vazio como vem no
     `.env.example` (ou remova a linha). Valores vazios ou só com espaços são
     descartados na inicialização para preservar o login da CLI e o acesso
     público anônimo.

   Uma variável de ambiente já definida prevalece sobre o `.env`; um
   `HF_TOKEN` preenchido prevalece sobre o login salvo pela CLI. Não compartilhe
   o token. O `.env` é ignorado pelo Git, mas precisa ser excluído manualmente
   de qualquer ZIP de entrega.
5. Se o modo de alto desempenho consumir recursos demais, defina
   `HF_XET_HIGH_PERFORMANCE=0` no ambiente ou no `.env` antes da execução.
   As configurações do Hub devem ser estabelecidas antes da importação das
   bibliotecas; scripts e testes que já as importaram devem ser reiniciados
   (os módulos de teste da busca executam essa configuração antes de importar o
   Hub).

### Rede e offline

O padrão é online. Defina `HF_HUB_OFFLINE=1` no ambiente ou no `.env` antes
de iniciar o processo para proibir consultas ao Hub; nesse modo a credencial
não é verificada e downloads parciais `.part` não são retomados. Se não houver
JSON de passagens para os parâmetros pedidos, o manifesto validado de uma
execução online desta versão e todos os shards selecionados precisam estar
disponíveis localmente. O manifesto é salvo em
`dados/cache_ingles/manifesto_wikipedia_<revisao>.json`; a versão no antigo
`busca/cache_indice` continua sendo aceita offline. Seus metadados são
usados para verificar tamanho e SHA-256 dos shards também offline. Caches
antigos apenas com Parquet não incluem esse manifesto e precisam de uma
preparação online ou de um JSON de passagens já existente. Os modelos também
precisam estar completos no cache; embeddings podem ser recalculados.

Se a verificação da credencial falhar por conexão, timeout ou erro de protocolo
HTTP (`RemoteProtocolError`), o RAG avisa e continua somente com arquivos
locais, propagando o estado `offline` para as etapas seguintes. Da mesma forma,
falhas de rede na consulta do manifesto, obtenção de shards (`baixar_shards`) ou
carregamento de modelos via `carregar_modelo` (mesmo quando encapsuladas em
cadeias `__cause__`/`__context__`) acionam o fallback com `local_files_only=True`
e propagam `offline=True` entre embeddings, corpus, LLM e avaliação RAGAS.
Quando o carregamento de modelos recorre internamente ao cache sem propagar
exceção (observado com `SentenceTransformer`/`AutoTokenizer` sem token), o
estado retornado por `carregar_modelo` permanece inalterado para chamadas
posteriores. Não há fallback geral para erro HTTP (`4xx`/`5xx`), credencial
inválida, erro de servidor ou falha arbitrária de transferência; arquivos
ausentes ou inválidos causam erro. No modo online, uma credencial configurada é
verificada mesmo quando todos os arquivos estão em cache.

## Por que `corpus.py` mudou?

### Fluxo anterior

O módulo importava `manifesto` e `baixar_continuavel` de
`dados/coletar_ingles.py`. O manifesto era consultado por `urllib` e o mesmo
downloader fazia requisições HTTP, retomava arquivos `.part` e validava os
shards em `dados/cache_ingles`. Esse fluxo próprio não usa automaticamente
as credenciais ou o mecanismo de transferência configurados no Hub.

### Fluxo atual

1. A configuração do ambiente é carregada antes de importar o Hub.
2. `HfApi.list_repo_tree` consulta `wikimedia/wikipedia/20231101.en`, usando
   a revisão definida em `infraestrutura/wikipedia.py`.
3. O código exige 41 entradas com nomes esperados e metadados LFS e rejeita
   nomes duplicados. Salva o manifesto validado de forma atômica para uso
   offline. A ordem por nome continua determinando os primeiros `n_shards`
   selecionados.
4. Para cada shard, o código reutiliza primeiro um arquivo completo já
   existente em `dados/cache_ingles`. Se não existir, chama `hf_hub_download`,
   que usa as credenciais e o cache do Hub e pode usar Xet.
5. Em ambos os casos, o arquivo é verificado pelo tamanho e SHA-256 informados
   no manifesto antes de ser lido pelo PyArrow. Um arquivo inválido causa erro;
   não é substituído silenciosamente.
6. Os artigos são divididos em parágrafos e amostrados; cada passagem mantém
   texto, título e URL para a recuperação e a apresentação das fontes.

A mudança do downloader é necessária porque apenas definir variáveis do Hub
não altera o comportamento das requisições `urllib` anteriores. Ela também
muda o local dos **novos** downloads para o cache do Hugging Face. Arquivos
`.part` do downloader antigo não são retomados pelo Hub. No modo online, o
coletor inglês pode retomá-los pelo downloader antigo usando HTTP público (sem
recalcular o SHA-256 em duplicidade após validar e renomear o arquivo); no modo
offline ou na busca, `.part` não são retomados. Na execução online que precisa
construir o corpus, o manifesto remoto ainda é consultado, mesmo que os shards
antigos já estejam em disco.

### Caches de passagens e embeddings

Os antigos JSONs e arquivos `.npy` em `busca/cache_indice` não registram todos
os parâmetros e não são apagados nem reutilizados automaticamente. Se existir
um JSON antigo com os parâmetros solicitados e não houver cache novo, o
programa interrompe a execução e pede `--regenerar-cache`. Essa opção
reamostra os shards e calcula os embeddings; não atribui proveniência suposta
aos arquivos antigos. Pode exigir rede, espaço e tempo de processamento.

Os novos JSONs incluem revisão, quantidade, semente, shards e parâmetros de
amostragem, além de checksum do conteúdo; os diretórios de cache são criados
automaticamente e todas as gravações usam `gravar_atomico` com `flush` e
`fsync` antes da substituição final. Os novos `.npz` contêm metadados com o
hash das passagens, modelo e revisão, configurações de codificação e versões
de bibliotecas, validando tipos, formato e integridade na leitura e rejeitando
arquivos corrompidos ou adulterados com `ValueError`.

O modelo de embeddings está fixado na revisão
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` em `rag.py`, `avaliar.py` (embeddings
do RAGAS) e `benchmark.py` (padrão para o modelo do projeto, com opção
`--revisao`). Isso evita divergência de revisão e falhas offline quando o cache
contém apenas `snapshots/<revisao>` sem `refs/main`. Alterações na amostragem,
nas passagens ou no modelo geram outra identidade de cache. Esses arquivos
derivados não precisam ser versionados.

## Validações realizadas e limites

- **Download autenticado e transporte (teste manual).** Com token real e sem
  modo offline, na venv isolada (`huggingface_hub` 1.33.0, `hf-xet` instalado,
  `HF_XET_HIGH_PERFORMANCE=1`): `whoami` foi aceito pelo Hub; o shard
  `train-00003-of-00041.parquet` (330.779.424 bytes), ausente do cache local, foi
  baixado para um cache temporário vazio e validado por tamanho e SHA-256 contra
  o manifesto. Instrumentando `file_download.xet_get` e `file_download.http_get`
  com `wraps`, houve 1 chamada a `xet_get` e nenhuma a `http_get`: o transporte
  foi **Xet**. O token não foi impresso, o cache temporário foi removido e o
  teste não alterou o repositório. Isso depende de funções internas do Hub 1.33
  e pode não refletir outras versões, arquivos ou redes; o fallback HTTP e a
  retomada de downloads interrompidos por rede real não foram exercitados.
- **Suítes automatizadas após os ajustes finais:** passaram **58 testes
  unitários** (**44 em `busca/`** e **14 em `dados/`**), cobrindo
  configuração do `.env` e `HF_TOKEN` vazio, ordem de importação, política de
  autenticação, propagação do estado `offline` (inclusive entre execuções do
  benchmark) e `RemoteProtocolError` em cadeias de exceções, fallback em
  `carregar_modelo`, manifesto, shards, bloqueio de retomada de `.part` em
  `HF_HUB_OFFLINE=1`, criação de diretórios em gravação atômica, rejeição de
  `.json`/`.npz` corrompidos ou com largura incompatível com o modelo e uso de
  `REVISAO_EMB` em `avaliar.py` e `benchmark.py`.
- **Validações locais sem rede (`HF_HUB_OFFLINE=1`):** com cache temporário
  contendo apenas `snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41` e
  `blobs/` (sem `refs/main`), o carregamento de embeddings de `avaliar.py`
  (`LangchainEmbeddingsWrapper.embed_query`) e `benchmark.metodo_sentence` com
  `revisao=""` executaram com sucesso. A inicialização dos scripts CLI via
  `--help`, a importação direta dos módulos e `python -m compileall -q
  infraestrutura busca dados` também passaram sem erros.
- **Limites conhecidos:** a execução completa ponta a ponta de
  `busca/avaliar.py` com `ragas.evaluate` e LLM juiz real não foi rodada até o
  fim em CPU, e a retomada HTTP real de shards `.part` de ~330 MB foi validada
  apenas por testes unitários simulados. A consulta experimental e a resolução de
  dependências em ambiente limpo (`datasets` 5.0.1, `pyarrow` 23.0.1,
  `huggingface_hub` 1.33.0, `hf-xet` 1.8.0) continuam sujeitas à resolução de
  pacotes enquanto não houver lockfile.

## Estado do repositório e verificação

- `infraestrutura/`, os módulos e testes novos, `.env.example` (com `HF_TOKEN=`
  vazio) e `.gitignore` consolidado (com `*.py[cod]` e sem duplicatas) já estão
  integrados na branch.
- Os padrões de amostragem permanecem `min_car=300, por_artigo=3`.
- Os caches derivados anteriormente rastreados em `busca/cache_indice/` e os
  arquivos `__pycache__/*.pyc` já foram removidos do rastreamento do Git. Ao
  preparar um ZIP de entrega, exclua manualmente `.env`, `.vscode/` e caches
  locais gerados em execução.

Testes, a partir da raiz e com as dependências instaladas:

```powershell
python -m unittest discover -s busca -p "test_*.py" -v
python -m unittest discover -s dados -p "test_coletar*.py" -v
```
