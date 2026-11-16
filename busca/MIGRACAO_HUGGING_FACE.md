# Migração: autenticação e downloads do Hugging Face

Este documento descreve as alterações da branch em relação ao commit anterior,
os passos para atualizar os ambientes da equipe e os pontos que ainda precisam
de validação. A migração não treina modelos nem garante a qualidade das respostas.

## Resumo

- `infraestrutura/configuracao_hf.py` carrega o `.env` da raiz antes das
  bibliotecas do Hub; os scripts de busca mantêm uma entrada de compatibilidade.
- O token pode vir de `HF_TOKEN` ou do login local com `hf auth login`.
- `infraestrutura/huggingface.py` verifica credenciais ao iniciar o RAG ou carregar
  modelos no benchmark; sem credencial, o acesso público é anônimo. A verificação
  não ocorre offline, e uma credencial inválida não é trocada por acesso anônimo.
- O Hub usa `hf_xet`, com `HF_XET_HIGH_PERFORMANCE=1` por padrão. Se o módulo
  não puder ser importado ou estiver desabilitado via `HF_HUB_DISABLE_XET=1`,
  a configuração emite um aviso; os downloads usam HTTP. Como `hf-xet` também está nos
  requisitos obrigatórios, esse fallback de execução não resolve uma falha
  de instalação do pacote em uma plataforma incompatível.
- `infraestrutura/wikipedia.py` centraliza revisão, manifesto e acesso aos
  shards ingleses; `corpus.py` cuida da amostragem.
- A coleta inglesa e `extrair_10gb_en.py` compartilham o acesso ao Hub.
  Dumps da Wikimedia continuam no downloader próprio; parciais antigos de
  shards são retomados pelo coletor via HTTP público. A coleta portuguesa em
  `coletar_lote.py` mantém seu downloader anterior.

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

Um teste manual com credencial real (ver "Validações realizadas") confirmou a
aceitação do token e um download novo via Xet. Cada membro da equipe pode repetir
a verificação no próprio ambiente. Evite divulgar o token ou compartilhar logs de
diagnóstico que possam conter dados sensíveis. As mensagens da aplicação não
informam qual transporte foi usado.

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
   - Ou use `hf auth login`; nesse caso, deixe `HF_TOKEN` vazio ou ausente.

   Uma variável de ambiente já definida prevalece sobre o `.env`; um
   `HF_TOKEN` preenchido prevalece sobre o login salvo pela CLI. Não compartilhe
   o token. O `.env` é ignorado pelo Git, mas precisa ser excluído manualmente
   de qualquer ZIP de entrega.
5. Se o modo de alto desempenho consumir recursos demais, defina
   `HF_XET_HIGH_PERFORMANCE=0` no ambiente ou no `.env` antes da execução.
   As configurações do Hub devem ser estabelecidas antes da importação das
   bibliotecas; scripts que já as importaram devem ser reiniciados.

### Rede e offline

O padrão é online. Defina `HF_HUB_OFFLINE=1` no ambiente ou no `.env` antes
de iniciar o processo para proibir consultas ao Hub; nesse modo a credencial
não é verificada. Se não houver JSON de passagens para os parâmetros pedidos,
o manifesto validado de uma execução online desta versão e todos os shards
selecionados precisam estar disponíveis localmente. O manifesto é salvo em
`dados/cache_ingles/manifesto_wikipedia_<revisao>.json`; a versão no antigo
`busca/cache_indice` continua sendo aceita offline. Seus metadados são
usados para verificar tamanho e SHA-256 dos shards também offline. Caches
antigos apenas com Parquet não incluem esse manifesto e precisam de uma
preparação online ou de um JSON de passagens já existente. Os modelos também
precisam estar completos no cache; embeddings podem ser recalculados.

Se a verificação da credencial falhar por conexão ou timeout, o RAG avisa e
continua somente com arquivos locais, sem verificar a credencial. Se a consulta
do manifesto falhar por conexão ou timeout, usa o manifesto e shards locais
quando disponíveis. As bibliotecas dos modelos podem recorrer ao cache quando
a rede falha. Não há fallback geral para erro HTTP, credencial inválida, erro
de servidor ou falha arbitrária de transferência; arquivos ausentes ou
inválidos causam erro. No modo online, uma credencial configurada é verificada
mesmo quando todos os arquivos estão em cache.

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
   nomes duplicados. Salva o manifesto validado para uso offline. A ordem por
   nome continua determinando os primeiros `n_shards` selecionados.
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
`.part` do downloader antigo não são retomados pelo Hub. O coletor inglês
pode retomá-los pelo downloader antigo usando HTTP público; a busca não os
retoma. Na execução online que precisa construir o corpus, o manifesto remoto
ainda é consultado, mesmo que os shards antigos já estejam em disco.

### Caches de passagens e embeddings

Os antigos JSONs e arquivos `.npy` em `busca/cache_indice` não registram todos
os parâmetros e não são apagados nem reutilizados automaticamente. Se existir
um JSON antigo com os parâmetros solicitados e não houver cache novo, o
programa interrompe a execução e pede `--regenerar-cache`. Essa opção
reamostra os shards e calcula os embeddings; não atribui proveniência suposta
aos arquivos antigos. Pode exigir rede, espaço e tempo de processamento.

Os novos JSONs incluem revisão, quantidade, semente, shards e parâmetros de
amostragem, além de checksum do conteúdo. Os novos `.npz` contêm metadados
com o hash das passagens, modelo e revisão, configurações de codificação e
versões de bibliotecas. O modelo de embeddings está fixado na revisão
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`: uma instalação offline precisa
ter essa revisão em cache. Alterações na amostragem, nas passagens ou no
modelo geram outra identidade de cache. Esses arquivos derivados não precisam
ser versionados.

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
  retomada de downloads interrompidos não foram exercitados com rede real.
- Após a extração da infraestrutura, passaram 26 testes de busca e 12 de
  coleta. Foi feita uma consulta offline com modelo real e corpus sintético,
  verificando criação e reutilização dos dois caches novos. O manifesto
  público e a validação de um shard local também foram exercitados online e
  offline.
- A política de acesso foi exercitada com 20 testes de configuração,
  autenticação simulada, falhas de conexão, manifesto e shards. Uma consulta
  real offline reutilizou modelo e passagens locais; sem modelo em cache,
  produziu erro explícito. Uma consulta anônima ao manifesto público e a
  validação de um shard existente também passaram, inclusive em modo offline
  com o manifesto salvo. Essas verificações não testaram download autenticado
  nem identificaram o transporte Xet.
- A instalação conjunta dos dois arquivos foi concluída em uma venv isolada
  com Python 3.12.14. Foram resolvidos, entre outros, `datasets` 5.0.1,
  `pyarrow` 23.0.1, `huggingface_hub` 1.33.0 e `hf-xet` 1.8.0; `pip check`
  não encontrou conflitos. As dependências não estão fixadas integralmente,
  então instalações futuras podem resolver outras versões.
- Nessa venv, sete testes de configuração e corpus passaram com dependências
  simuladas nos pontos de acesso ao Hub; sete testes dos coletores também
  passaram.
- A consulta real ao manifesto público, sem credenciais, retornou 41 arquivos
  com metadados LFS. O comportamento de retomada do Hub após interrupção
  ainda não foi validado em ambiente da equipe.
- A inicialização dos comandos de consulta, avaliação e benchmark foi testada
  com `--help`. A consulta com 50 mil passagens e três shards foi executada
  com caches locais na venv isolada, incluindo uma geração limitada a 20
  tokens; não validou download nem autenticação. Após os limites de versão
  serem adicionados, a resolução conjunta foi novamente verificada sem usar
  as versões instaladas como ponto de partida.
- As consultas experimentais sobre França e Brasil mostraram respostas corretas
  sem suporte nos trechos, e uma pergunta fictícia recebeu uma resposta
  inventada. O prompt e a geração não foram alterados; essas falhas exigem
  avaliação e tratamento próprios. A revisão do modelo de embeddings foi
  fixada e o limite de 128 tokens é aplicado também ao reutilizar caches.

## Antes de preparar o commit

- Inclua `infraestrutura/`, os módulos e testes novos de busca, `.env.example`
  e `.gitignore`, que precisam ser adicionados ao Git. Não inclua `.env` ou tokens.
- Os padrões de amostragem permanecem `min_car=300, por_artigo=3`. Uma
  alteração anterior para `min_car=80, por_artigo=5` não faz parte desta
  migração.
- A remoção dos caches de índice já rastreados é intencional; confirme-a no
  diff. Arquivos `__pycache__` já rastreados também aparecem modificados:
  `.gitignore` não os remove do rastreamento automaticamente. Não inclua
  bytecode gerado em um commit de código.
- O download novo autenticado via Xet já foi verificado em um ambiente; cada
  membro da equipe deve conferir o próprio token e ambiente antes de depender
  do desempenho esperado.

Testes, a partir da raiz e com as dependências instaladas:

```powershell
python -m unittest discover -s busca -p "test_*.py" -v
python -m unittest discover -s dados -p "test_coletar*.py" -v
```
