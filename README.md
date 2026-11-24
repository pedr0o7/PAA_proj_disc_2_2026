# Sistema RAG com LLM Local para Consulta a Livros do Project Gutenberg

![GitHub repo size](https://img.shields.io/github/repo-size/pedr0o7/PAA_proj_disc_2_2026?style=for-the-badge)
![GitHub language count](https://img.shields.io/github/languages/count/pedr0o7/PAA_proj_disc_2_2026?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/pedr0o7/PAA_proj_disc_2_2026?style=for-the-badge)
![GitHub open issues](https://img.shields.io/github/issues/pedr0o7/PAA_proj_disc_2_2026?style=for-the-badge)
![GitHub open pull requests](https://img.shields.io/github/issues-pr/pedr0o7/PAA_proj_disc_2_2026?style=for-the-badge)

<p align="center">
  <img src="assets/logo-cic-unb.png" alt="Logo do CIC/UnB" width="220">
</p>

> Projeto acadêmico que implementa um sistema de Retrieval-Augmented Generation (RAG) para responder perguntas sobre literatura, com base em passagens de livros em inglês de domínio público do Project Gutenberg (corpus de pelo menos 10 GB de texto). O sistema combina busca híbrida — semântica e por palavras-chave — com um modelo de linguagem executado localmente, apresentando as fontes recuperadas (livro, autor e link).

## Ajustes e melhorias

O projeto está em desenvolvimento. Entre as próximas melhorias estão:

- [x] Busca híbrida com BM25 e embeddings semânticos
- [x] Geração local de respostas com o modelo SmolLM2
- [x] Exibição das fontes recuperadas e citações
- [x] Ferramentas de avaliação e benchmarking
- [ ] Indexar uma fração maior do corpus de livros
- [x] Rejeitar respostas que não citam uma passagem recuperada
- [ ] Aprimorar a validação semântica de citações
- [x] Interface web de chat com respostas em tempo real e fontes clicáveis
- [ ] Ampliar e melhorar a cobertura do corpus
- [ ] Aprimorar a extração e a validação de citações
- [ ] Expandir os testes e a avaliação das respostas

## 💻 Pré-requisitos

Antes de começar, verifique se você tem:

- Python 3.12.
- Windows 11, sistema usado nos testes do projeto. Linux e macOS podem funcionar, mas não foram validados.
- Conexão com a internet para baixar os dados e os modelos do Hugging Face.
- Cerca de 25 GB livres em disco (download de até ~11 GB, texto extraído de 10 GB e índice) e memória suficiente para o índice e o modelo local.
- Familiaridade básica com terminal e ambientes virtuais Python.

## 🚀 Instalando

Abra um terminal na pasta do projeto e crie um ambiente virtual.

### Windows (PowerShell)

```powershell
cd busca
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux e macOS

```bash
cd busca
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Coletar o corpus

Antes da primeira consulta, baixe os livros (a partir da raiz do projeto; veja [dados/README.md](dados/README.md)):

```powershell
python dados\coletar_gutenberg.py --contato "seu-email@exemplo.com"
```

Na primeira consulta, o programa escolhe até 20.000 passagens, calcula os embeddings (cerca de 20 minutos em CPU, dependendo da máquina) e baixa os modelos necessários. O cache é reutilizado nas consultas seguintes.

### Autenticação no Hugging Face

Após instalar as dependências, autentique-se de uma das formas:

- Copie `.env.example` da raiz para `.env` e preencha `HF_TOKEN` com um [Access Token](https://huggingface.co/settings/tokens) de leitura.
- Ou execute `hf auth login` e informe o token quando solicitado. Nesse caso, não é necessário preencher `HF_TOKEN`.

Variáveis já definidas no ambiente têm prioridade sobre o `.env`; um `HF_TOKEN` preenchido tem prioridade sobre o login da CLI. O `.env` é ignorado pelo Git, mas **remova-o também do ZIP de entrega**.

O acesso usa a rede por padrão e, sem credencial, os recursos públicos são acessados anonimamente. Quando há credencial, o pipeline a verifica no início; uma credencial inválida causa erro explícito, sem tentativa anônima. A verificação exige uma chamada adicional ao Hub.

Os downloads dos modelos pelo Hub usam `hf_xet` em modo de alto desempenho por padrão. Para desativar esse modo, defina `HF_XET_HIGH_PERFORMANCE=0` no `.env` ou no ambiente. Se `hf_xet` não estiver disponível, ou se `HF_HUB_DISABLE_XET=1`, será exibido um aviso e os downloads usarão HTTP. O corpus Gutenberg é coletado separadamente pelo script descrito em [dados/README.md](dados/README.md).

Para executar **offline**, prepare previamente os modelos e o corpus com rede e defina `HF_HUB_OFFLINE=1` antes de iniciar o programa. No PowerShell, a partir de `busca`:

```powershell
$env:HF_HUB_OFFLINE = "1"
python consultar.py --sem-llm -p "What is the capital of France?"
Remove-Item Env:\HF_HUB_OFFLINE
```

Uma variável já definida no terminal tem prioridade sobre o `.env`; reinicie o processo ao alterar o modo. Offline, a credencial não é verificada e arquivos de modelo ausentes causam erro. É necessária a revisão fixa do modelo de embeddings e, se usado, o modelo de geração. O corpus Gutenberg deve ter sido coletado previamente; embeddings ausentes podem ser calculados a partir do modelo e das passagens locais.

Uma falha de conexão ou timeout durante a verificação de credencial permite tentar os caches locais, com aviso; as bibliotecas de modelos também podem usar seus caches quando a rede falha. Isso não garante fallback para toda falha: erros de autenticação, permissão, servidor ou transferência podem interromper a execução. Um cache incompleto também causa erro.

Na consulta, mensagens `INFO` indicam o carregamento dos modelos. A validação da credencial é informada quando concluída; usar um cache ou baixar um recurso público por si só não comprova autenticação. Os logs não exibem o token e não confirmam qual transporte foi usado.

## Organização

`infraestrutura/` concentra a configuração de acesso ao Hub e componentes de
manifesto e integridade; `dados/` coleta e extrai o corpus Gutenberg; `busca/`
amostra passagens, gera embeddings e executa consultas e avaliações; `frontend/`
contém a interface web local.

## ☕ Usando

Execute os comandos a partir da pasta `busca`.

### Fazer uma pergunta

```bash
python consultar.py -p "Who is Captain Ahab?"
```

### Iniciar o modo interativo

```bash
python consultar.py
```

Digite as perguntas em inglês. Para encerrar, envie uma linha vazia ou digite `sair`.

### Consultar sem gerar resposta

Este modo executa a recuperação e mostra as passagens encontradas, sem chamar o LLM:

```bash
python consultar.py -p "Who is Captain Ahab?" --sem-llm
```

### Mostrar o conteúdo das passagens recuperadas

```bash
python consultar.py -p "Who is Captain Ahab?" --trechos
```

### Opções principais

| Opção | Descrição | Padrão |
|---|---|---:|
| `-p`, `--pergunta` | Faz uma pergunta única e encerra | — |
| `--n` | Número de passagens indexadas | `20000` |
| `--k` | Número de trechos recuperados por pergunta | `3` |
| `--semente` | Semente usada na amostragem do corpus | `42` |
| `--min-sim` | Similaridade cosseno mínima (não é porcentagem) para a busca semântica; não se aplica a consultas diretas de metadados ou frases | `0.4` |
| `--llm` | Identificador do modelo de geração | SmolLM2-1.7B-Instruct |
| `--sem-llm` | Recupera trechos sem gerar resposta | — |
| `--max-tokens` | Limite de tokens da resposta | `200` |
| `--trechos` | Exibe o texto dos trechos recuperados | — |

Exemplo com mais passagens:

```bash
python consultar.py --n 50000 --k 5 -p "Who is Captain Ahab?"
```

O índice contém uma passagem de cada livro amostrado (o embedding dos 10 GB inteiros não é viável em CPU). Se nenhuma passagem passar o limiar ou o modelo não citar uma passagem válida, a resposta será "contexto insuficiente". Isso evita respostas sem citação, mas não garante por si só que uma citação válida sustente semanticamente a resposta.

### Busca experimental em todo o corpus

Para indexar os livros coletados em passagens sobrepostas, execute uma vez na pasta `busca`:

```powershell
python indice_lexical.py
```

O índice SQLite é salvo em `cache_indice\gutenberg_lexical_v1.sqlite`. A construção lê os 10 GB,
pode demorar bastante e necessita de espaço adicional em disco. Se interrompida, execute o mesmo
comando para retomá-la; um índice incompleto não é usado para consultas. Se o corpus mudar, crie
um novo índice em outro caminho com `--indice CAMINHO`.

Depois de concluída a construção, consulte:

```powershell
python consultar.py --indice-completo --trechos -p "Who wrote Pride and Prejudice?"
```

Para um caminho personalizado, use `--indice-completo CAMINHO`. Nesse modo, `--n` e `--semente`
não limitam o índice: a busca lexical considera o corpus completo e a similaridade semântica é
calculada apenas para os melhores candidatos lexicais. As passagens são divididas em janelas de
tokens sobrepostas para o cálculo de embeddings; a similaridade exibida é o maior cosseno entre
essas janelas e a pergunta. Isso pode aumentar o tempo de recuperação. Não é necessário reconstruir
o índice existente. A busca não garante que uma passagem relevante
seja recuperada, pois candidatos sem palavras correspondentes podem ficar de fora.
O modo padrão por amostragem continua disponível sem `--indice-completo`.

Para perguntas em inglês no formato `Who wrote <título exato>?`, o índice completo também consulta
os metadados do catálogo, quando há um único autor inequívoco. A resposta cita os metadados e não
uma passagem do livro; por isso não é exibida similaridade nem aplicado `--min-sim` nessa consulta.
Indícios de adaptações, autor ausente, colaboradores ou atribuições conflitantes levam à abstenção.
Essa verificação é conservadora, mas depende da qualidade dos metadados e não identifica
necessariamente todas as adaptações ou papéis de colaboradores.
Se o título exato não estiver no catálogo, o sistema se abstém, sem afirmar que a obra não existe.
`--sem-llm` continua apenas recuperando passagens.

Para consultas no formato `Where does the phrase <frase> come from?`, o índice completo procura
primeiro a sequência de palavras no texto, ignorando diferenças de pontuação e maiúsculas.
Também aceita, por exemplo, `Which book contains the quote <frase>?`.
Uma correspondência é apresentada como ocorrência em um livro, não como prova da origem da frase.
Frases com menos de cinco palavras levam à abstenção por serem potencialmente ambíguas.
Se não houver correspondência literal, o sistema pode apresentar um trecho com redação semelhante,
explicitando que a origem e o significado não foram confirmados. Esse recurso compara palavras:
paráfrases com vocabulário muito diferente podem não ser reconhecidas. Na ausência de correspondência
suficiente, o sistema se abstém e mostra apenas candidatos lexicais, se houver.
Esses resultados não usam o cosseno nem `--min-sim`. `--sem-llm` mostra os candidatos sem responder.

Por exemplo, no PowerShell:

```powershell
python consultar.py --indice-completo --trechos -p "Where does the phrase tranquillise the mind as a steady purpose come from?"
```

Para outras perguntas, o LLM ainda pode citar uma passagem que não sustenta a resposta: o validador
verifica os números das citações, mas não confirma automaticamente seu suporte semântico.

## 🖥️ Interface web

Interface de chat com fontes e citações `[n]` clicáveis. O progresso é exibido durante a consulta;
a resposta é apresentada após a validação das citações, sem mostrar texto provisório.

### Tecnologias

| Camada | Tecnologia | Licença | Uso |
|---|---|---|---|
| Servidor | [FastAPI](https://fastapi.tiangolo.com/) | MIT | Rotas HTTP (`/`, `/info`, `/perguntar`) e resposta em fluxo |
| Servidor | [Uvicorn](https://www.uvicorn.org/) | BSD-3-Clause | Servidor ASGI que executa a aplicação |
| Servidor | [Pydantic](https://docs.pydantic.dev/) | MIT | Validação da pergunta recebida (1 a 2000 caracteres) |
| Página | HTML5, CSS3 e JavaScript puro | — | Interface sem frameworks, sem Node.js e sem etapa de build |
| Testes | `unittest` (Python) + `TestClient` do FastAPI + [HTTPX](https://www.python-httpx.org/) | PSF / MIT / BSD-3-Clause | Testes da rota com um sistema RAG falso |

Fontes do próprio sistema operacional (serifada para as respostas, sem serifa para a interface); nenhuma fonte, ícone ou biblioteca externa é carregada.

### Métodos

- **Progresso com NDJSON:** a rota `POST /perguntar` devolve eventos `status`, `fontes`,
  `resposta` (quando há LLM) e `fim`, ou `erro`. A consulta pode levar vários segundos ou minutos em CPU;
  os trechos e a resposta são exibidos depois que o processamento termina.
- **Uma geração por vez:** um `threading.Lock` impede duas gerações simultâneas, que só disputariam os mesmos núcleos da CPU.
- **Leitura do fluxo no navegador:** `fetch` + `ReadableStream` + `TextDecoderStream`, que monta as linhas NDJSON conforme chegam.
- **Citações:** o texto é escapado (proteção contra injeção de HTML) e cada `[n]` válido vira um link para o cartão da fonte `n`; as fontes citadas ficam destacadas ao final.
- **Fontes como cartões expansíveis:** título, trecho e link para o Project Gutenberg, com similaridade
  quando disponível ou identificação de metadados, correspondência textual, redação semelhante ou candidato.
- **Visual responsivo e acessível:** variáveis CSS (*custom properties*), modo escuro automático (`prefers-color-scheme`), animações desligadas para quem prefere menos movimento (`prefers-reduced-motion`), layout para celular e rótulos para leitores de tela.
- **Segurança:** o servidor escuta só em `127.0.0.1` (acesso local), valida a entrada e abre links externos com `rel="noopener noreferrer"`.

### Arquivos

- `frontend/servidor.py`: servidor FastAPI que usa o `SistemaRAG` de `busca/rag.py`.
- `frontend/static/index.html`: página do chat (HTML, CSS e JS em um único arquivo).
- `frontend/test_servidor.py`: testes automáticos.
- `frontend/requirements.txt`: dependências (inclui as de `busca/`).

### Executando

Com o ambiente virtual ativo, a partir da pasta `busca`:

```bash
python -m pip install -r ../frontend/requirements.txt
python ../frontend/servidor.py
```

Por padrão o servidor usa uma amostra de até 5.000 passagens. Para consultar o índice completo já
construído, execute:

```powershell
python ..\frontend\servidor.py --indice-completo
```

Opcionalmente informe um caminho após `--indice-completo`; `--n` só se aplica à amostragem.
O índice deve existir, estar completo e corresponder ao corpus. O servidor mostra erro na
inicialização caso contrário; não muda silenciosamente para a amostra. A interface indica o modo
ativo. `--min-sim` controla o limiar cosseno da busca comum (padrão `0.4`), sem garantir que uma
fonte sustente a resposta. Para inspecionar trechos sem geração, use `--sem-llm`.

## 🧪 Avaliação e testes

Para executar a avaliação sem as métricas do RAGAS:

```bash
python avaliar.py --q 30 --sem-ragas
```

Após criar o índice completo, é possível avaliá-lo com `python avaliar.py --indice-completo
--min-sim 0.4 --sem-ragas`. Revise as perguntas e fontes do conjunto de avaliação antes de
interpretar as métricas; perguntas geradas a partir de passagens indexadas podem favorecer a busca.

Para executar os testes do coletor de dados, a partir da raiz do projeto:

```powershell
py -3.12 -m unittest discover -s dados -p "test_coletar*.py" -v
```

Para testar a configuração do Hugging Face e os downloads do corpus:

```powershell
py -3.12 -m unittest discover -s busca -p "test_*.py" -v
```

## 📚 Documentação

- [Documentação técnica](Documentacao.md)
- [Coleta de dados](dados/README.md)
- [Histórico da migração do Hugging Face](busca/MIGRACAO_HUGGING_FACE.md)

## 📫 Contribuindo

Contribuições são bem-vindas. Para colaborar:

1. Faça um fork deste repositório.
2. Crie uma branch: `git checkout -b minha-contribuicao`.
3. Faça as alterações e crie um commit: `git commit -m "Descreve a alteração"`.
4. Envie a branch: `git push origin minha-contribuicao`.
5. Abra um pull request.

Consulte também a documentação do GitHub sobre [como criar um pull request](https://docs.github.com/pt/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

## 🤝 Integrantes

| Integrante | Matrícula |
|---|---:|
| Esthefany Torres | 212005382 |
| João Felipe Silva | 202014388 |
| Pedro Rafael Faria Ferreira | 190094591 |
| Rafael de Jesus Bento Oliveira | 190115777 |
| Mauro Fernandes de Almeida | 211020910 |

## 📝 Licença

A licença deste projeto ainda não está definida. As bibliotecas e os conjuntos de dados utilizados podem ter licenças e condições de uso próprias; consulte os respectivos termos antes de redistribuí-los ou reutilizá-los.
