# Sistema RAG com LLM Local para Consulta à Wikipédia

![GitHub repo size](https://img.shields.io/github/repo-size/pedr0o7/PAA_proj_disc_2_2026?style=for-the-badge)
![GitHub language count](https://img.shields.io/github/languages/count/pedr0o7/PAA_proj_disc_2_2026?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/pedr0o7/PAA_proj_disc_2_2026?style=for-the-badge)
![GitHub open issues](https://img.shields.io/github/issues/pedr0o7/PAA_proj_disc_2_2026?style=for-the-badge)
![GitHub open pull requests](https://img.shields.io/github/issues-pr/pedr0o7/PAA_proj_disc_2_2026?style=for-the-badge)

<p align="center">
  <img src="assets/logo-cic-unb.png" alt="Logo do CIC/UnB" width="220">
</p>

> Projeto acadêmico que implementa um sistema de Retrieval-Augmented Generation (RAG) para responder perguntas com base em passagens da Wikipédia. O sistema combina busca híbrida — semântica e por palavras-chave — com um modelo de linguagem executado localmente, apresentando as fontes recuperadas.

## Ajustes e melhorias

O projeto está em desenvolvimento. Entre as próximas melhorias estão:

- [x] Busca híbrida com BM25 e embeddings semânticos
- [x] Geração local de respostas com o modelo SmolLM2
- [x] Exibição das fontes recuperadas e citações
- [x] Ferramentas de avaliação e benchmarking
- [ ] Interface web de chat com respostas em tempo real e fontes clicáveis
- [ ] Ampliar e melhorar a cobertura do corpus
- [ ] Aprimorar a extração e a validação de citações
- [ ] Expandir os testes e a avaliação das respostas

## 💻 Pré-requisitos

Antes de começar, verifique se você tem:

- Python 3.12.
- Windows 11, sistema usado nos testes do projeto. Linux e macOS podem funcionar, mas não foram validados.
- Conexão com a internet na primeira execução, para baixar os dados e os modelos do Hugging Face.
- Espaço em disco e memória suficientes para os dados, o índice e o modelo local. A necessidade varia conforme o tamanho do corpus.
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

Na primeira execução, o programa pode baixar as passagens da Wikipédia e os modelos necessários. Esse processo pode levar alguns minutos.

## ☕ Usando

Execute os comandos a partir da pasta `busca`.

### Fazer uma pergunta

```bash
python consultar.py -p "What is the capital of France?"
```

### Iniciar o modo interativo

```bash
python consultar.py
```

Digite as perguntas em inglês. Para encerrar, envie uma linha vazia ou digite `sair`.

### Consultar sem gerar resposta

Este modo executa a recuperação e mostra as passagens encontradas, sem chamar o LLM:

```bash
python consultar.py -p "What is the capital of France?" --sem-llm
```

### Mostrar o conteúdo das passagens recuperadas

```bash
python consultar.py -p "What is the capital of France?" --trechos
```

### Opções principais

| Opção | Descrição | Padrão |
|---|---|---:|
| `-p`, `--pergunta` | Faz uma pergunta única e encerra | — |
| `--n` | Número de passagens indexadas | `5000` |
| `--k` | Número de trechos recuperados por pergunta | `3` |
| `--semente` | Semente usada na amostragem do corpus | `42` |
| `--shards` | Número de arquivos Parquet usados como fonte | `1` |
| `--llm` | Identificador do modelo de geração | SmolLM2-1.7B-Instruct |
| `--sem-llm` | Recupera trechos sem gerar resposta | — |
| `--max-tokens` | Limite de tokens da resposta | `200` |
| `--trechos` | Exibe o texto dos trechos recuperados | — |

Exemplo com mais passagens:

```bash
python consultar.py --n 20000 --k 5 -p "What is the capital of France?"
```

## 🖥️ Interface web

Interface de chat no estilo de assistentes de IA: a pessoa faz a pergunta, as fontes aparecem logo após a busca e a resposta do LLM é exibida enquanto é gerada, com as citações `[n]` clicáveis levando ao trecho de origem.

### Tecnologias

| Camada | Tecnologia | Licença | Uso |
|---|---|---|---|
| Servidor | [FastAPI](https://fastapi.tiangolo.com/) | MIT | Rotas HTTP (`/`, `/info`, `/perguntar`) e resposta em fluxo |
| Servidor | [Uvicorn](https://www.uvicorn.org/) | BSD-3-Clause | Servidor ASGI que executa a aplicação |
| Servidor | [Pydantic](https://docs.pydantic.dev/) | MIT | Validação da pergunta recebida (1 a 2000 caracteres) |
| Servidor | [Transformers](https://github.com/huggingface/transformers) (`TextIteratorStreamer`) | Apache 2.0 | Entrega dos tokens do LLM à medida que são gerados |
| Página | HTML5, CSS3 e JavaScript puro | — | Interface sem frameworks, sem Node.js e sem etapa de build |
| Testes | `unittest` (Python) + `TestClient` do FastAPI + [HTTPX](https://www.python-httpx.org/) | PSF / MIT / BSD-3-Clause | Testes da rota com um sistema RAG falso |

Fontes do próprio sistema operacional (serifada para as respostas, sem serifa para a interface); nenhuma fonte, ícone ou biblioteca externa é carregada.

### Métodos

- **Resposta em fluxo (streaming) com NDJSON:** a rota `POST /perguntar` devolve um objeto JSON por linha, na ordem `fontes` → vários `texto` → `fim` (ou `erro`). As fontes chegam em menos de 0,1 s, enquanto a geração pode levar de 30 s a 3 min na CPU. Assim a tela nunca fica parada.
- **Geração em thread separada:** o `model.generate` roda em uma `threading.Thread`, e o `TextIteratorStreamer` repassa o texto ao servidor parte por parte.
- **Uma geração por vez:** um `threading.Lock` impede duas gerações simultâneas, que só disputariam os mesmos núcleos da CPU.
- **Leitura do fluxo no navegador:** `fetch` + `ReadableStream` + `TextDecoderStream`, que monta as linhas NDJSON conforme chegam.
- **Citações:** o texto é escapado (proteção contra injeção de HTML) e cada `[n]` válido vira um link para o cartão da fonte `n`; as fontes citadas ficam destacadas ao final.
- **Fontes como cartões expansíveis:** elementos nativos `<details>`/`<summary>`, com título, similaridade, trecho e link para a Wikipédia.
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

## 🧪 Avaliação e testes

Para executar a avaliação sem as métricas do RAGAS:

```bash
python avaliar.py --q 30 --sem-ragas
```

Para executar os testes dos coletores de dados, a partir da raiz do projeto:

```powershell
py -3.12 -m unittest discover -s dados -p "test_coletar*.py" -v
```

## 📚 Documentação

- [Documentação técnica](Documentacao.md)
- [Coleta de dados](dados/README.md)

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
