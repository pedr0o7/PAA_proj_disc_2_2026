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

Na primeira consulta, o programa sorteia as passagens, calcula os embeddings (alguns minutos em CPU) e baixa os modelos necessários.

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
| `--min-sim` | Similaridade (cosseno) mínima para uma passagem virar fonte; abaixo disso a resposta é "contexto insuficiente" | `0.4` |
| `--llm` | Identificador do modelo de geração | SmolLM2-1.7B-Instruct |
| `--sem-llm` | Recupera trechos sem gerar resposta | — |
| `--max-tokens` | Limite de tokens da resposta | `200` |
| `--trechos` | Exibe o texto dos trechos recuperados | — |

Exemplo com mais passagens:

```bash
python consultar.py --n 50000 --k 5 -p "Who is Captain Ahab?"
```

O índice cobre uma amostra de livros do corpus (o embedding dos 10 GB inteiros não é viável em CPU); perguntas sobre livros fora da amostra recebem "contexto insuficiente".

## 🧪 Avaliação e testes

Para executar a avaliação sem as métricas do RAGAS:

```bash
python avaliar.py --q 30 --sem-ragas
```

Para executar os testes do coletor de dados, a partir da raiz do projeto:

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
