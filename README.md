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

### Autenticação no Hugging Face

Após instalar as dependências, autentique-se de uma das formas:

- Copie `.env.example` da raiz para `.env` e preencha `HF_TOKEN` com um [Access Token](https://huggingface.co/settings/tokens) de leitura.
- Ou execute `hf auth login` e informe o token quando solicitado. Nesse caso, não é necessário preencher `HF_TOKEN`.

Variáveis já definidas no ambiente têm prioridade sobre o `.env`; um `HF_TOKEN` preenchido tem prioridade sobre o login da CLI. O `.env` é ignorado pelo Git, mas **remova-o também do ZIP de entrega**.

O acesso usa a rede por padrão e, sem credencial, os recursos públicos são acessados anonimamente. Quando há credencial, o pipeline a verifica no início; uma credencial inválida causa erro explícito, sem tentativa anônima. A verificação exige uma chamada adicional ao Hub.

Os downloads do Hub usam `hf_xet` em modo de alto desempenho por padrão. Para desativar esse modo, defina `HF_XET_HIGH_PERFORMANCE=0` no `.env` ou no ambiente. Se `hf_xet` não estiver disponível, ou se `HF_HUB_DISABLE_XET=1`, será exibido um aviso e os downloads usarão HTTP. Shards completos em `dados/cache_ingles` são verificados e reutilizados; novos downloads usam o cache do Hugging Face. O coletor inglês ainda retoma seus arquivos `.part` antigos por HTTP público; a busca não os retoma.

Para executar **offline**, prepare previamente os modelos e o corpus com rede e defina `HF_HUB_OFFLINE=1` antes de iniciar o programa. No PowerShell, a partir de `busca`:

```powershell
$env:HF_HUB_OFFLINE = "1"
python consultar.py --sem-llm -p "What is the capital of France?"
Remove-Item Env:\HF_HUB_OFFLINE
```

Uma variável já definida no terminal tem prioridade sobre o `.env`; reinicie o processo ao alterar o modo. Offline, a credencial não é verificada e arquivos ausentes causam erro. É necessária a revisão fixa do modelo de embeddings e, se usado, o modelo de geração. O cache de passagens compatível evita a leitura dos shards; caso ele não exista, são necessários o manifesto salvo em uma execução online e todos os shards selecionados completos. Os shards são verificados pelo tamanho e SHA-256 do manifesto salvo. Embeddings ausentes podem ser calculados a partir do modelo e das passagens locais.

Uma falha de conexão ou timeout durante a verificação de credencial ou consulta do manifesto permite tentar os caches locais, com aviso; as bibliotecas de modelos também podem usar seus caches quando a rede falha. Isso não garante fallback para toda falha: erros de autenticação, permissão, servidor ou transferência podem interromper a execução. Um cache incompleto também causa erro.

Na consulta, mensagens `INFO` indicam o carregamento dos modelos e se o corpus
usa cache ou precisa obter shards. A validação da credencial é informada quando
concluída; usar um cache ou baixar um recurso público por si só não comprova
autenticação. A validação de tamanho e SHA-256 é informada após cada shard ser
verificado. Os logs não exibem o token e não confirmam qual transporte foi usado.

### Caches anteriores

Os novos caches de passagens registram a revisão do corpus e os parâmetros de
amostragem; os embeddings registram a identidade das passagens, a revisão fixa
do modelo e as versões das bibliotecas. Arquivos antigos não possuem esses
metadados e **não são apagados nem reutilizados automaticamente**.

Se um cache antigo corresponder aos parâmetros solicitados, o programa pede
confirmação explícita para regenerar os arquivos novos. Na pasta `busca`:

```powershell
python consultar.py --regenerar-cache --sem-llm -p "What is the capital of France?"
```

Isso não converte os arquivos antigos: amostra as passagens novamente e calcula
os embeddings, podendo exigir rede, espaço e tempo. Offline, o manifesto, os
shards e a revisão do modelo precisam estar completos no cache. A opção também
existe em `avaliar.py` e `gerar_dataset.py`. Depois da criação, não é preciso
repeti-la para a mesma configuração.

## Organização

`infraestrutura/` concentra configuração do Hub, manifesto da Wikipédia e
verificação de integridade; `dados/` coleta e extrai as fontes; `busca/`
amostra passagens, gera embeddings e executa consultas e avaliações.

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
| `--regenerar-cache` | Permite recriar caches antigos sem metadados | desativado |

Exemplo com mais passagens:

```bash
python consultar.py --n 20000 --k 5 -p "What is the capital of France?"
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

Para testar a configuração do Hugging Face e os downloads do corpus:

```powershell
py -3.12 -m unittest discover -s busca -p "test_*.py" -v
```

## 📚 Documentação

- [Documentação técnica](Documentacao.md)
- [Coleta de dados](dados/README.md)
- [Migração do Hugging Face e compatibilidade](busca/MIGRACAO_HUGGING_FACE.md)

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
