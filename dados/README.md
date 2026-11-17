# Coleta de dados: livros do Project Gutenberg

O tema do projeto é **literatura**. O corpus são livros em inglês de domínio público do
[Project Gutenberg](https://www.gutenberg.org/), obtidos do dataset
[`sedthh/gutenberg_english`](https://huggingface.co/datasets/sedthh/gutenberg_english) no Hugging Face
(48.284 livros, cerca de 18 GB de texto; download de 10,7 GB em 37 arquivos Parquet). O cabeçalho e o rodapé de
licença do Gutenberg já foram removidos pelo dataset, e cada livro traz metadados (id, título, autores).

`coletar_gutenberg.py` baixa os arquivos na ordem, com verificação de tamanho e SHA-256, e para assim que a soma
do texto atinge a meta de 10 GB decimais (10.000.000.000 bytes). Isso exige cerca de 21 dos 37 arquivos
(aproximadamente 6 GB de download).

## Executar

Na raiz do projeto:

```powershell
python -m pip install -r dados\requirements.txt
python dados\coletar_gutenberg.py --contato "seu-email@exemplo.com"
```

Substitua o contato por um endereço real, usado no `User-Agent` das requisições. Reserve cerca de 25 GB livres.
Se o download for interrompido, execute o mesmo comando: o arquivo parcial (`.part`) é retomado. Se ele estiver
corrompido, apague apenas esse `.part`.

Opções: `--meta` (bytes de texto, padrão 10.000.000.000), `--cache` (padrão `dados\cache_gutenberg`) e
`--saida` (padrão `dados\gutenberg_10gb`; é sobrescrita a cada execução).

## Saída (`dados\gutenberg_10gb`)

- `textos.txt`: livros em UTF-8, cada um seguido de uma linha em branco.
- `livros.jsonl`: um livro por linha, com `id` (Text# do Gutenberg), `titulo`, `autores`, `url`
  (`https://www.gutenberg.org/ebooks/<id>`), `deslocamento` e `bytes` do texto dentro de `textos.txt`.
- `relatorio.json`: livros únicos, bytes de texto e `atingiu_meta`. A meta só está comprovada se esse campo for
  verdadeiro. Os livros são deduplicados por `id`, e o último livro é gravado inteiro, então o total pode
  ultrapassar levemente a meta.

Esses dados e o cache não são versionados (veja `.gitignore`).

## Testes

```powershell
python -m unittest discover -s dados -p "test_coletar*.py" -v
```

Os testes não usam a rede.

## Licença e atribuição

Os livros são de domínio público nos Estados Unidos, mas alguns podem ter direitos autorais em outros países; o
próprio dataset avisa para conferir os metadados antes de reutilizar. O dataset está sob licença MIT. O
Project Gutenberg pede a seguinte nota junto aos livros: *"This eBook is for the use of anyone anywhere in the
United States and most other parts of the world at no cost and with almost no restrictions whatsoever. You may
copy it, give it away or re-use it under the terms of the Project Gutenberg License included with this eBook or
online at www.gutenberg.org. If you are not located in the United States, you will have to check the laws of the
country where you are located before using this eBook."* Termos completos:
<https://www.gutenberg.org/policy/permission.html>.
