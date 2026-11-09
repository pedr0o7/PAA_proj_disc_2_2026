# Coleta inicial de geografia

O script `coletar_geografia.py` coleta artigos diretamente associados à
categoria Geografia da Wikipédia em português. A profundidade é zero: não
inclui subcategorias como países e cidades. A classificação é feita pela
Wikipédia e não garante que todo artigo seja igualmente pertinente ao projeto.

Na raiz do projeto, execute no PowerShell:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" dados\coletar_geografia.py --contato "seu-email@exemplo.com"
```

Substitua o contato por um endereço real para identificar o programa perante a
Wikipédia. A saída padrão é `dados\geografia_coleta`. Ela não é sobrescrita:
para uma nova coleta, escolha uma pasta que ainda não exista com `--saida`,
por exemplo `--saida dados\geografia_coleta_2`. Se ocorrer uma falha, a pasta
pode conter uma coleta incompleta, sem relatório; escolha outra pasta ao repetir.

Em `textos`, cada arquivo é nomeado pelo identificador único do artigo. O
`relatorio.json` contém títulos, URLs, revisões, datas e tamanhos em bytes, além
da soma dos textos e da comparação com a meta de 10 GB decimais
(10.000.000.000 bytes). Metadados não entram nessa soma. Os títulos das seções
podem aparecer marcados com sinais de igual no texto retornado pela API.

A classificação e os artigos mudam ao longo do tempo, portanto outra coleta pode
ter resultados diferentes. Para reutilizar os textos, consulte as condições de
licença e atribuição da Wikipédia; o relatório preserva informações de origem.

Para executar os testes:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m unittest discover -s dados -p "test_coletar_geografia.py" -v
```

## Coleta em lote

`coletar_lote.py` identifica a categoria Geografia e suas subcategorias
até a profundidade escolhida, sem repetir artigos. Por padrão usa
profundidade 2. A categoria de origem vem dos dumps da Wikipédia em português
de 01/11/2026; os textos vêm do conjunto `wikimedia/wikipedia`,
`20231101.pt`, de 01/11/2023. Como as datas são diferentes, alguns artigos
selecionados pelas categorias atuais não estarão no conjunto antigo.
O relatório informa quantos não foram encontrados.

As categorias são uma aproximação do tema, não uma garantia de relevância.
Subcategorias mais profundas podem incluir assuntos distantes de geografia.
Um resultado inferior a 10 GB não demonstra que o conjunto completo da
Wikipédia em português ou outros recortes de geografia sejam insuficientes.

Na raiz do projeto:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m pip install -r dados\requirements.txt
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" dados\coletar_lote.py --contato "seu-email@exemplo.com" --profundidade 2
```

O programa verifica os checksums dos downloads e os guarda em
`dados\cache_lote` para reutilização. Exige conexão e vários gigabytes de espaço
em disco. Se um download for interrompido, o respectivo arquivo `.part`
permanece no cache: remova apenas esse arquivo parcial antes de repetir.

A saída padrão é `dados\geografia_lote` e não é sobrescrita. Escolha um novo
destino com `--saida dados\outra_coleta` para executar novamente. Se ocorrer
uma falha durante a extração, o destino pode conter uma coleta incompleta;
nesse caso, use outro destino.

`textos.txt` contém os textos em UTF-8, separados por duas quebras de linha.
`artigos.jsonl` registra o título, a URL e o deslocamento e tamanho em bytes
de cada texto dentro desse arquivo, um artigo por linha. `relatorio.json`
soma apenas os bytes dos artigos, excluindo os separadores e os metadados,
e compara o total com 10.000.000.000 bytes. Os dados mantêm a origem para
atribuição, mas é necessário consultar as condições de uso e licença da Wikipédia.
O relatório também mede os bytes do campo `text` de todos os artigos da versão
portuguesa do dataset, inclusive os que não foram selecionados, para comparar
o teto dessa fonte com a meta.

Na coleta verificada em `dados\geografia_lote_verificada`, o recorte de
profundidade 2 contém 68.642 artigos e 52.028.195 bytes de texto. O conjunto
português inteiro de 2023 contém 2.650.621.405 bytes no campo `text`.
Portanto, essa versão do dataset, mesmo sem filtro de tema, não fornece
10 GB de texto. Isso não determina o tamanho da Wikipédia atual nem de outras
fontes. A pasta `dados\geografia_lote` guarda a primeira medição, anterior ao
filtro de namespace e à medição do dataset completo; prefira a verificada.

Para executar todos os testes de coleta:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m unittest discover -s dados -p "test_coletar*.py" -v
```

## Complemento em inglês

`coletar_ingles.py` usa os dumps `enwiki` de 01/11/2026 e os textos
`wikimedia/wikipedia`, `20231101.en`, de 01/11/2023. A combinação de datas
tem a mesma limitação descrita para o português. Por padrão, considera a
categoria `Geography` e duas camadas de subcategorias. Profundidades maiores
podem incluir muitos artigos que não são pertinentes a geografia.

Esse processo baixa aproximadamente 18 GB de arquivos compactados e precisa
de espaço adicional para os índices SQLite e textos extraídos. A indexação é
feita em disco para reduzir o consumo de memória. Pode demorar bastante.

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" dados\coletar_ingles.py --contato "seu-email@exemplo.com" --profundidade 2
```

O cache fica em `dados\cache_ingles`. O programa verifica o tamanho e o checksum
dos arquivos; se um download for interrompido, pode ser retomado com o mesmo
comando, desde que o servidor aceite requisições parciais. Se a construção do
índice for interrompida, o arquivo `.sqlite.part` permanece no cache; remova
apenas esse índice incompleto antes de tentar novamente.

A saída fica em `dados\geografia_ingles` e não é sobrescrita. Além dos textos,
metadados e relatório individuais, o arquivo `resumo_bilingue.json` soma o texto
dessa coleta em inglês ao da coleta verificada em português. Versões do mesmo
artigo em línguas diferentes são consideradas documentos distintos. A meta
somente estará comprovada se o campo `atingiu_meta` for verdadeiro.
