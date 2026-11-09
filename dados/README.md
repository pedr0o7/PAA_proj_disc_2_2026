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
