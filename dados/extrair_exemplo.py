import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# Substitua pelo seu contato antes de executar.
contato = "seu-email@exemplo.com"

parametros = {
    "action": "query",
    "prop": "extracts",
    "explaintext": 1,
    "titles": "Geografia",
    "redirects": 1,
    "format": "json",
    "formatversion": 2,
}

url_api = "https://pt.wikipedia.org/w/api.php?" + urlencode(parametros)

requisicao = Request(
    url_api,
    headers={
        "User-Agent": f"ProjetoGeografia/1.0 ({contato})",
    },
)

with urlopen(requisicao, timeout=30) as resposta:
    dados = json.load(resposta)

pagina = dados["query"]["pages"][0]

if "missing" in pagina or not pagina.get("extract"):
    raise ValueError("A página não foi encontrada ou não possui texto.")

texto = pagina["extract"]

pasta = Path("dados")
pasta.mkdir(exist_ok=True)

arquivo_texto = pasta / "geografia.txt"
arquivo_texto.write_text(texto, encoding="utf-8")

metadados = {
    "titulo": pagina["title"],
    "pageid": pagina["pageid"],
    "url": "https://pt.wikipedia.org/wiki/Geografia",
    "fonte": "Wikipédia em português",
}

arquivo_metadados = pasta / "geografia_metadados.json"
arquivo_metadados.write_text(
    json.dumps(metadados, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Artigo: {pagina['title']}")
print(f"Arquivo salvo em: {arquivo_texto}")
print(f"Tamanho do texto: {len(texto.encode('utf-8'))} bytes")