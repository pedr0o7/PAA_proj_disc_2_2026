import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API = "https://pt.wikipedia.org/w/api.php"
CATEGORIA = "Categoria:Geografia"
META_BYTES = 10_000_000_000


class Wikipedia:
    def __init__(self, contato):
        self.user_agent = f"ProjetoGeografia/1.0 ({contato})"
        self.ultima_requisicao = 0.0

    def consultar(self, parametros):
        espera = 1.0 - (time.monotonic() - self.ultima_requisicao)
        if espera > 0:
            time.sleep(espera)
        parametros = {"format": "json", "formatversion": 2, "maxlag": 5, **parametros}
        requisicao = Request(
            API + "?" + urlencode(parametros),
            headers={"User-Agent": self.user_agent},
        )
        self.ultima_requisicao = time.monotonic()
        with urlopen(requisicao, timeout=60) as resposta:
            dados = json.load(resposta)
        if "error" in dados:
            raise RuntimeError(f"Erro da API: {dados['error']}")
        return dados

    def artigos_da_categoria(self):
        parametros = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": CATEGORIA,
            "cmtype": "page",
            "cmnamespace": 0,
            "cmlimit": 500,
        }
        artigos = {}
        while True:
            resposta = self.consultar(parametros)
            for artigo in resposta["query"]["categorymembers"]:
                artigos[artigo["pageid"]] = artigo["title"]
            if "continue" not in resposta:
                return artigos
            parametros.update(resposta["continue"])

    def extrair_artigo(self, pageid):
        resposta = self.consultar({
            "action": "query",
            "prop": "extracts|info|revisions",
            "pageids": pageid,
            "explaintext": 1,
            "inprop": "url",
            "rvprop": "ids|timestamp",
        })
        pagina = resposta["query"]["pages"][0]
        if "missing" in pagina or not pagina.get("extract"):
            raise ValueError(f"Artigo {pageid} ausente ou sem texto.")
        if pagina["pageid"] != pageid:
            raise ValueError(f"Identificador inesperado para o artigo {pageid}.")
        return pagina


def coletar(wikipedia, pasta):
    if pasta.exists():
        raise FileExistsError(f"A pasta de saída já existe: {pasta}")
    artigos = wikipedia.artigos_da_categoria()
    if not artigos:
        raise ValueError(f"Nenhum artigo encontrado em {CATEGORIA}.")

    pasta.mkdir(parents=True, exist_ok=False)
    pasta_textos = pasta / "textos"
    pasta_textos.mkdir()
    registros = []
    total = 0

    for indice, pageid in enumerate(artigos, start=1):
        pagina = wikipedia.extrair_artigo(pageid)
        texto = pagina["extract"].encode("utf-8")
        caminho = pasta_textos / f"{pageid}.txt"
        caminho.write_bytes(texto)
        total += len(texto)
        revisao = pagina["revisions"][0]
        registros.append({
            "pageid": pageid,
            "titulo": pagina["title"],
            "url": pagina["fullurl"],
            "revisao_id": revisao["revid"],
            "revisao_data": revisao["timestamp"],
            "arquivo": f"textos/{pageid}.txt",
            "bytes_texto": len(texto),
        })
        print(f"[{indice}/{len(artigos)}] {pagina['title']}: {len(texto)} bytes")

    relatorio = {
        "fonte": "Wikipédia em português",
        "categoria": CATEGORIA,
        "profundidade_subcategorias": 0,
        "coletado_em_utc": datetime.now(timezone.utc).isoformat(),
        "quantidade_artigos": len(registros),
        "total_bytes_texto": total,
        "meta_bytes": META_BYTES,
        "atingiu_meta": total >= META_BYTES,
        "artigos": registros,
    }
    (pasta / "relatorio.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return relatorio


def main():
    parser = argparse.ArgumentParser(
        description="Coleta artigos diretamente classificados na categoria Geografia."
    )
    parser.add_argument("--contato", required=True, help="Contato para o User-Agent da Wikipédia.")
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path(__file__).resolve().parent / "geografia_coleta",
        help="Nova pasta para textos e relatório; não pode já existir.",
    )
    args = parser.parse_args()
    if not args.contato.strip():
        parser.error("--contato não pode estar vazio.")

    relatorio = coletar(Wikipedia(args.contato), args.saida)
    print(f"\nArtigos únicos: {relatorio['quantidade_artigos']}")
    print(f"Total de texto: {relatorio['total_bytes_texto']:,} bytes")
    print(f"Meta de 10 GB: {'atingida' if relatorio['atingiu_meta'] else 'não atingida'}")
    print(f"Relatório: {args.saida / 'relatorio.json'}")


if __name__ == "__main__":
    main()
