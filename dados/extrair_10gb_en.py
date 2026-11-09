"""Extrai texto da Wikipédia EN (parquets do cache) até somar META bytes UTF-8."""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from coletar_ingles import BASE_HF, baixar_continuavel, manifesto

META = 10_000_000_000
CACHE = Path(__file__).parent / "cache_ingles"
SAIDA = Path(__file__).parent / "wikipedia_en_10gb"


def extrair(contato, cache=CACHE, saida=SAIDA, meta=META):
    saida.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)
    disponiveis = manifesto(contato)
    nomes = sorted(disponiveis)
    total = artigos = 0
    ids = set()
    with open(saida / "textos.txt", "wb") as out:
        for nome in nomes:
            caminho, sha256, tamanho = disponiveis[nome]
            arq = cache / nome
            baixar_continuavel(BASE_HF + caminho, arq, "sha256", sha256, tamanho, contato)
            for lote in pq.ParquetFile(arq).iter_batches(batch_size=2000, columns=["id", "text"]):
                for id_, texto in zip(lote.column("id").to_pylist(), lote.column("text").to_pylist()):
                    if id_ in ids or not texto:
                        continue
                    dados = texto.encode("utf-8")
                    ids.add(id_)
                    out.write(dados + b"\n\n")
                    total += len(dados)
                    artigos += 1
                    if total >= meta:
                        break
                else:
                    continue
                break
            else:
                print(f"{arq.name}: {artigos} artigos, {total} bytes", flush=True)
                continue
            break
    relatorio = {
        "artigos_unicos": artigos,
        "bytes_texto": total,
        "meta": meta,
        "atingiu_meta": total >= meta,
    }
    (saida / "relatorio.json").write_text(json.dumps(relatorio, indent=2), encoding="utf-8")
    print(relatorio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contato", required=True, help="Contato para o User-Agent.")
    extrair(parser.parse_args().contato)
