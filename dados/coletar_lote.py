import argparse
import gzip
import hashlib
import json
import re
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pyarrow.compute as compute
import pyarrow.parquet as parquet


DATA_DUMP = "20261101"
REVISAO_HF = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
META = 10_000_000_000
PASTA_PADRAO = Path(__file__).resolve().parent
ARQUIVOS_DUMP = {
    "page": "52b1dcb57f7b548a7b430e9f3c27f613cc99fa0f",
    "linktarget": "543e1322c815de0d1fa8ff47ca0d60bd2f30260a",
    "categorylinks": "3078ce4b5543e683e14d2cd8f82fcf4954716ef1",
}
ARQUIVOS_HF = [
    "059136d457150f9c033a4bdfb7fbea2eef2b6e58e03e2c0cb8275144ce99fb34",
    "5e9b4476c0d69b0bffdcf76529de424dfda9510e3a6867676e5d482e6c70f1be",
    "d136dc9359e96408ed739bedc3c7da0bb4ee4481597299eb5d4170b36d1fca9c",
    "786f1ee2e400298e4b0889de8eef4d8bc465629cf773970a84adb520ae900e4f",
    "181d7004d5cedf4704ec5baa5c1c549520634234e3bdf7ea9aa8cfeb0f505cbc",
    "5d60245e7e8c8176e2b9f7f0ed5ba5dcf4eb748bb94f852582ccac01d7ec72fd",
]
REGISTRO_CATEGORIA = re.compile(rb"^\((\d+),14,'((?:\\.|[^'\\])*)'")
REGISTRO_PAGINA = re.compile(rb"^\((\d+),(-?\d+),")
REGISTRO_RELACAO = re.compile(rb"^\((\d+),.*,'(page|subcat|file)',\d+,(\d+)\)[,;]\s*$")


def hash_arquivo(caminho, algoritmo):
    digest = hashlib.new(algoritmo)
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def baixar(url, destino, algoritmo, esperado, contato):
    if destino.exists():
        if hash_arquivo(destino, algoritmo) != esperado:
            raise ValueError(f"Checksum incorreto para {destino}; não foi sobrescrito.")
        print(f"Arquivo já verificado: {destino.name}", flush=True)
        return
    parcial = destino.with_name(destino.name + ".part")
    if parcial.exists():
        raise FileExistsError(f"Download parcial existente: {parcial}. Remova-o antes de repetir.")
    requisicao = Request(url, headers={"User-Agent": f"ProjetoGeografia/1.0 ({contato})"})
    print(f"Baixando {destino.name}...", flush=True)
    digest = hashlib.new(algoritmo)
    with urlopen(requisicao, timeout=120) as resposta, parcial.open("xb") as arquivo:
        while True:
            bloco = resposta.read(1024 * 1024)
            if not bloco:
                break
            arquivo.write(bloco)
            digest.update(bloco)
    if digest.hexdigest() != esperado:
        raise ValueError(f"Checksum incorreto para {parcial}; download incompleto ou alterado.")
    parcial.rename(destino)
    print(f"Download validado: {destino.name}", flush=True)


def ler_categorias(caminho):
    resultado = {}
    with gzip.open(caminho, "rb") as arquivo:
        for linha in arquivo:
            resultado_linha = REGISTRO_CATEGORIA.match(linha)
            if resultado_linha:
                resultado[int(resultado_linha.group(1))] = resultado_linha.group(2)
    if not resultado:
        raise ValueError(f"Nenhuma categoria encontrada em {caminho}; formato inesperado.")
    return resultado


def ler_paginas(caminho):
    categorias = {}
    artigos = set()
    with gzip.open(caminho, "rb") as arquivo:
        for linha in arquivo:
            if not linha.startswith(b"("):
                continue
            pagina = REGISTRO_PAGINA.match(linha)
            if not pagina:
                raise ValueError(f"Formato inesperado em {caminho}: {linha[:100]!r}")
            identificador = int(pagina.group(1))
            namespace = int(pagina.group(2))
            if namespace == 0:
                artigos.add(identificador)
            elif namespace == 14:
                categoria = REGISTRO_CATEGORIA.match(linha)
                if not categoria:
                    raise ValueError(f"Categoria inválida em {caminho}: {linha[:100]!r}")
                categorias[identificador] = categoria.group(2)
    if not categorias or not artigos:
        raise ValueError(f"Não foram encontradas páginas e categorias em {caminho}.")
    return categorias, artigos


def ler_relacoes(caminho):
    with gzip.open(caminho, "rb") as arquivo:
        for linha in arquivo:
            if not linha.startswith(b"("):
                continue
            resultado = REGISTRO_RELACAO.match(linha)
            if not resultado:
                raise ValueError(f"Formato inesperado em {caminho}: {linha[:100]!r}")
            yield int(resultado.group(1)), resultado.group(2), int(resultado.group(3))


def categorias_alcancaveis(raiz, filhos, profundidade):
    distancias = {raiz: 0}
    fila = deque([raiz])
    while fila:
        categoria = fila.popleft()
        if distancias[categoria] >= profundidade:
            continue
        for filho in filhos.get(categoria, ()):
            if filho not in distancias:
                distancias[filho] = distancias[categoria] + 1
                fila.append(filho)
    return distancias


def identificar_artigos(arquivos, profundidade):
    paginas, paginas_artigos = ler_paginas(arquivos["page"])
    destinos = ler_categorias(arquivos["linktarget"])
    titulo_para_destino = {titulo: identificador for identificador, titulo in destinos.items()}
    if b"Geografia" not in titulo_para_destino:
        raise ValueError("Categoria Geografia não encontrada no dump de destinos.")

    filhos = defaultdict(set)
    for origem, tipo, destino in ler_relacoes(arquivos["categorylinks"]):
        if tipo == b"subcat" and origem in paginas:
            filho = titulo_para_destino.get(paginas[origem])
            if filho is not None:
                filhos[destino].add(filho)
    categorias = categorias_alcancaveis(titulo_para_destino[b"Geografia"], filhos, profundidade)
    artigos = set()
    for origem, tipo, destino in ler_relacoes(arquivos["categorylinks"]):
        if tipo == b"page" and destino in categorias and origem in paginas_artigos:
            artigos.add(origem)
    if not artigos:
        raise ValueError("Nenhum artigo encontrado no recorte selecionado.")
    return artigos, categorias


def extrair(
    arquivos_parquet, selecionados, saida, profundidade, categorias,
    idioma="pt", fonte_categorias=f"ptwiki-{DATA_DUMP}",
):
    saida.mkdir(parents=True, exist_ok=False)
    encontrados = set()
    bytes_texto = 0
    bytes_separadores = 0
    bytes_dataset = 0
    artigos_dataset = 0
    quantidade = 0
    artigos_sem_texto = 0
    inicio = time.monotonic()
    with (saida / "textos.txt").open("xb") as textos, (
        saida / "artigos.jsonl"
    ).open("x", encoding="utf-8", newline="\n") as metadados:
        for caminho in arquivos_parquet:
            print(f"Lendo {caminho.name}...", flush=True)
            arquivo = parquet.ParquetFile(caminho)
            for lote in arquivo.iter_batches(batch_size=1024, columns=["id", "url", "title", "text"]):
                textos_lote = lote.column(lote.schema.get_field_index("text"))
                if textos_lote.null_count:
                    raise ValueError(f"O dataset contém texto nulo em {caminho}.")
                bytes_dataset += compute.sum(compute.binary_length(textos_lote)).as_py()
                artigos_dataset += lote.num_rows
                for artigo in lote.to_pylist():
                    identificador = int(artigo["id"])
                    if identificador not in selecionados:
                        continue
                    if identificador in encontrados:
                        raise ValueError(f"ID de artigo duplicado no dataset: {identificador}")
                    encontrados.add(identificador)
                    if not isinstance(artigo["text"], str):
                        raise ValueError(f"Texto inválido para o artigo {identificador}")
                    if not artigo["text"].strip():
                        artigos_sem_texto += 1
                        continue
                    conteudo = artigo["text"].encode("utf-8")
                    offset = textos.tell()
                    textos.write(conteudo)
                    textos.write(b"\n\n")
                    bytes_texto += len(conteudo)
                    bytes_separadores += 2
                    quantidade += 1
                    registro = {
                        "pageid": identificador,
                        "titulo": artigo["title"],
                        "url": artigo["url"],
                        "offset_bytes": offset,
                        "bytes_texto": len(conteudo),
                    }
                    metadados.write(json.dumps(registro, ensure_ascii=False) + "\n")
            print(f"  {quantidade:,} artigos; {bytes_texto:,} bytes de texto", flush=True)

    relatorio = {
        "fonte_textos": f"wikimedia/wikipedia, 20231101.{idioma}",
        "revisao_dataset": REVISAO_HF,
        "fonte_categorias": fonte_categorias,
        "observacao": "Categorias de 2026 aplicadas a textos de 2023; correspondências podem faltar.",
        "categoria_raiz": "Geografia" if idioma == "pt" else "Geography",
        "profundidade_maxima": profundidade,
        "categorias_por_profundidade": {
            str(profundidade_atual): sum(d == profundidade_atual for d in categorias.values())
            for profundidade_atual in range(profundidade + 1)
        },
        "artigos_selecionados_no_dump": len(selecionados),
        "artigos_nao_encontrados_no_dataset": len(selecionados - encontrados),
        "artigos_sem_texto": artigos_sem_texto,
        "artigos_extraidos": quantidade,
        "bytes_texto": bytes_texto,
        "artigos_dataset_total": artigos_dataset,
        "bytes_texto_dataset_total": bytes_dataset,
        "bytes_separadores_nao_contados": bytes_separadores,
        "meta_bytes": META,
        "atingiu_meta": bytes_texto >= META,
        "coletado_em_utc": datetime.now(timezone.utc).isoformat(),
        "duracao_segundos": round(time.monotonic() - inicio, 1),
    }
    if (saida / "textos.txt").stat().st_size != bytes_texto + bytes_separadores:
        raise ValueError("Tamanho do arquivo de textos não corresponde ao total calculado.")
    (saida / "relatorio.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return relatorio


def main():
    parser = argparse.ArgumentParser(description="Coleta em lote artigos de geografia.")
    parser.add_argument("--contato", required=True, help="Contato para identificar o programa.")
    parser.add_argument("--profundidade", type=int, default=2)
    parser.add_argument("--cache", type=Path, default=PASTA_PADRAO / "cache_lote")
    parser.add_argument("--saida", type=Path, default=PASTA_PADRAO / "geografia_lote")
    args = parser.parse_args()
    if not args.contato.strip():
        parser.error("--contato não pode estar vazio.")
    if args.profundidade < 0:
        parser.error("--profundidade deve ser zero ou maior.")
    if args.saida.exists():
        parser.error(f"A pasta de saída já existe: {args.saida}")
    args.cache.mkdir(parents=True, exist_ok=True)

    dumps = {}
    for nome, checksum in ARQUIVOS_DUMP.items():
        nome_arquivo = f"ptwiki-{DATA_DUMP}-{nome}.sql.gz"
        caminho = args.cache / nome_arquivo
        baixar(f"https://dumps.wikimedia.org/ptwiki/{DATA_DUMP}/{nome_arquivo}",
               caminho, "sha1", checksum, args.contato)
        dumps[nome] = caminho
    selecionados, categorias = identificar_artigos(dumps, args.profundidade)
    print(f"Categorias: {len(categorias):,}; artigos únicos: {len(selecionados):,}", flush=True)

    arquivos_parquet = []
    for indice, checksum in enumerate(ARQUIVOS_HF):
        nome = f"train-{indice:05d}-of-00006.parquet"
        caminho = args.cache / nome
        url = f"https://huggingface.co/datasets/wikimedia/wikipedia/resolve/{REVISAO_HF}/20231101.pt/{nome}"
        baixar(url, caminho, "sha256", checksum, args.contato)
        arquivos_parquet.append(caminho)
    relatorio = extrair(arquivos_parquet, selecionados, args.saida, args.profundidade, categorias)
    print(json.dumps(relatorio, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
