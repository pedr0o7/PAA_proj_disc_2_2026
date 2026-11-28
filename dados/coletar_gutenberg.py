"""Coleta livros em inglês do Project Gutenberg (dataset sedthh/gutenberg_english) até somar META bytes de texto.

Saída (dados/gutenberg_10gb):
  textos.txt     livros em UTF-8, cada um seguido de uma linha em branco
  livros.jsonl   um livro por linha: id, titulo, autores, url, deslocamento e bytes dentro de textos.txt
  relatorio.json totais e se a meta foi atingida
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

import pyarrow.parquet as pq

RAIZ = Path(__file__).resolve().parent
REPOSITORIO = "sedthh/gutenberg_english"
BASE_HF = f"https://huggingface.co/datasets/{REPOSITORIO}/resolve/main/"
API_ARQUIVOS = f"https://huggingface.co/api/datasets/{REPOSITORIO}/tree/main/data?limit=100"
URL_LIVRO = "https://www.gutenberg.org/ebooks/{}"
META = 10_000_000_000
CACHE = RAIZ / "cache_gutenberg"
SAIDA = RAIZ / "gutenberg_10gb"
ARQUIVO = re.compile(r"train-\d{5}-of-\d{5}-[0-9a-f]+\.parquet")


def agente(contato):
    return {"User-Agent": f"ProjetoPAA-Gutenberg/1.0 ({contato})"}


def manifesto(contato):
    """Retorna {nome: (caminho, sha256, tamanho)} dos parquets do dataset, em ordem de nome."""
    with urlopen(Request(API_ARQUIVOS, headers=agente(contato)), timeout=60) as resposta:
        itens = json.load(resposta)
    arquivos = {}
    for item in itens:
        nome = Path(item["path"]).name
        if item["type"] == "file" and ARQUIVO.fullmatch(nome):
            arquivos[nome] = (item["path"], item["lfs"]["oid"], item["size"])
    if not arquivos:
        raise ValueError("Nenhum parquet encontrado no dataset.")
    return dict(sorted(arquivos.items()))


def calcular_sha256(caminho):
    """Retorna o objeto hashlib com o SHA-256 do arquivo, lido em blocos de 1 MiB."""
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest


def _verificar_existente(destino, tamanho, esperado):
    if destino.stat().st_size != tamanho or calcular_sha256(destino).hexdigest() != esperado:
        raise ValueError(f"Arquivo existente inválido: {destino}")
    print(f"Verificado: {destino.name}", flush=True)


def _gravar_fluxo(resposta, parcial, digest, existente, tamanho):
    """Anexa o corpo da resposta a `parcial` e atualiza `digest`; recusa exceder `tamanho`."""
    recebido = existente
    with parcial.open("ab" if existente else "xb") as arquivo:
        for bloco in iter(lambda: resposta.read(1024 * 1024), b""):
            arquivo.write(bloco)
            digest.update(bloco)
            recebido += len(bloco)
            if recebido > tamanho:
                raise ValueError(f"Download excedeu o tamanho esperado: {parcial}")


def _baixar_restante(url, destino, parcial, existente, tamanho, contato, digest):
    cabecalhos = agente(contato)
    if existente:
        cabecalhos["Range"] = f"bytes={existente}-"
    print(f"Baixando {destino.name}: {existente:,} de {tamanho:,} bytes", flush=True)
    with urlopen(Request(url, headers=cabecalhos), timeout=180) as resposta:
        if existente and resposta.status != 206:
            raise RuntimeError(f"Servidor não aceitou continuar o download de {destino.name}.")
        _gravar_fluxo(resposta, parcial, digest, existente, tamanho)


def baixar_continuavel(url, destino, esperado, tamanho, contato):
    """Baixa com retomada (.part) e valida tamanho e SHA-256."""
    if destino.exists():
        _verificar_existente(destino, tamanho, esperado)
        return
    parcial = destino.with_name(destino.name + ".part")
    existente = parcial.stat().st_size if parcial.exists() else 0
    if existente > tamanho:
        raise ValueError(f"Arquivo parcial maior que o esperado: {parcial}")
    digest = calcular_sha256(parcial) if existente else hashlib.sha256()
    if existente < tamanho:
        _baixar_restante(url, destino, parcial, existente, tamanho, contato, digest)
    if parcial.stat().st_size != tamanho or digest.hexdigest() != esperado:
        raise ValueError(f"Download incompleto ou checksum incorreto: {parcial}")
    parcial.rename(destino)
    print(f"Download validado: {destino.name}", flush=True)


def ler_metadados(bruto):
    """Converte o campo METADATA (JSON serializado) em (id, titulo, autores); None se inválido."""
    try:
        meta = json.loads(bruto)
        id_ = int(meta["text_id"])
    except (TypeError, ValueError, KeyError):
        return None
    titulo = " ".join(str(meta.get("title") or "").split())
    autores = " ".join(str(meta.get("authors") or "").split())
    return id_, titulo or f"Gutenberg #{id_}", autores


def livros_dos_arquivos(arquivos):
    """Gera (texto, metadados) de cada livro, lendo os parquets em lotes pequenos (os livros são grandes)."""
    for arq in arquivos:
        for lote in pq.ParquetFile(arq).iter_batches(batch_size=32, columns=["TEXT", "METADATA"]):
            yield from zip(lote.column("TEXT").to_pylist(), lote.column("METADATA").to_pylist())
        print(f"Lido: {Path(arq).name}", flush=True)


def normalizar(texto):
    """Quebras de linha Unix, sem espaços no fim das linhas e com no máximo duas linhas em branco seguidas.

    A diferença entre uma e duas linhas em branco marca os parágrafos em parte dos livros, por isso é preservada.
    """
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    # Remover espaços finais linha a linha evita a regex `[ \t]+\n`, que tem backtracking quadrático.
    texto = "\n".join(linha.rstrip(" \t") for linha in texto.split("\n"))
    return re.sub(r"\n{4,}", "\n\n\n", texto).strip()


def extrair(livros, saida, meta=META):
    """Grava textos.txt, livros.jsonl e relatorio.json até somar `meta` bytes (o último livro é gravado inteiro)."""
    saida.mkdir(parents=True, exist_ok=True)
    ids, total, posicao = set(), 0, 0
    with open(saida / "textos.txt", "wb") as out, open(saida / "livros.jsonl", "w", encoding="utf-8") as indice:
        for texto, bruto in livros:
            dados_meta = ler_metadados(bruto)
            texto = normalizar(texto or "")
            if dados_meta is None or not texto or dados_meta[0] in ids:
                continue
            id_, titulo, autores = dados_meta
            dados = texto.encode("utf-8")
            ids.add(id_)
            out.write(dados + b"\n\n")
            indice.write(json.dumps({"id": id_, "titulo": titulo, "autores": autores, "url": URL_LIVRO.format(id_),
                                     "deslocamento": posicao, "bytes": len(dados)}, ensure_ascii=False) + "\n")
            posicao += len(dados) + 2
            total += len(dados)
            if total >= meta:
                break
    relatorio = {"fonte": REPOSITORIO, "livros_unicos": len(ids), "bytes_texto": total, "meta": meta,
                 "atingiu_meta": total >= meta}
    (saida / "relatorio.json").write_text(json.dumps(relatorio, indent=2), encoding="utf-8")
    return relatorio


def coletar(contato, cache=CACHE, saida=SAIDA, meta=META, limpar_parquets=False):
    cache.mkdir(parents=True, exist_ok=True)
    disponiveis = manifesto(contato)

    def arquivos():
        for nome, (caminho, sha256, tamanho) in disponiveis.items():
            destino = cache / nome
            baixar_continuavel(BASE_HF + caminho, destino, sha256, tamanho, contato)
            yield destino

    relatorio = extrair(livros_dos_arquivos(arquivos()), saida, meta)
    if limpar_parquets and relatorio.get("atingiu_meta"):
        for padrao in ("*.parquet", "*.parquet.part"):
            for arq in cache.glob(padrao):
                arq.unlink(missing_ok=True)
    return relatorio


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--contato", required=True, help="Contato para o User-Agent.")
    ap.add_argument("--meta", type=int, default=META, help="Bytes de texto a coletar (padrão: 10 GB decimais).")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--saida", type=Path, default=SAIDA)
    ap.add_argument("--limpar-parquets", action="store_true",
                        help="remove os parquets em --cache após concluir a extração")
    a = ap.parse_args()
    print(coletar(a.contato, a.cache, a.saida, a.meta, limpar_parquets=a.limpar_parquets))
