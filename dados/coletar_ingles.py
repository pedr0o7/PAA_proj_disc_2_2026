import argparse
import gzip
import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from urllib.request import Request, urlopen

from coletar_lote import (
    REGISTRO_CATEGORIA,
    REGISTRO_PAGINA,
    REVISAO_HF,
    extrair,
    ler_relacoes,
)


RAIZ = Path(__file__).resolve().parent
DATA_DUMP = "20261101"
DUMPS = {
    "page": ("98dc4d3e94ece0319cb83deed84fde646e752f3b", 2426942287),
    "linktarget": ("470830ea4afd2dc0fc5aa2b34403c29ed14aaf9b", 1452254955),
    "categorylinks": ("73bc2f2dc2e4c5518560c5030fde5c1a7e6a5cbe", 2566934222),
}
BASE_HF = f"https://huggingface.co/datasets/wikimedia/wikipedia/resolve/{REVISAO_HF}/"


def manifesto(contato):
    url = f"https://huggingface.co/api/datasets/wikimedia/wikipedia/tree/{REVISAO_HF}/20231101.en?limit=100"
    requisicao = Request(url, headers={"User-Agent": f"ProjetoGeografia/1.0 ({contato})"})
    with urlopen(requisicao, timeout=60) as resposta:
        dados = json.load(resposta)
    if len(dados) != 41:
        raise ValueError(f"Esperados 41 arquivos ingleses, recebidos {len(dados)}.")
    arquivos = {}
    for item in dados:
        nome = Path(item["path"]).name
        if not re.fullmatch(r"train-\d{5}-of-00041\.parquet", nome):
            raise ValueError(f"Arquivo inesperado no dataset: {item['path']}")
        arquivos[nome] = (item["path"], item["lfs"]["oid"], item["size"])
    if len(arquivos) != 41:
        raise ValueError("O manifesto contém nomes duplicados.")
    return arquivos


def baixar_continuavel(url, destino, algoritmo, esperado, tamanho, contato):
    def calcular_hash(caminho):
        digest = hashlib.new(algoritmo)
        with caminho.open("rb") as arquivo:
            for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
                digest.update(bloco)
        return digest

    if destino.exists():
        if destino.stat().st_size != tamanho or calcular_hash(destino).hexdigest() != esperado:
            raise ValueError(f"Arquivo existente inválido: {destino}")
        print(f"Verificado: {destino.name}", flush=True)
        return
    parcial = destino.with_name(destino.name + ".part")
    existente = parcial.stat().st_size if parcial.exists() else 0
    if existente > tamanho:
        raise ValueError(f"Arquivo parcial maior que o esperado: {parcial}")
    digest = calcular_hash(parcial) if existente else hashlib.new(algoritmo)
    if existente < tamanho:
        cabecalhos = {"User-Agent": f"ProjetoGeografia/1.0 ({contato})"}
        if existente:
            cabecalhos["Range"] = f"bytes={existente}-"
        requisicao = Request(url, headers=cabecalhos)
        print(f"Baixando {destino.name}: {existente:,} de {tamanho:,} bytes", flush=True)
        with urlopen(requisicao, timeout=180) as resposta:
            if existente and resposta.status != 206:
                raise RuntimeError(f"Servidor não aceitou continuar o download de {destino.name}.")
            if existente and not resposta.headers.get("Content-Range", "").startswith(f"bytes {existente}-"):
                raise RuntimeError(f"Intervalo incorreto ao continuar o download de {destino.name}.")
            with parcial.open("ab" if existente else "xb") as arquivo:
                recebido = existente
                while True:
                    bloco = resposta.read(1024 * 1024)
                    if not bloco:
                        break
                    arquivo.write(bloco)
                    digest.update(bloco)
                    recebido += len(bloco)
                    if recebido > tamanho:
                        raise ValueError(f"Download excedeu tamanho esperado: {parcial}")
    if parcial.stat().st_size != tamanho or digest.hexdigest() != esperado:
        raise ValueError(f"Download incompleto ou checksum incorreto: {parcial}")
    parcial.rename(destino)
    print(f"Download validado: {destino.name}", flush=True)


def baixar_fontes(cache, contato):
    cache.mkdir(parents=True, exist_ok=True)
    arquivos = manifesto(contato)
    for nome, (sha1, tamanho) in DUMPS.items():
        arquivo = f"enwiki-{DATA_DUMP}-{nome}.sql.gz"
        url = f"https://dumps.wikimedia.org/enwiki/{DATA_DUMP}/{arquivo}"
        baixar_continuavel(url, cache / arquivo, "sha1", sha1, tamanho, contato)
    for nome, (caminho, sha256, tamanho) in sorted(arquivos.items()):
        baixar_continuavel(BASE_HF + caminho, cache / nome, "sha256", sha256, tamanho, contato)
    return arquivos


def inserir_lote(conexao, sql, lote):
    if lote:
        conexao.executemany(sql, lote)
        conexao.commit()
        lote.clear()


def indexar_paginas(conexao, caminho):
    artigos, categorias = [], []
    with gzip.open(caminho, "rb") as arquivo:
        for linha in arquivo:
            if not linha.startswith(b"("):
                continue
            pagina = REGISTRO_PAGINA.match(linha)
            if not pagina:
                raise ValueError(f"Página inválida em {caminho}: {linha[:100]!r}")
            identificador, namespace = int(pagina.group(1)), int(pagina.group(2))
            if namespace == 0:
                artigos.append((identificador,))
            elif namespace == 14:
                categoria = REGISTRO_CATEGORIA.match(linha)
                if not categoria:
                    raise ValueError(f"Categoria inválida em {caminho}: {linha[:100]!r}")
                categorias.append((identificador, categoria.group(2)))
            if len(artigos) >= 50000:
                inserir_lote(conexao, "INSERT INTO paginas VALUES (?)", artigos)
            if len(categorias) >= 50000:
                inserir_lote(conexao, "INSERT INTO categorias VALUES (?, ?)", categorias)
    inserir_lote(conexao, "INSERT INTO paginas VALUES (?)", artigos)
    inserir_lote(conexao, "INSERT INTO categorias VALUES (?, ?)", categorias)


def indexar_destinos(conexao, caminho):
    lote = []
    with gzip.open(caminho, "rb") as arquivo:
        for linha in arquivo:
            categoria = REGISTRO_CATEGORIA.match(linha)
            if categoria:
                lote.append((categoria.group(2), int(categoria.group(1))))
                if len(lote) >= 50000:
                    inserir_lote(conexao, "INSERT INTO destinos VALUES (?, ?)", lote)
    inserir_lote(conexao, "INSERT INTO destinos VALUES (?, ?)", lote)
    if conexao.execute("SELECT COUNT(*) FROM destinos").fetchone()[0] == 0:
        raise ValueError("Não foram encontrados destinos de categorias.")


def construir_indice(cache, profundidade):
    destino = cache / f"categorias_en_profundidade_{profundidade}.sqlite"
    parcial = destino.with_name(destino.name + ".part")
    if destino.exists():
        return destino
    if parcial.exists():
        raise FileExistsError(f"Índice incompleto existente: {parcial}")

    conexao = sqlite3.connect(parcial)
    try:
        conexao.executescript("""
            PRAGMA synchronous=NORMAL;
            PRAGMA temp_store=FILE;
            PRAGMA cache_size=-65536;
            CREATE TABLE paginas (id INTEGER PRIMARY KEY);
            CREATE TABLE categorias (id INTEGER PRIMARY KEY, titulo BLOB NOT NULL);
            CREATE TABLE destinos (titulo BLOB PRIMARY KEY, id INTEGER NOT NULL);
            CREATE TABLE ligacoes (origem INTEGER NOT NULL, destino INTEGER NOT NULL);
        """)
        print("Indexando páginas em SQLite...", flush=True)
        indexar_paginas(conexao, cache / f"enwiki-{DATA_DUMP}-page.sql.gz")
        print("Indexando nomes de categorias...", flush=True)
        indexar_destinos(conexao, cache / f"enwiki-{DATA_DUMP}-linktarget.sql.gz")
        raiz = conexao.execute("SELECT id FROM destinos WHERE titulo = ?", (b"Geography",)).fetchone()
        if raiz is None:
            raise ValueError("Categoria Geography não encontrada.")
        caminho = cache / f"enwiki-{DATA_DUMP}-categorylinks.sql.gz"
        print("Indexando ligações entre categorias...", flush=True)
        lote = []
        for origem, tipo, alvo in ler_relacoes(caminho):
            if tipo == b"subcat":
                lote.append((origem, alvo))
                if len(lote) >= 50000:
                    inserir_lote(conexao, "INSERT INTO ligacoes VALUES (?, ?)", lote)
        inserir_lote(conexao, "INSERT INTO ligacoes VALUES (?, ?)", lote)
        conexao.executescript("""
            CREATE TABLE arestas (pai INTEGER NOT NULL, filho INTEGER NOT NULL);
            INSERT INTO arestas
                SELECT ligacoes.destino, destinos.id
                FROM ligacoes
                JOIN categorias ON categorias.id = ligacoes.origem
                JOIN destinos ON destinos.titulo = categorias.titulo;
            CREATE INDEX arestas_pai ON arestas(pai);
            DROP TABLE ligacoes;
            CREATE TABLE alcancaveis (id INTEGER PRIMARY KEY, profundidade INTEGER NOT NULL);
        """)
        conexao.execute("INSERT INTO alcancaveis VALUES (?, 0)", raiz)
        for nivel in range(profundidade):
            conexao.execute("""
                INSERT OR IGNORE INTO alcancaveis
                SELECT arestas.filho, ?
                FROM alcancaveis JOIN arestas ON arestas.pai = alcancaveis.id
                WHERE alcancaveis.profundidade = ?
            """, (nivel + 1, nivel))
            conexao.commit()
        categorias = {r[0] for r in conexao.execute("SELECT id FROM alcancaveis")}
        print(f"Categorias encontradas: {len(categorias):,}. Selecionando páginas...", flush=True)
        conexao.execute("CREATE TABLE selecionadas (id INTEGER PRIMARY KEY)")
        lote = []
        for origem, tipo, alvo in ler_relacoes(caminho):
            if tipo == b"page" and alvo in categorias:
                lote.append((origem,))
                if len(lote) >= 50000:
                    inserir_lote(conexao, "INSERT OR IGNORE INTO selecionadas VALUES (?)", lote)
        inserir_lote(conexao, "INSERT OR IGNORE INTO selecionadas VALUES (?)", lote)
        conexao.execute("DELETE FROM selecionadas WHERE id NOT IN (SELECT id FROM paginas)")
        conexao.commit()
        print("Índice finalizado.", flush=True)
    finally:
        conexao.close()
    parcial.rename(destino)
    return destino


def coletar_ingles(cache, arquivos, saida, profundidade):
    indice = construir_indice(cache, profundidade)
    with closing(sqlite3.connect(indice)) as conexao:
        selecionados = {r[0] for r in conexao.execute("SELECT id FROM selecionadas")}
        categorias = {r[0]: r[1] for r in conexao.execute("SELECT id, profundidade FROM alcancaveis")}
    if not selecionados:
        raise ValueError("Nenhum artigo em inglês foi encontrado nesse recorte.")
    return extrair(
        [cache / nome for nome in sorted(arquivos)],
        selecionados, saida, profundidade, categorias,
        idioma="en", fonte_categorias=f"enwiki-{DATA_DUMP}",
    )


def main():
    parser = argparse.ArgumentParser(description="Coleta em lote da Wikipédia em inglês.")
    parser.add_argument("--contato", required=True)
    parser.add_argument("--cache", type=Path, default=RAIZ / "cache_ingles")
    parser.add_argument("--etapa", choices=("baixar", "tudo"), default="tudo")
    parser.add_argument("--profundidade", type=int, default=2)
    parser.add_argument("--saida", type=Path, default=RAIZ / "geografia_ingles")
    args = parser.parse_args()
    if not args.contato.strip():
        parser.error("--contato não pode estar vazio.")
    if args.profundidade < 0:
        parser.error("--profundidade deve ser zero ou maior.")
    if args.etapa == "tudo" and args.saida.exists():
        parser.error(f"A pasta de saída já existe: {args.saida}")
    portugues = RAIZ / "geografia_lote_verificada" / "relatorio.json"
    if args.etapa == "tudo" and not portugues.exists():
        parser.error(f"Relatório da coleta em português não encontrado: {portugues}")
    arquivos = baixar_fontes(args.cache, args.contato)
    if args.etapa == "tudo":
        relatorio = coletar_ingles(args.cache, arquivos, args.saida, args.profundidade)
        total_pt = json.loads(portugues.read_text(encoding="utf-8"))["bytes_texto"]
        total = total_pt + relatorio["bytes_texto"]
        resumo = {
            "bytes_texto_portugues": total_pt,
            "bytes_texto_ingles": relatorio["bytes_texto"],
            "bytes_texto_total": total,
            "meta_bytes": 10_000_000_000,
            "atingiu_meta": total >= 10_000_000_000,
            "observacao": "Artigos nas duas línguas são contados como documentos distintos.",
        }
        (args.saida / "resumo_bilingue.json").write_text(
            json.dumps(resumo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(resumo, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
