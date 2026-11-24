"""Índice lexical persistente de passagens do corpus Gutenberg."""
import argparse
from contextlib import closing
import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path

from corpus import DADOS, ler_livros

INDICE = Path(__file__).parent / "cache_indice" / "gutenberg_lexical_v1.sqlite"
VERSAO = 1
TAMANHO = 1000
SOBREPOSICAO = 150
PALAVRAS_COMUNS = frozenset(
    "a an and are as at be by did do does for from how in is it of on or the to was what when "
    "where which who whom why with".split()
)


def assinatura(dados):
    textos = dados / "textos.txt"
    livros = dados / "livros.jsonl"
    digest = hashlib.sha256()
    with livros.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    estatistica = textos.stat()
    return json.dumps({
        "versao": VERSAO, "livros_sha256": digest.hexdigest(),
        "textos_bytes": estatistica.st_size, "textos_mtime_ns": estatistica.st_mtime_ns,
        "tamanho": TAMANHO, "sobreposicao": SOBREPOSICAO,
    }, sort_keys=True)


def segmentar(texto, tamanho=TAMANHO, sobreposicao=SOBREPOSICAO):
    """Divide o texto completo em janelas sobrepostas, preservando a ordem dos caracteres."""
    if tamanho <= 0 or not 0 <= sobreposicao < tamanho:
        raise ValueError("Tamanho e sobreposição inválidos.")
    inicio = 0
    while inicio < len(texto):
        fim = min(inicio + tamanho, len(texto))
        if fim < len(texto):
            espaco = texto.rfind(" ", inicio + tamanho // 2, fim)
            quebra = texto.rfind("\n", inicio + tamanho // 2, fim)
            limite = max(espaco, quebra)
            if limite > inicio + sobreposicao:
                fim = limite
        trecho = texto[inicio:fim]
        if trecho.strip():
            yield inicio, trecho
        if fim == len(texto):
            break
        inicio = fim - sobreposicao


def construir(dados=DADOS, caminho=INDICE):
    """Constrói ou retoma o índice; cada livro e seu checkpoint são uma transação."""
    dados, caminho = Path(dados), Path(caminho)
    identidade = assinatura(dados)
    livros = ler_livros(dados)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    inicio = time.perf_counter()
    with closing(sqlite3.connect(caminho)) as con:
        con.execute("CREATE TABLE IF NOT EXISTS estado (id INTEGER PRIMARY KEY CHECK (id = 1), "
                    "assinatura TEXT NOT NULL, proximo INTEGER NOT NULL, completo INTEGER NOT NULL)")
        con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS passagens USING fts5("
                    "titulo, autores, texto, livro_id UNINDEXED, url UNINDEXED, inicio UNINDEXED)")
        estado = con.execute("SELECT assinatura, proximo, completo FROM estado WHERE id = 1").fetchone()
        if estado is None:
            con.execute("INSERT INTO estado VALUES (1, ?, 0, 0)", (identidade,))
            con.commit()
            proximo, completo = 0, 0
        else:
            anterior, proximo, completo = estado
            if anterior != identidade:
                raise ValueError("O corpus ou a versão do índice mudou. Escolha um novo caminho para o índice.")
        if completo:
            print(f"Índice já concluído: {caminho}", flush=True)
            return
        print(f"Indexando {len(livros):,} livros; retomando no livro {proximo + 1:,}.", flush=True)
        with (dados / "textos.txt").open("rb") as arquivo:
            for i in range(proximo, len(livros)):
                livro = livros[i]
                arquivo.seek(livro["deslocamento"])
                bruto = arquivo.read(livro["bytes"])
                if len(bruto) != livro["bytes"]:
                    raise ValueError(f"Livro {livro['id']} incompleto em textos.txt.")
                texto = bruto.decode("utf-8")
                with con:
                    con.execute("BEGIN IMMEDIATE")
                    atual = con.execute("SELECT proximo FROM estado WHERE id = 1").fetchone()[0]
                    if atual != i:
                        raise RuntimeError("Outro processo alterou o índice; reinicie a construção.")
                    con.executemany(
                        "INSERT INTO passagens (titulo, autores, texto, livro_id, url, inicio) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        ((livro["titulo"], livro["autores"], trecho, livro["id"], livro["url"], pos)
                         for pos, trecho in segmentar(texto)),
                    )
                    con.execute("UPDATE estado SET proximo = ? WHERE id = 1", (i + 1,))
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1:,}/{len(livros):,} livros; {time.perf_counter() - inicio:.0f}s",
                          flush=True)
        with con:
            con.execute("UPDATE estado SET completo = 1 WHERE id = 1")
        total = con.execute("SELECT count(*) FROM passagens").fetchone()[0]
        print(f"Índice concluído: {len(livros):,} livros, {total:,} passagens; "
              f"{time.perf_counter() - inicio:.0f}s.", flush=True)


class IndiceLexical:
    def __init__(self, caminho=INDICE, dados=DADOS):
        caminho = Path(caminho)
        self.dados = Path(dados)
        self.catalogo = None
        if not caminho.is_file():
            raise FileNotFoundError(f"Índice lexical ausente: {caminho}. Execute python indice_lexical.py.")
        self.trava = threading.RLock()
        self.con = sqlite3.connect(caminho, check_same_thread=False)
        try:
            estado = self.con.execute(
                "SELECT assinatura, completo, proximo FROM estado WHERE id = 1"
            ).fetchone()
            if not estado or not estado[1]:
                raise ValueError("Índice lexical incompleto. Execute python indice_lexical.py para retomá-lo.")
            if estado[0] != assinatura(self.dados):
                raise ValueError("O índice lexical não corresponde ao corpus atual; construa um novo índice.")
            self.total_livros = estado[2]
        except (sqlite3.Error, OSError, ValueError):
            self.con.close()
            raise

    def buscar_titulo(self, titulo):
        with self.trava:
            if self.catalogo is None:
                catalogo = {}
                for livro in ler_livros(self.dados):
                    chave = " ".join(livro["titulo"].casefold().split())
                    catalogo.setdefault(chave, []).append(livro)
                self.catalogo = catalogo
            return self.catalogo.get(" ".join(titulo.casefold().split()), [])

    def buscar_frase(self, frase, limite):
        consulta = 'texto : "' + frase.replace('"', '""') + '"'
        return self._consultar(consulta, limite)

    def _consultar(self, consulta, limite):
        with self.trava:
            linhas = self.con.execute(
                "SELECT titulo, autores, texto, livro_id, url, inicio FROM passagens "
                "WHERE passagens MATCH ? ORDER BY bm25(passagens, 3.0, 2.0, 1.0) LIMIT ?",
                (consulta, limite),
            )
            return [{"titulo": titulo, "autores": autores, "texto": texto, "id": int(id_),
                     "url": url, "inicio": int(inicio)}
                    for titulo, autores, texto, id_, url, inicio in linhas]

    def recuperar(self, pergunta, limite):
        termos = list(dict.fromkeys(
            termo for termo in re.findall(r"\w+", pergunta.lower())
            if termo not in PALAVRAS_COMUNS
        ))[:32]
        if not termos:
            return []
        consulta = " OR ".join(f'"{termo}"' for termo in termos)
        return self._consultar(consulta, limite)

    def close(self):
        with self.trava:
            self.con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cria ou retoma o índice lexical completo do Gutenberg.")
    parser.add_argument("--dados", type=Path, default=DADOS)
    parser.add_argument("--indice", type=Path, default=INDICE)
    args = parser.parse_args()
    construir(args.dados, args.indice)
