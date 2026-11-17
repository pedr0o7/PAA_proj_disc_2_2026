"""Passagens de livros do Project Gutenberg com título/autor/URL de origem (necessários para citar as fontes).

Lê dados/gutenberg_10gb (gerado por dados/coletar_gutenberg.py): livros.jsonl indica onde cada livro está em
textos.txt. Sorteia livros e, de cada um, alguns parágrafos de prosa, mantendo {texto, titulo, autores, url, id}.
"""
import json
import random
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados" / "gutenberg_10gb"
ESPACOS = re.compile(r"\s+")


def parece_prosa(p):
    """Descarta índices, versos soltos, tabelas e linhas em maiúsculas."""
    letras = sum(c.isalpha() for c in p)
    return (letras / len(p) > 0.75 and p[-1] in '.!?"”\'’' and not p.isupper()
            and sum(c.isdigit() for c in p) / len(p) < 0.05)


def paragrafos(texto, min_car, max_car):
    """Parágrafos de prosa com tamanho entre min_car e max_car, com quebras internas trocadas por espaço.

    Em parte dos livros cada linha do HTML vem separada por uma linha em branco e o parágrafo por duas;
    em outros o parágrafo é uma única linha em branco. O estilo é detectado pela proporção entre os dois.
    """
    uma = len(re.findall(r"(?<!\n)\n\n(?!\n)", texto))
    duas = len(re.findall(r"(?<!\n)\n\n\n(?!\n)", texto))
    separador = r"\n{3,}" if duas > 0.03 * (uma + duas) else r"\n{2,}"
    juntos = (ESPACOS.sub(" ", p).strip() for p in re.split(separador, texto))
    return [p for p in juntos if min_car <= len(p) <= max_car and parece_prosa(p)]


def ler_livros(dados=DADOS):
    with open(dados / "livros.jsonl", encoding="utf-8") as f:
        return [json.loads(linha) for linha in f]


def amostrar_passagens(n, semente, dados=DADOS, min_car=300, max_car=1000, por_livro=3):
    """Sorteia livros e, de cada um, até `por_livro` parágrafos; devolve no máximo n passagens."""
    livros = ler_livros(dados)
    random.Random(semente).shuffle(livros)
    rng = random.Random(semente + 1)
    passagens, vistas = [], set()
    with open(dados / "textos.txt", "rb") as f:
        for livro in livros:
            f.seek(livro["deslocamento"])
            texto = f.read(livro["bytes"]).decode("utf-8", errors="ignore")
            candidatos = paragrafos(texto, min_car, max_car)
            for p in rng.sample(candidatos, min(por_livro, len(candidatos))):
                if p not in vistas:
                    vistas.add(p)
                    passagens.append({"texto": p, "titulo": livro["titulo"], "autores": livro["autores"],
                                      "url": livro["url"], "id": livro["id"]})
            if len(passagens) >= n:
                break
    return passagens[:n]


def texto_indexado(p):
    """Texto usado nos embeddings e no BM25: o título e o autor ajudam a achar o livro certo."""
    cabecalho = f"{p['titulo']} by {p['autores']}" if p["autores"] else p["titulo"]
    return f"{cabecalho}. {p['texto']}"


def carregar_passagens(cache_dir, n, semente, dados=DADOS):
    """Retorna lista de dicts {texto, titulo, autores, url, id}; usa cache em disco se existir."""
    arq = cache_dir / f"passagens_gutenberg_{n}_{semente}.json"
    if arq.exists():
        return json.loads(arq.read_text(encoding="utf-8"))
    if not (dados / "livros.jsonl").exists():
        raise FileNotFoundError(
            f"Corpus não encontrado em {dados}. Execute: python dados\\coletar_gutenberg.py --contato SEU-EMAIL")
    passagens = amostrar_passagens(n, semente, dados)
    arq.write_text(json.dumps(passagens, ensure_ascii=False), encoding="utf-8")
    return passagens
