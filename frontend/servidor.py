#Interface web de chat para o sistema RAG (busca/rag.py).

import argparse
import json
import logging
import sys
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
LOG = logging.getLogger(__name__)


class Pergunta(BaseModel):
    pergunta: str = Field(min_length=1, max_length=2000)


def linha(evento):
    return json.dumps(evento, ensure_ascii=False) + "\n"


def criar_app(sistema):
    app = FastAPI(title="RAG Project Gutenberg")
    trava = threading.Lock()  # um único LLM na CPU: gerações em paralelo só disputariam os núcleos

    @app.get("/")
    def pagina():
        return FileResponse(STATIC / "index.html")

    @app.get("/info")
    def info():
        indice = sistema.indice_lexical
        return {"modo": "completo" if indice is not None else "amostra",
                "livros": indice.total_livros if indice is not None else None,
                "passagens": None if indice is not None else len(sistema.passagens),
                "k": sistema.k, "llm": sistema.lm is not None}

    @app.post("/perguntar")
    def perguntar(p: Pergunta):
        return StreamingResponse(eventos(sistema, trava, p.pergunta.strip()), media_type="application/x-ndjson")

    return app


def eventos(sistema, trava, pergunta):
    yield linha({"tipo": "status", "mensagem": "Consultando as fontes e preparando a resposta…"})
    try:
        with trava:
            resultado = sistema.perguntar(pergunta)
        yield linha({"tipo": "fontes", "trechos": resultado["trechos"], "aviso": resultado.get("aviso")})
        if sistema.lm is not None:
            yield linha({"tipo": "resposta", "texto": resultado["resposta"]})
        yield linha({"tipo": "fim", "citados": resultado["citados"],
                     "tempo_recuperacao": resultado["tempo_recuperacao"],
                     "tempo_geracao": resultado["tempo_geracao"]})
    except Exception as e:  # erros do processamento são comunicados ao navegador
        LOG.exception("Falha ao processar a pergunta.")
        yield linha({"tipo": "erro", "mensagem": str(e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="passagens indexadas")
    ap.add_argument("--k", type=int, default=3, help="trechos recuperados por pergunta")
    ap.add_argument("--min-sim", type=float, default=0.4, help="similaridade cosseno mínima")
    ap.add_argument("--sem-llm", action="store_true", help="só mostra as fontes (não carrega o LLM)")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--porta", type=int, default=8000)

    sys.path.insert(0, str(RAIZ / "busca"))
    import uvicorn
    from indice_lexical import INDICE
    from rag import LLM, SistemaRAG

    ap.add_argument("--indice-completo", type=Path, nargs="?", const=INDICE,
                    help="usa o índice lexical completo; opcionalmente informe seu caminho")
    a = ap.parse_args()

    if a.k < 1:
        ap.error("--k deve ser maior que zero")
    sistema = SistemaRAG(n=a.n, llm=None if a.sem_llm else LLM, k=a.k, max_tokens=a.max_tokens,
                         min_sim=a.min_sim, indice_completo=a.indice_completo)
    print(f"Pronto: http://localhost:{a.porta}")
    try:
        uvicorn.run(criar_app(sistema), host="127.0.0.1", port=a.porta)
    finally:
        if sistema.indice_lexical is not None:
            sistema.indice_lexical.close()


if __name__ == "__main__":
    main()
