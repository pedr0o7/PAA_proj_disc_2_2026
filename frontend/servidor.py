#Interface web de chat para o sistema RAG (busca/rag.py).

import argparse
import json
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"


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
        return {"passagens": len(sistema.passagens), "k": sistema.k, "llm": sistema.lm is not None}

    @app.post("/perguntar")
    def perguntar(p: Pergunta):
        return StreamingResponse(eventos(sistema, trava, p.pergunta.strip()), media_type="application/x-ndjson")

    return app


def eventos(sistema, trava, pergunta):
    with trava:
        try:
            t0 = time.perf_counter()
            candidatos = sistema.recuperar(pergunta)
            limiar = getattr(sistema, "min_sim", 0.0)
            trechos = [t for t in candidatos if t.get("similaridade", 1.0) >= limiar]
            t1 = time.perf_counter()
            yield linha({"tipo": "fontes", "trechos": trechos})
            resposta = ""
            if sistema.lm is not None:
                caminho_busca = str(RAIZ / "busca")
                if caminho_busca not in sys.path:
                    sys.path.insert(0, caminho_busca)
                from rag import SEM_CONTEXTO, validar_citacoes

                if trechos:
                    for delta in gerar_em_partes(sistema, pergunta, trechos):
                        resposta += delta
                        yield linha({"tipo": "texto", "delta": delta})
                    resposta, citados = validar_citacoes(resposta, len(trechos))
                else:
                    resposta, citados = SEM_CONTEXTO, []
                yield linha({"tipo": "resposta", "texto": resposta})
            else:
                citados = []
            t2 = time.perf_counter()
            yield linha({"tipo": "fim", "citados": citados, "tempo_recuperacao": t1 - t0, "tempo_geracao": t2 - t1})
        except Exception as e:  # o erro vai para a tela em vez de cortar a resposta sem aviso
            yield linha({"tipo": "erro", "mensagem": str(e)})


def gerar_em_partes(sistema, pergunta, trechos):
    """Mesma geração de SistemaRAG.gerar/completar, mas devolvendo o texto aos poucos."""
    from rag import PROMPT_SISTEMA
    from transformers import TextIteratorStreamer

    # shortcut: repete a montagem do prompt de rag.py; se gerar() mudar lá, atualizar aqui também
    contexto = "\n\n".join(f"[{i}] ({t['titulo']}) {t['texto']}" for i, t in enumerate(trechos, 1))
    msgs = [{"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": f"Context:\n{contexto}\n\nQuestion: {pergunta}"}]
    ids = sistema.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    streamer = TextIteratorStreamer(sistema.tok, skip_prompt=True, skip_special_tokens=True)
    tarefa = threading.Thread(target=sistema.lm.generate, kwargs=dict(
        **ids, max_new_tokens=sistema.max_tokens, do_sample=False, repetition_penalty=1.1, streamer=streamer))
    tarefa.start()
    try:
        yield from streamer
    finally:
        tarefa.join()  # se o navegador desconectar, espera a geração acabar antes de liberar a trava


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="passagens indexadas")
    ap.add_argument("--k", type=int, default=3, help="trechos recuperados por pergunta")
    ap.add_argument("--sem-llm", action="store_true", help="só mostra as fontes (não carrega o LLM)")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--porta", type=int, default=8000)
    a = ap.parse_args()

    sys.path.insert(0, str(RAIZ / "busca"))
    import uvicorn
    from rag import LLM, SistemaRAG

    sistema = SistemaRAG(n=a.n, llm=None if a.sem_llm else LLM, k=a.k, max_tokens=a.max_tokens)
    print(f"Pronto: http://localhost:{a.porta}")
    uvicorn.run(criar_app(sistema), host="127.0.0.1", port=a.porta)


if __name__ == "__main__":
    main()
