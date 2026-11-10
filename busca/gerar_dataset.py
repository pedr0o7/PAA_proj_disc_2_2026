"""Gera o conjunto de avaliação (pergunta, resposta de referência, artigo esperado) a partir de passagens indexadas.

Para cada passagem sorteada o LLM local propõe uma pergunta factual e a resposta curta; a passagem de
origem é o contexto de referência. Revise avaliacao_dataset.json manualmente se quiser maior rigor.
Uso: gerar_dataset.py [--q 30]
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

from rag import SistemaRAG

sys.stdout.reconfigure(encoding="utf-8")

SAIDA = Path(__file__).parent / "avaliacao_dataset.json"
PROMPT = (
    "Read the passage and write ONE specific factual question that can be answered ONLY from it, and the "
    "short answer. The question must name its subject explicitly (never use 'this', 'he', 'it' without "
    "a name) so it makes sense without the passage. Output exactly two lines:\nQuestion: ...\nAnswer: ..."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=30)
    ap.add_argument("--semente", type=int, default=7)
    ap.add_argument("--n", type=int, default=5000)
    a = ap.parse_args()

    sistema = SistemaRAG(n=a.n, k=1)
    rng = random.Random(a.semente)
    itens = []
    for idx in rng.sample(range(len(sistema.passagens)), a.q * 3):
        p = sistema.passagens[idx]
        saida = sistema.completar([{"role": "system", "content": PROMPT},
                                   {"role": "user", "content": f"Title: {p['titulo']}\nPassage: {p['texto']}"}], 80)
        m = re.search(r"Question:\s*(.+?)\s*\n+\s*Answer:\s*(.+)", saida, re.S)
        pergunta = m.group(1).strip() if m else ""
        if not m or "?" not in pergunta or re.search(r"\b(passage|mentioned|the text|this|the person|current)\b", pergunta, re.I):
            continue
        itens.append({"pergunta": pergunta, "referencia": m.group(2).strip().split("\n")[0],
                      "titulo_esperado": p["titulo"], "url_esperada": p["url"], "contexto_referencia": p["texto"]})
        print(f"{len(itens)}/{a.q} {itens[-1]['pergunta']}")
        if len(itens) >= a.q:
            break
    SAIDA.write_text(json.dumps(itens, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Salvo em {SAIDA}")


if __name__ == "__main__":
    main()
