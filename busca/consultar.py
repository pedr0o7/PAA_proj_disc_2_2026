"""RAG interativo: busca semântica em passagens da Wikipédia + resposta de um LLM local com citação das fontes.

Uso: consultar.py [--n 5000] [--k 3] [--llm HF_ID] [--sem-llm] [-p "pergunta"]
"""
import argparse
import sys
import textwrap

from rag import LLM, SistemaRAG

sys.stdout.reconfigure(encoding="utf-8")


def imprimir(r, mostrar_trechos):
    def bloco(texto):
        return textwrap.fill(texto, 100, initial_indent="    ", subsequent_indent="    ")

    if r["resposta"]:
        print("\nResposta:")
        print(bloco(r["resposta"]))
    print("\nFontes:")
    for i, t in enumerate(r["trechos"], 1):
        marca = "*" if i in r["citados"] else " "
        print(f" {marca}[{i}] {t['titulo']} - {t['url']} (similaridade {t['similaridade']:.3f})")
        if mostrar_trechos or not r["resposta"]:
            print(bloco(t["texto"]))
    print(f"\nTempo: recuperação {r['tempo_recuperacao']:.2f}s | geração {r['tempo_geracao']:.1f}s"
          f" | total {r['tempo_total']:.1f}s   (* = citada na resposta)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="passagens indexadas")
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--shards", type=int, default=1, help="parquets da Wikipédia usados como fonte")
    ap.add_argument("--k", type=int, default=3, help="trechos recuperados por pergunta")
    ap.add_argument("-p", "--pergunta", help="pergunta única (sem modo interativo)")
    ap.add_argument("--llm", default=LLM, help="modelo local de geração (HF id)")
    ap.add_argument("--sem-llm", action="store_true", help="só recupera trechos, sem gerar resposta")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--trechos", action="store_true", help="mostra o texto dos trechos recuperados")
    a = ap.parse_args()

    sistema = SistemaRAG(a.n, a.semente, a.shards, None if a.sem_llm else a.llm, a.k, a.max_tokens)
    if a.pergunta:
        imprimir(sistema.perguntar(a.pergunta), a.trechos)
        return
    print(f"{len(sistema.passagens)} passagens indexadas. Pergunte em inglês; linha vazia ou 'sair' encerra.")
    while True:
        pergunta = input("\nPergunta> ").strip()
        if pergunta.lower() in ("", "sair", "exit"):
            break
        imprimir(sistema.perguntar(pergunta), a.trechos)


if __name__ == "__main__":
    main()
