"""RAG interativo: busca híbrida em passagens de livros do Project Gutenberg + resposta de um LLM local com citação das fontes.

Uso: consultar.py [--n 20000] [--k 3] [--min-sim 0.4] [--llm HF_ID] [--sem-llm] [-p "pergunta"]
"""
import argparse
import sys
import textwrap
from pathlib import Path

from indice_lexical import INDICE
from rag import LLM, SistemaRAG

sys.stdout.reconfigure(encoding="utf-8")


def imprimir(r, mostrar_trechos):
    def bloco(texto):
        return textwrap.fill(texto, 100, initial_indent="    ", subsequent_indent="    ")

    if r["resposta"]:
        print("\nResposta:")
        print(bloco(r["resposta"]))
    print("\nFontes citadas:" if r["citados"] else "\nPassagens recuperadas (não citadas):")
    if r.get("aviso"):
        print(f"  {r['aviso']}")
    elif not r["trechos"]:
        print("  Nenhuma passagem do corpus é relevante o bastante para esta pergunta.")
    elif not r["citados"]:
        print("  O modelo não confirmou suporte em uma passagem; trate estes resultados apenas como candidatos.")
    for t in r.get("descartados", []):
        print(f"  (descartada) {t['titulo']} (similaridade {t['similaridade']:.3f})")
    for i, t in enumerate(r["trechos"], 1):
        marca = "*" if i in r["citados"] else " "
        autor = f" - {t['autores']}" if t["autores"] else ""
        descricao = "metadados do catálogo" if t.get("tipo_fonte") == "metadados" else \
                    f"similaridade {t['similaridade']:.3f}"
        print(f" {marca}[{i}] {t['titulo']}{autor} - {t['url']} ({descricao})")
        if mostrar_trechos or not r["resposta"]:
            print(bloco(t["texto"]))
    print(f"\nTempo: recuperação {r['tempo_recuperacao']:.2f}s | geração {r['tempo_geracao']:.1f}s"
          f" | total {r['tempo_total']:.1f}s   (* = citada na resposta)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="passagens indexadas")
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--k", type=int, default=3, help="trechos recuperados por pergunta")
    ap.add_argument("--min-sim", type=float, default=0.4,
                    help="similaridade (cosseno) mínima para uma passagem ser usada como fonte")
    ap.add_argument("-p", "--pergunta", help="pergunta única (sem modo interativo)")
    ap.add_argument("--llm", default=LLM, help="modelo local de geração (HF id)")
    ap.add_argument("--sem-llm", action="store_true", help="só recupera trechos, sem gerar resposta")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--trechos", action="store_true", help="mostra o texto dos trechos recuperados")
    ap.add_argument("--indice-completo", type=Path, nargs="?", const=INDICE,
                    help="usa o índice lexical completo já construído; opcionalmente informe seu caminho")
    a = ap.parse_args()

    if a.k < 1:
        ap.error("--k deve ser maior que zero")
    sistema = SistemaRAG(a.n, a.semente, None if a.sem_llm else a.llm, a.k, a.max_tokens, a.min_sim,
                         indice_completo=a.indice_completo)
    if a.pergunta:
        imprimir(sistema.perguntar(a.pergunta), a.trechos)
        return
    if a.indice_completo:
        print("Índice lexical completo carregado. Pergunte em inglês; linha vazia ou 'sair' encerra.")
    else:
        print(f"{len(sistema.passagens)} passagens indexadas. Pergunte em inglês; linha vazia ou 'sair' encerra.")
    while True:
        pergunta = input("\nPergunta> ").strip()
        if pergunta.lower() in ("", "sair", "exit"):
            break
        imprimir(sistema.perguntar(pergunta), a.trechos)


if __name__ == "__main__":
    main()
