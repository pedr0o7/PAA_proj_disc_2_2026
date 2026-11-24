import json
import tempfile
import unittest
from pathlib import Path

from corpus import amostrar_passagens


class AmostrarPassagensTest(unittest.TestCase):
    def test_amostra_no_maximo_uma_passagem_por_livro(self):
        with tempfile.TemporaryDirectory() as tmp:
            dados = Path(tmp)
            conteudo = ""
            livros = []
            for id_ in (1, 2):
                texto = (
                    f"Book {id_} first paragraph has enough words and a sentence ending. " * 8
                    + "\n\n"
                    + f"Book {id_} second paragraph also has enough words and a sentence ending. " * 8
                )
                inicio = len(conteudo.encode("utf-8"))
                conteudo += texto + "\n\n"
                livros.append({
                    "id": id_,
                    "titulo": f"Book {id_}",
                    "autores": "Author",
                    "url": f"https://www.gutenberg.org/ebooks/{id_}",
                    "deslocamento": inicio,
                    "bytes": len(texto.encode("utf-8")),
                })
            (dados / "textos.txt").write_bytes(conteudo.encode("utf-8"))
            (dados / "livros.jsonl").write_text(
                "".join(json.dumps(livro) + "\n" for livro in livros), encoding="utf-8"
            )

            passagens = amostrar_passagens(10, 4, dados, min_car=100, max_car=1000)

            self.assertEqual(len(passagens), 2)
            self.assertEqual(len({p["id"] for p in passagens}), len(passagens))


if __name__ == "__main__":
    unittest.main()
