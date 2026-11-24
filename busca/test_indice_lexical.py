import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from indice_lexical import IndiceLexical, construir, segmentar


class IndiceLexicalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dados = Path(self.tmp.name) / "dados"
        self.dados.mkdir()
        self.indice = Path(self.tmp.name) / "indice.sqlite"
        textos = [
            "The beginning of the novel. " * 80 + "Ahab pursued the white whale.",
            "A distinct book about sailing. " * 60 + "The navigator reached the harbor.",
        ]
        livros = []
        with (self.dados / "textos.txt").open("wb") as arquivo:
            for id_, texto in enumerate(textos, 1):
                conteudo = texto.encode("utf-8")
                livros.append({"id": id_, "titulo": f"Book {id_}", "autores": "Writer",
                               "url": f"https://example.org/{id_}", "deslocamento": arquivo.tell(),
                               "bytes": len(conteudo)})
                arquivo.write(conteudo + b"\n\n")
        (self.dados / "livros.jsonl").write_text(
            "".join(json.dumps(livro) + "\n" for livro in livros), encoding="utf-8"
        )

    def test_busca_trecho_no_fim_do_livro_e_preserva_fonte(self):
        construir(self.dados, self.indice)
        indice = IndiceLexical(self.indice, self.dados)
        self.addCleanup(indice.close)
        achados = indice.recuperar("Who pursued the white whale?", 5)
        fonte = next(item for item in achados if "Ahab pursued" in item["texto"])
        self.assertEqual((fonte["id"], fonte["titulo"], fonte["url"]),
                         (1, "Book 1", "https://example.org/1"))
        self.assertGreater(fonte["inicio"], 0)
        self.assertEqual(indice.recuperar("the and of?", 5), [])

    def test_retoma_transacao_interrompida_sem_duplicar_passagens(self):
        from indice_lexical import segmentar as original
        chamadas = 0

        def interromper(texto):
            nonlocal chamadas
            chamadas += 1
            if chamadas == 2:
                yield next(original(texto))
                raise RuntimeError("interrupção simulada")
            yield from original(texto)

        with patch("indice_lexical.segmentar", interromper):
            with self.assertRaisesRegex(RuntimeError, "interrupção simulada"):
                construir(self.dados, self.indice)
        with self.assertRaisesRegex(ValueError, "incompleto"):
            IndiceLexical(self.indice, self.dados)
        construir(self.dados, self.indice)
        with closing(sqlite3.connect(self.indice)) as con:
            contagens = dict(con.execute("SELECT livro_id, count(*) FROM passagens GROUP BY livro_id"))
        self.assertEqual(contagens[1], len(list(original(("The beginning of the novel. " * 80)
                                                         + "Ahab pursued the white whale."))))
        self.assertEqual(contagens[2], len(list(original(("A distinct book about sailing. " * 60)
                                                         + "The navigator reached the harbor."))))

    def test_rejeita_indice_de_outro_corpus(self):
        construir(self.dados, self.indice)
        with (self.dados / "textos.txt").open("ab") as arquivo:
            arquivo.write(b"alteracao")
        with self.assertRaisesRegex(ValueError, "corpus atual"):
            IndiceLexical(self.indice, self.dados)
        with self.assertRaisesRegex(ValueError, "corpus ou a versão"):
            construir(self.dados, self.indice)

    def test_busca_titulo_exato_no_catalogo_do_indice(self):
        construir(self.dados, self.indice)
        indice = IndiceLexical(self.indice, self.dados)
        self.addCleanup(indice.close)
        self.assertEqual([livro["id"] for livro in indice.buscar_titulo("  BOOK   1  ")], [1])
        self.assertEqual(indice.buscar_titulo("Book"), [])

    def test_busca_frase_no_texto_sem_confundir_palavras_isoladas(self):
        construir(self.dados, self.indice)
        indice = IndiceLexical(self.indice, self.dados)
        self.addCleanup(indice.close)
        achados = indice.buscar_frase("Ahab pursued the white whale", 3)
        self.assertEqual([item["id"] for item in achados], [1])
        self.assertEqual(indice.buscar_frase("Ahab reached the white whale", 3), [])


class SegmentarTest(unittest.TestCase):
    def test_texto_completo_com_sobreposicao_e_unicode(self):
        texto = "Uma frase com café e espaço. " * 20
        partes = list(segmentar(texto, tamanho=80, sobreposicao=10))
        reconstruido = partes[0][1]
        for inicio, trecho in partes[1:]:
            self.assertLessEqual(inicio, len(reconstruido))
            self.assertEqual(reconstruido[inicio:], trecho[:len(reconstruido) - inicio])
            reconstruido += trecho[len(reconstruido) - inicio:]
        self.assertEqual(reconstruido, texto)


if __name__ == "__main__":
    unittest.main()
