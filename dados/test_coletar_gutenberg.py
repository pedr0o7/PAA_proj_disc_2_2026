import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import coletar_gutenberg as cg


def meta(id_, titulo="Livro", autores="Autor, A."):
    return json.dumps({"text_id": id_, "title": titulo, "authors": autores})


class LerMetadadosTest(unittest.TestCase):
    def test_extrai_campos_e_normaliza_espacos(self):
        self.assertEqual(cg.ler_metadados(meta(15, " Moby   Dick ", "Melville, Herman")),
                         (15, "Moby Dick", "Melville, Herman"))

    def test_titulo_ausente_usa_id(self):
        self.assertEqual(cg.ler_metadados(json.dumps({"text_id": "7"}))[:2], (7, "Gutenberg #7"))

    def test_invalido_retorna_none(self):
        for bruto in (None, "não é json", json.dumps({"title": "sem id"})):
            self.assertIsNone(cg.ler_metadados(bruto))


class NormalizarTest(unittest.TestCase):
    def test_crlf_espacos_e_linhas_em_branco(self):
        self.assertEqual(cg.normalizar("a  \r\n    \r\n\r\n\r\n\r\n\r\nb\r\n"), "a\n\n\nb")
        self.assertEqual(cg.normalizar("a\r\n\r\nb"), "a\n\nb")


class ExtrairTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saida = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def indice(self):
        return [json.loads(l) for l in (self.saida / "livros.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_deduplica_por_id_e_ignora_invalidos(self):
        livros = [("aaa", meta(1)), ("repetido", meta(1)), ("", meta(2)), ("bbb", "x"), ("ccc", meta(3))]
        rel = cg.extrair(livros, self.saida, meta=10_000)
        self.assertEqual([i["id"] for i in self.indice()], [1, 3])
        self.assertEqual((rel["livros_unicos"], rel["bytes_texto"], rel["atingiu_meta"]), (2, 6, False))

    def test_deslocamentos_apontam_para_o_texto_de_cada_livro(self):
        cg.extrair([("olá mundo", meta(1)), ("segundo livro", meta(2))], self.saida, meta=10_000)
        bruto = (self.saida / "textos.txt").read_bytes()
        for item, esperado in zip(self.indice(), ["olá mundo", "segundo livro"]):
            trecho = bruto[item["deslocamento"]:item["deslocamento"] + item["bytes"]]
            self.assertEqual(trecho.decode("utf-8"), esperado)
        self.assertTrue(self.indice()[0]["url"].endswith("/ebooks/1"))

    def test_para_ao_atingir_a_meta_gravando_o_livro_inteiro(self):
        livros = [("a" * 10, meta(1)), ("b" * 10, meta(2)), ("c" * 10, meta(3))]
        rel = cg.extrair(livros, self.saida, meta=15)
        self.assertEqual([i["id"] for i in self.indice()], [1, 2])
        self.assertEqual((rel["bytes_texto"], rel["atingiu_meta"]), (20, True))
        self.assertEqual(json.loads((self.saida / "relatorio.json").read_text()), rel)

    def test_le_parquet_com_colunas_do_dataset(self):
        arq = self.saida / "t.parquet"
        pq.write_table(pa.table({"TEXT": ["um", "dois"], "SOURCE": ["g", "g"], "METADATA": [meta(1), meta(2)]}), arq)
        rel = cg.extrair(cg.livros_dos_arquivos([arq]), self.saida / "out", meta=10_000)
        self.assertEqual(rel["livros_unicos"], 2)


if __name__ == "__main__":
    unittest.main()
