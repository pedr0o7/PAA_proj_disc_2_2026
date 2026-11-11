import gzip
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet

from coletar_lote import extrair, identificar_artigos


def gravar_dump(caminho, linhas):
    with gzip.open(caminho, "wb") as arquivo:
        arquivo.write(b"INSERT INTO tabela VALUES\n")
        arquivo.write(b"".join(linhas))


class TestColetaLote(unittest.TestCase):
    def test_categorias_ciclos_deduplicacao_e_texto_em_bytes(self):
        with tempfile.TemporaryDirectory() as diretorio:
            pasta = Path(diretorio)
            dumps = {nome: pasta / f"{nome}.gz" for nome in ("page", "linktarget", "categorylinks")}
            gravar_dump(dumps["page"], [
                b"(1,14,'Geografia',0),\n",
                b"(2,14,'Clima',0),\n",
                b"(3,14,'Deserto',0),\n",
                b"(100,0,'Clima',0),\n",
                b"(101,0,'Deserto',0),\n",
                b"(102,0,'Artigo',0),\n",
                b"(103,4,'Pagina_do_projeto',0);\n",
            ])
            gravar_dump(dumps["linktarget"], [
                b"(10,14,'Geografia'),\n",
                b"(11,14,'Clima'),\n",
                b"(12,14,'Deserto');\n",
            ])
            gravar_dump(dumps["categorylinks"], [
                b"(2,'sort','data','','subcat',1,10),\n",
                b"(3,'sort','data','','subcat',1,11),\n",
                b"(1,'sort','data','','subcat',1,12),\n",
                b"(100,'sort','data','','page',1,10),\n",
                b"(100,'sort','data','','page',1,11),\n",
                b"(101,'sort','data','','page',1,12),\n",
                b"(102,'sort','data','','page',1,11),\n",
                b"(103,'sort','data','','page',1,11);\n",
            ])
            selecionados, categorias = identificar_artigos(dumps, 1)
            self.assertEqual(selecionados, {100, 102})
            self.assertEqual(categorias, {10: 0, 11: 1})
            tabela = pa.Table.from_pylist([
                {"id": "100", "title": "Clima", "url": "https://exemplo/100", "text": "É árido."},
                {"id": "101", "title": "Deserto", "url": "https://exemplo/101", "text": "Fora"},
            ])
            parquet_path = pasta / "artigos.parquet"
            parquet.write_table(tabela, parquet_path)
            saida = pasta / "resultado"
            relatorio = extrair([parquet_path], selecionados, saida, 1, categorias)
            dados = (saida / "textos.txt").read_bytes()
            registro = json.loads((saida / "artigos.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(dados[registro["offset_bytes"]:registro["bytes_texto"]], "É árido.".encode())
            self.assertEqual(relatorio["bytes_texto"], len("É árido.".encode()))
            self.assertEqual(relatorio["bytes_texto_dataset_total"], len("É árido.Fora".encode()))
            self.assertEqual(relatorio["artigos_dataset_total"], 2)
            self.assertEqual(relatorio["artigos_nao_encontrados_no_dataset"], 1)
            self.assertEqual(relatorio["artigos_extraidos"], 1)
            self.assertFalse(relatorio["atingiu_meta"])

    def test_erro_ao_encontrar_formato_invalido(self):
        from coletar_lote import ler_relacoes

        with tempfile.TemporaryDirectory() as diretorio:
            caminho = Path(diretorio) / "invalido.gz"
            gravar_dump(caminho, [b"(formato inesperado);\n"])
            with self.assertRaises(ValueError):
                list(ler_relacoes(caminho))


if __name__ == "__main__":
    unittest.main()
