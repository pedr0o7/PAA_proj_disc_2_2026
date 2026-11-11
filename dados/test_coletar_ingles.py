import gzip
import hashlib
import io
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from coletar_ingles import baixar_continuavel, construir_indice


def gravar_dump(caminho, linhas):
    with gzip.open(caminho, "wb") as arquivo:
        arquivo.write(b"INSERT INTO tabela VALUES\n")
        arquivo.write(b"".join(linhas))


class Resposta(io.BytesIO):
    def __init__(self, dados, status, intervalo=""):
        super().__init__(dados)
        self.status = status
        self.headers = {"Content-Range": intervalo}


class TestColetaIngles(unittest.TestCase):
    def test_indice_filtra_namespace_e_ciclos(self):
        with tempfile.TemporaryDirectory() as diretorio:
            pasta = Path(diretorio)
            gravar_dump(pasta / "enwiki-20261101-page.sql.gz", [
                b"(1,14,'Geography',0),\n",
                b"(2,14,'Climate',0),\n",
                b"(3,14,'Deserts',0),\n",
                b"(100,0,'Climate',0),\n",
                b"(101,0,'Desert',0),\n",
                b"(102,4,'Project',0);\n",
            ])
            gravar_dump(pasta / "enwiki-20261101-linktarget.sql.gz", [
                b"(10,14,'Geography'),\n",
                b"(11,14,'Climate'),\n",
                b"(12,14,'Deserts');\n",
            ])
            gravar_dump(pasta / "enwiki-20261101-categorylinks.sql.gz", [
                b"(2,'s','data','','subcat',1,10),\n",
                b"(3,'s','data','','subcat',1,11),\n",
                b"(1,'s','data','','subcat',1,12),\n",
                b"(100,'s','data','','page',1,10),\n",
                b"(100,'s','data','','page',1,11),\n",
                b"(101,'s','data','','page',1,12),\n",
                b"(102,'s','data','','page',1,11);\n",
            ])
            indice = construir_indice(pasta, 1)
            with closing(sqlite3.connect(indice)) as conexao:
                artigos = {linha[0] for linha in conexao.execute("SELECT id FROM selecionadas")}
                categorias = dict(conexao.execute("SELECT id, profundidade FROM alcancaveis"))
            self.assertEqual(artigos, {100})
            self.assertEqual(categorias, {10: 0, 11: 1})
            self.assertEqual(construir_indice(pasta, 1), indice)

    def test_download_continua_e_valida_hash(self):
        dados = b"primeira parte e segunda parte"
        with tempfile.TemporaryDirectory() as diretorio:
            caminho = Path(diretorio) / "fonte.bin"
            parcial = caminho.with_name(caminho.name + ".part")
            parcial.write_bytes(dados[:8])
            with patch("coletar_ingles.urlopen", return_value=Resposta(dados[8:], 206, "bytes 8-29/30")):
                baixar_continuavel(
                    "https://example.com/fonte", caminho, "sha256",
                    hashlib.sha256(dados).hexdigest(), len(dados), "contato",
                )
            self.assertEqual(caminho.read_bytes(), dados)
            self.assertFalse(parcial.exists())

    def test_continuacao_rejeita_intervalo_incorreto(self):
        with tempfile.TemporaryDirectory() as diretorio:
            caminho = Path(diretorio) / "fonte.bin"
            caminho.with_name(caminho.name + ".part").write_bytes(b"abc")
            with patch("coletar_ingles.urlopen", return_value=Resposta(b"def", 206, "bytes 0-2/6")):
                with self.assertRaises(RuntimeError):
                    baixar_continuavel(
                        "https://example.com/fonte", caminho, "sha256", "0" * 64, 6, "contato"
                    )


if __name__ == "__main__":
    unittest.main()
