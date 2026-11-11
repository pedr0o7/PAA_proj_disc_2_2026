import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coletar_geografia import META_BYTES, Wikipedia, coletar


class TestColeta(unittest.TestCase):
    def test_paginacao_e_deduplicacao(self):
        wikipedia = Wikipedia("teste@example.com")
        respostas = [
            {
                "query": {"categorymembers": [
                    {"pageid": 1, "title": "Geografia"},
                    {"pageid": 2, "title": "Latitude"},
                ]},
                "continue": {"continue": "-||", "cmcontinue": "proxima"},
            },
            {
                "query": {"categorymembers": [
                    {"pageid": 1, "title": "Geografia"},
                    {"pageid": 3, "title": "Clima"},
                ]},
            },
        ]
        with patch.object(wikipedia, "consultar", side_effect=respostas) as consultar:
            artigos = wikipedia.artigos_da_categoria()
        self.assertEqual(artigos, {1: "Geografia", 2: "Latitude", 3: "Clima"})
        self.assertEqual(consultar.call_count, 2)

    def test_textos_e_contagem_em_utf8(self):
        wikipedia = Wikipedia("teste@example.com")
        pagina = {
            "pageid": 1,
            "title": "Geografia",
            "fullurl": "https://pt.wikipedia.org/wiki/Geografia",
            "extract": "Geografia é ciência.",
            "revisions": [{"revid": 42, "timestamp": "2026-01-01T00:00:00Z"}],
        }
        with tempfile.TemporaryDirectory() as diretorio:
            pasta = Path(diretorio) / "coleta"
            with patch.object(wikipedia, "artigos_da_categoria", return_value={1: "Geografia"}):
                with patch.object(wikipedia, "extrair_artigo", return_value=pagina):
                    relatorio = coletar(wikipedia, pasta)
            conteudo = (pasta / "textos" / "1.txt").read_bytes()
            salvo = json.loads((pasta / "relatorio.json").read_text(encoding="utf-8"))
            self.assertEqual(conteudo, pagina["extract"].encode("utf-8"))
            self.assertEqual(relatorio["total_bytes_texto"], len(conteudo))
            self.assertEqual(salvo["artigos"][0]["bytes_texto"], len(conteudo))
            self.assertEqual(salvo["meta_bytes"], META_BYTES)
            self.assertFalse(salvo["atingiu_meta"])
            with self.assertRaises(FileExistsError):
                coletar(wikipedia, pasta)


if __name__ == "__main__":
    unittest.main()
