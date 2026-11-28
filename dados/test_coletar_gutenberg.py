import hashlib
import io
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

    def test_remove_tabs_e_espacos_no_fim_de_cada_linha(self):
        self.assertEqual(cg.normalizar("a \t \nb\t\n\t\nc   "), "a\nb\n\nc")
        self.assertEqual(cg.normalizar("  início\n\tmeio\t"), "início\n\tmeio")
        self.assertEqual(cg.normalizar(""), "")


class FakeResposta:
    """Simula urlopen: entrega `dados` em blocos e expõe o status HTTP."""

    def __init__(self, dados, status=200):
        self.buf = io.BytesIO(dados)
        self.status = status

    def read(self, n=-1):
        return self.buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class BaixarContinuavelTest(unittest.TestCase):
    CONTEUDO = b"0123456789" * 1000
    URL = "https://exemplo.invalido/arquivo.parquet"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.destino = self.dir / "train.parquet"
        self.parcial = self.dir / "train.parquet.part"
        self.chamadas = []
        self.resposta = None
        self._urlopen = cg.urlopen
        cg.urlopen = self._fake_urlopen

    def tearDown(self):
        cg.urlopen = self._urlopen
        self.tmp.cleanup()

    def _fake_urlopen(self, req, timeout=None):
        self.chamadas.append(req)
        return self.resposta

    def baixar(self, esperado=None, tamanho=None):
        cg.baixar_continuavel(self.URL, self.destino,
                              esperado or hashlib.sha256(self.CONTEUDO).hexdigest(),
                              tamanho if tamanho is not None else len(self.CONTEUDO), "teste")

    def test_download_novo_valida_e_renomeia(self):
        self.resposta = FakeResposta(self.CONTEUDO)
        self.baixar()
        self.assertEqual(self.destino.read_bytes(), self.CONTEUDO)
        self.assertFalse(self.parcial.exists())
        self.assertNotIn("Range", self.chamadas[0].headers)

    def test_arquivo_existente_valido_nao_baixa_de_novo(self):
        self.destino.write_bytes(self.CONTEUDO)
        self.baixar()
        self.assertEqual(self.chamadas, [])

    def test_arquivo_existente_invalido_levanta_erro(self):
        self.destino.write_bytes(b"x" * len(self.CONTEUDO))
        with self.assertRaisesRegex(ValueError, "Arquivo existente inválido"):
            self.baixar()

    def test_retoma_a_partir_do_parcial_com_range(self):
        metade = len(self.CONTEUDO) // 2
        self.parcial.write_bytes(self.CONTEUDO[:metade])
        self.resposta = FakeResposta(self.CONTEUDO[metade:], status=206)
        self.baixar()
        self.assertEqual(self.chamadas[0].headers["Range"], f"bytes={metade}-")
        self.assertEqual(self.destino.read_bytes(), self.CONTEUDO)
        self.assertFalse(self.parcial.exists())

    def test_servidor_sem_suporte_a_range_levanta_erro(self):
        self.parcial.write_bytes(self.CONTEUDO[:10])
        self.resposta = FakeResposta(self.CONTEUDO, status=200)
        with self.assertRaisesRegex(RuntimeError, "não aceitou continuar"):
            self.baixar()

    def test_checksum_incorreto_mantem_parcial_e_nao_cria_destino(self):
        self.resposta = FakeResposta(self.CONTEUDO)
        with self.assertRaisesRegex(ValueError, "checksum incorreto"):
            self.baixar(esperado="0" * 64)
        self.assertTrue(self.parcial.exists())
        self.assertFalse(self.destino.exists())

    def test_download_maior_que_o_esperado_levanta_erro(self):
        self.resposta = FakeResposta(self.CONTEUDO)
        with self.assertRaisesRegex(ValueError, "excedeu"):
            self.baixar(tamanho=len(self.CONTEUDO) - 1)

    def test_parcial_maior_que_o_esperado_levanta_erro(self):
        self.parcial.write_bytes(self.CONTEUDO)
        with self.assertRaisesRegex(ValueError, "maior que o esperado"):
            self.baixar(tamanho=len(self.CONTEUDO) - 1)
        self.assertEqual(self.chamadas, [])


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


class ColetarLimpezaTest(unittest.TestCase):
    def test_limpar_parquets_remove_apenas_quando_solicitado_e_atingiu_meta(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            cache = raiz / "cache"
            cache.mkdir()
            arq = cache / "train-00000-of-00001.parquet"
            pq.write_table(pa.table({"TEXT": ["a" * 20], "SOURCE": ["g"], "METADATA": [meta(1)]}), arq)
            manifesto_fake = {"train-00000-of-00001.parquet": ("p", "0" * 64, 1)}

            with mock.patch.object(cg, "manifesto", return_value=manifesto_fake), \
                 mock.patch.object(cg, "baixar_continuavel"):
                cg.coletar("c", cache=cache, saida=raiz / "s1", meta=10, limpar_parquets=False)
                self.assertTrue(arq.exists())
                cg.coletar("c", cache=cache, saida=raiz / "s2", meta=10, limpar_parquets=True)
                self.assertFalse(arq.exists())


if __name__ == "__main__":
    unittest.main()
