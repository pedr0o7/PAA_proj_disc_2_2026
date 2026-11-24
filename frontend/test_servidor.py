import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import servidor

sys.path.insert(0, str(servidor.RAIZ / "busca"))
from indice_lexical import IndiceLexical, construir
from rag import SEM_CONTEXTO, SistemaRAG


class TestServidor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        dados = Path(self.tmp.name) / "dados"
        dados.mkdir()
        texto = ("On the voyage, nothing contributes so much to tranquillise the mind "
                 "as a steady purpose a point on which the soul may fix its intellectual eye.")
        bruto = texto.encode("utf-8")
        livro = {"id": 84, "titulo": "Frankenstein", "autores": "Shelley, Mary",
                 "url": "https://www.gutenberg.org/ebooks/84", "deslocamento": 0, "bytes": len(bruto)}
        (dados / "textos.txt").write_bytes(bruto)
        (dados / "livros.jsonl").write_text(json.dumps(livro) + "\n", encoding="utf-8")
        caminho = Path(self.tmp.name) / "indice.sqlite"
        construir(dados, caminho)
        self.sistema = SistemaRAG.__new__(SistemaRAG)
        self.sistema.indice_lexical = IndiceLexical(caminho, dados)
        self.addCleanup(self.sistema.indice_lexical.close)
        self.sistema.passagens = []
        self.sistema.k, self.sistema.min_sim, self.sistema.lm = 3, 0.4, object()
        self.cliente = TestClient(servidor.criar_app(self.sistema))
        self.addCleanup(self.cliente.close)

    def eventos(self, pergunta):
        resposta = self.cliente.post("/perguntar", json={"pergunta": pergunta})
        self.assertEqual(resposta.status_code, 200)
        return [json.loads(linha) for linha in resposta.text.splitlines()]

    def test_frase_literal_e_parafrase_no_indice_de_outra_thread(self):
        perguntas = [
            ("Where does the phrase tranquillise the mind as a steady purpose come from?",
             "correspondencia_literal"),
            ("Where does the phrase nothing contributes so much to calming the mind as a firm purpose "
             "a point on which the soul can fix its intellectual gaze come from?",
             "correspondencia_aproximada"),
        ]
        for pergunta, tipo in perguntas:
            with self.subTest(tipo=tipo):
                ev = self.eventos(pergunta)
                self.assertEqual([e["tipo"] for e in ev], ["status", "fontes", "resposta", "fim"])
                self.assertEqual(ev[1]["trechos"][0]["id"], 84)
                self.assertEqual(ev[1]["trechos"][0]["tipo_fonte"], tipo)
                self.assertEqual(ev[-1]["citados"], [1])
                self.assertIn("Frankenstein", ev[2]["texto"])

    def test_autoria_e_titulo_ausente(self):
        ev = self.eventos("Who wrote Frankenstein?")
        self.assertEqual(ev[1]["trechos"][0]["tipo_fonte"], "metadados")
        self.assertIn("Shelley", ev[2]["texto"])
        ev = self.eventos("Who wrote Harry Potter?")
        self.assertEqual(ev[1]["trechos"], [])
        self.assertIn("título exato", ev[1]["aviso"])
        self.assertEqual(ev[2]["texto"], SEM_CONTEXTO)
        self.assertEqual(ev[-1]["citados"], [])

    def test_sem_llm_e_abstencao_sem_expor_resposta_bruta(self):
        self.sistema.lm = None
        ev = self.eventos("Where does the phrase tranquillise the mind as a steady purpose come from?")
        self.assertEqual([e["tipo"] for e in ev], ["status", "fontes", "fim"])
        self.assertEqual(ev[-1]["citados"], [])
        self.sistema.lm = object()
        self.sistema.recuperar = lambda pergunta: [{"similaridade": 0.8, "texto": "context"}]
        self.sistema.gerar = lambda pergunta, trechos: "A resposta sem citação"
        ev = self.eventos("What happened?")
        self.assertEqual(ev[2]["texto"], SEM_CONTEXTO)
        self.assertNotIn("A resposta sem citação", json.dumps(ev))

    def test_pagina_info_e_pergunta_vazia(self):
        self.assertIn("RAG Project Gutenberg", self.cliente.get("/").text)
        self.assertEqual(self.cliente.get("/info").json(),
                         {"modo": "completo", "livros": 1, "passagens": None, "k": 3, "llm": True})
        self.assertEqual(self.cliente.post("/perguntar", json={"pergunta": ""}).status_code, 422)

    def test_amostragem_informa_total_de_passagens(self):
        self.sistema.indice_lexical.close()
        self.sistema.indice_lexical = None
        self.sistema.passagens = [{"texto": "passagem"}]
        self.assertEqual(self.cliente.get("/info").json(),
                         {"modo": "amostra", "livros": None, "passagens": 1, "k": 3, "llm": True})

    def test_opcoes_sao_passadas_ao_rag(self):
        with mock.patch.object(sys, "argv", ["servidor.py", "--indice-completo", "--min-sim", "0.5",
                                            "--sem-llm"]), \
             mock.patch("rag.SistemaRAG", return_value=self.sistema) as construtor, \
             mock.patch("uvicorn.run"):
            servidor.main()
        self.assertEqual(construtor.call_args.kwargs["min_sim"], 0.5)
        self.assertEqual(construtor.call_args.kwargs["indice_completo"].name, "gutenberg_lexical_v1.sqlite")
        self.assertIsNone(construtor.call_args.kwargs["llm"])

    def test_erro_vira_evento(self):
        def falhar(pergunta):
            raise RuntimeError("índice indisponível")
        self.sistema.perguntar = falhar
        ev = self.eventos("question")
        self.assertEqual(ev[-1], {"tipo": "erro", "mensagem": "índice indisponível"})


if __name__ == "__main__":
    unittest.main()
