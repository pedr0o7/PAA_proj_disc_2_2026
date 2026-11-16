import json
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import servidor

TRECHOS = [{"texto": "Paris is the capital of France.", "titulo": "Paris", "url": "https://en.wikipedia.org/wiki/Paris",
            "similaridade": 0.8},
           {"texto": "Lyon is a city.", "titulo": "Lyon", "url": "https://en.wikipedia.org/wiki/Lyon", "similaridade": 0.4}]


class SistemaFalso:
    def __init__(self, lm=None, falhar=False):
        self.passagens, self.k, self.lm, self.falhar = [{}] * 10, 2, lm, falhar

    def recuperar(self, pergunta):
        if self.falhar:
            raise RuntimeError("índice indisponível")
        return TRECHOS


def eventos(sistema, pergunta="What is the capital of France?"):
    r = TestClient(servidor.criar_app(sistema)).post("/perguntar", json={"pergunta": pergunta})
    return r.status_code, [json.loads(l) for l in r.text.splitlines() if l]


class TestServidor(unittest.TestCase):
    def test_sem_llm_so_fontes(self):
        status, ev = eventos(SistemaFalso())
        self.assertEqual(status, 200)
        self.assertEqual([e["tipo"] for e in ev], ["fontes", "fim"])
        self.assertEqual(ev[0]["trechos"], TRECHOS)
        self.assertEqual(ev[1]["citados"], [])

    def test_com_llm_texto_em_partes_e_citacoes(self):
        # [3] e [0] estão fora do intervalo de trechos e não contam como citação.
        partes = ["Paris is the capital [1]", " of France [1][3].", " See [0]."]
        with mock.patch.object(servidor, "gerar_em_partes", return_value=iter(partes)):
            _, ev = eventos(SistemaFalso(lm=object()))
        self.assertEqual([e["tipo"] for e in ev], ["fontes", "texto", "texto", "texto", "fim"])
        self.assertEqual("".join(e["delta"] for e in ev if e["tipo"] == "texto"), "".join(partes))
        self.assertEqual(ev[-1]["citados"], [1])

    def test_erro_vira_evento(self):
        _, ev = eventos(SistemaFalso(falhar=True))
        self.assertEqual(ev, [{"tipo": "erro", "mensagem": "índice indisponível"}])

    def test_pergunta_vazia_rejeitada(self):
        status, _ = eventos(SistemaFalso(), pergunta="")
        self.assertEqual(status, 422)

    def test_pagina_e_info(self):
        c = TestClient(servidor.criar_app(SistemaFalso()))
        self.assertIn("RAG Wikipédia", c.get("/").text)
        self.assertEqual(c.get("/info").json(), {"passagens": 10, "k": 2, "llm": False})


if __name__ == "__main__":
    unittest.main()
