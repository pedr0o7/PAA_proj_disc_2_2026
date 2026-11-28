import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from rag import EXEMPLOS_GERACAO, PROMPT_SISTEMA, SEM_CONTEXTO, SistemaRAG, validar_citacoes


class ValidarCitacoesTest(unittest.TestCase):
    def test_abstains_when_answer_has_no_citation(self):
        self.assertEqual(validar_citacoes("An unsupported answer.", 3), (SEM_CONTEXTO, []))

    def test_removes_nonexistent_citations(self):
        self.assertEqual(validar_citacoes("Answer [8].", 3), (SEM_CONTEXTO, []))

    def test_invalid_citation_only_becomes_insufficient_context(self):
        self.assertEqual(validar_citacoes("[8]", 3), (SEM_CONTEXTO, []))

    def test_keeps_only_citations_for_available_passages(self):
        self.assertEqual(validar_citacoes("Supported [2], invalid [8].", 3),
                         ("Supported [2], invalid .", [2]))

    def test_keeps_insufficient_context_response(self):
        self.assertEqual(validar_citacoes(SEM_CONTEXTO, 0), (SEM_CONTEXTO, []))

    def test_normalizes_passage_and_source_citation_formats(self):
        self.assertEqual(
            validar_citacoes("Ahab hunts the whale [Passage 1] and Source [2].", 3),
            ("Ahab hunts the whale [1] and [2].", [1, 2]),
        )


class SuporteSemanticoCitacoesTest(unittest.TestCase):
    def _sistema_com_emb(self, mapa):
        sistema = SistemaRAG.__new__(SistemaRAG)
        modelo = Mock()
        del modelo.tokenizer  # usa o caminho direto sem janelas

        def fake_encode(textos, **kwargs):
            return np.array([mapa.get(t, [0.0, 1.0]) for t in textos], dtype=np.float32)

        modelo.encode.side_effect = fake_encode
        sistema.modelo_emb = modelo
        return sistema

    def test_mantem_citacao_com_suporte_semantico_e_rejeita_alucinacao(self):
        trecho_moby = {
            "titulo": "Moby-Dick", "autores": "Melville",
            "texto": "Captain Ahab hunts the white whale.",
        }
        sistema = self._sistema_com_emb({
            "Captain Ahab hunts the whale.": [1.0, 0.0],
            "Moby-Dick by Melville. Captain Ahab hunts the white whale.": [1.0, 0.0],
            "The capital of Brazil is Rio de Janeiro.": [0.0, 1.0],
        })
        resp_ok, cit_ok = sistema._validar_suporte_semantico(
            "Captain Ahab hunts the whale. [1]", [trecho_moby], [1], min_suporte=0.3
        )
        self.assertEqual(cit_ok, [1])
        self.assertIn("[1]", resp_ok)

        resp_aluc, cit_aluc = sistema._validar_suporte_semantico(
            "The capital of Brazil is Rio de Janeiro. [1]", [trecho_moby], [1], min_suporte=0.3
        )
        self.assertEqual((resp_aluc, cit_aluc), (SEM_CONTEXTO, []))

    def test_sem_modelo_emb_preserva_citacoes_sintaticas(self):
        sistema = SistemaRAG.__new__(SistemaRAG)
        sistema.modelo_emb = None
        self.assertEqual(sistema._validar_suporte_semantico("Answer [1].", [{"texto": "t"}], [1]),
                         ("Answer [1].", [1]))

    def test_rejeita_resposta_inteira_se_alguma_citacao_falhar_no_suporte_semantico(self):
        trecho_ok = {"titulo": "Book 1", "autores": "A", "texto": "Ahab hunts the white whale."}
        trecho_ruim = {"titulo": "Book 2", "autores": "B", "texto": "Unrelated gardening tips."}
        sistema = self._sistema_com_emb({
            "Ahab hunts the whale.": [1.0, 0.0],
            "Cooking tips.": [1.0, 0.0],
            "Book 1 by A. Ahab hunts the white whale.": [1.0, 0.0],
            "Book 2 by B. Unrelated gardening tips.": [0.0, 1.0],
        })
        resp, cit = sistema._validar_suporte_semantico(
            "Ahab hunts the whale. [1] Cooking tips. [2]", [trecho_ok, trecho_ruim], [1, 2], min_suporte=0.3
        )
        self.assertEqual((resp, cit), (SEM_CONTEXTO, []))

    def test_valida_cada_citacao_contra_sua_propria_frase_em_resposta_multipassagem(self):
        t1 = {"titulo": "Moby-Dick", "autores": "Melville", "texto": "Ahab hunts the white whale."}
        t2 = {"titulo": "Frankenstein", "autores": "Shelley", "texto": "Elizabeth lives in Geneva."}
        sistema = self._sistema_com_emb({
            "Ahab hunts the whale.": [1.0, 0.0],
            "Moby-Dick by Melville. Ahab hunts the white whale.": [1.0, 0.0],
            "Elizabeth lives in Geneva.": [0.0, 1.0],
            "Frankenstein by Shelley. Elizabeth lives in Geneva.": [0.0, 1.0],
        })
        resp, cit = sistema._validar_suporte_semantico(
            "Ahab hunts the whale [1]. Elizabeth lives in Geneva [2].",
            [t1, t2],
            [1, 2],
            min_suporte=0.3,
        )
        self.assertEqual(cit, [1, 2])
        self.assertIn("[1]", resp)
        self.assertIn("[2]", resp)

    def test_tempo_total_inclui_validacao_semantica(self):
        sistema = SistemaRAG.__new__(SistemaRAG)
        sistema.indice_lexical = None
        sistema.k = 1
        sistema.min_sim = 0.0
        sistema.lm = object()
        sistema.recuperar = lambda p: [{"id": 1, "titulo": "T", "autores": "A", "texto": "x", "similaridade": 0.9}]
        sistema.gerar = lambda p, t: "Answer [1]."

        def validar_lento(resp, trechos, citados):
            time.sleep(0.05)
            return resp, citados

        sistema._validar_suporte_semantico = validar_lento
        res = sistema.perguntar("Question?")
        self.assertGreaterEqual(res["tempo_total"], res["tempo_recuperacao"] + res["tempo_geracao"] + 0.04)


class GerarMensagensTest(unittest.TestCase):
    def test_gerar_inclui_prompt_sistema_e_exemplos_few_shot(self):
        sistema = SistemaRAG.__new__(SistemaRAG)
        sistema.max_tokens = 120
        capturado = {}

        def fake_completar(msgs, max_tokens):
            capturado["msgs"] = msgs
            capturado["max_tokens"] = max_tokens
            return "Ahab hunts Moby Dick [1]."

        sistema.completar = fake_completar
        trechos = [{"titulo": "Moby-Dick", "autores": "Herman Melville", "texto": "Ahab hunts Moby Dick."}]
        resp = sistema.gerar("Who is Ahab?", trechos)

        self.assertEqual(resp, "Ahab hunts Moby Dick [1].")
        self.assertEqual(capturado["max_tokens"], 120)
        msgs = capturado["msgs"]
        self.assertEqual(msgs[0], {"role": "system", "content": PROMPT_SISTEMA})
        self.assertEqual(msgs[1:-1], EXEMPLOS_GERACAO)
        self.assertTrue(any(m["role"] == "assistant" and "[1]" in m["content"] for m in EXEMPLOS_GERACAO))
        self.assertTrue(any(m["role"] == "assistant" and m["content"] == SEM_CONTEXTO for m in EXEMPLOS_GERACAO))
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertIn("[1] (Moby-Dick by Herman Melville) Ahab hunts Moby Dick.", msgs[-1]["content"])
        self.assertIn("Question: Who is Ahab?", msgs[-1]["content"])
        self.assertIn("([1])", msgs[-1]["content"])


if __name__ == "__main__":
    unittest.main()
