import unittest

from rag import SEM_CONTEXTO, validar_citacoes


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


if __name__ == "__main__":
    unittest.main()
