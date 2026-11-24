import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from consultar import imprimir
from rag import SEM_CONTEXTO, SistemaRAG


class IndiceFalso:
    def __init__(self, passagens):
        self.passagens = passagens

    def recuperar(self, pergunta, limite):
        return self.passagens[:limite]

    def buscar_titulo(self, titulo):
        return [p for p in self.passagens if p["titulo"].casefold() == titulo.casefold()]


class ModeloFalso:
    def encode(self, textos, **kwargs):
        return np.array([[0.0, 1.0], [0.6, 0.8], [1.0, 0.0]], dtype=np.float32)[:len(textos)]


class RecuperacaoCompletaTest(unittest.TestCase):
    def test_reordena_candidatos_e_preserva_similaridade_cosseno(self):
        sistema = SistemaRAG.__new__(SistemaRAG)
        sistema.k = 3
        sistema.indice_lexical = IndiceFalso([
            {"id": 1, "titulo": "First", "autores": "Author", "texto": "First passage"},
            {"id": 2, "titulo": "Second", "autores": "Author", "texto": "Second passage"},
            {"id": 3, "titulo": "Third", "autores": "Author", "texto": "Third passage"},
        ])
        sistema.modelo_emb = ModeloFalso()
        sistema._vetor = lambda pergunta: np.array([1.0, 0.0], dtype=np.float32)

        achados = sistema.recuperar("question")
        self.assertEqual([p["id"] for p in achados], [1, 3, 2])
        self.assertEqual([p["similaridade"] for p in achados[:2]], [0.0, 1.0])
        self.assertAlmostEqual(achados[2]["similaridade"], 0.6)

    def test_retorna_lista_vazia_se_nao_ha_candidatos(self):
        sistema = SistemaRAG.__new__(SistemaRAG)
        sistema.k = 3
        sistema.indice_lexical = IndiceFalso([])
        self.assertEqual(sistema.recuperar("question"), [])


class AutoriaTest(unittest.TestCase):
    def setUp(self):
        self.livro = {"id": 1342, "titulo": "Pride and Prejudice", "autores": "Austen, Jane, 1775-1817",
                      "url": "https://www.gutenberg.org/ebooks/1342"}
        self.sistema = SistemaRAG.__new__(SistemaRAG)
        self.sistema.indice_lexical = IndiceFalso([self.livro])
        self.sistema.lm = object()
        self.sistema.min_sim = 0.99
        self.sistema.recuperar = lambda pergunta: self.fail("não deveria recuperar passagens")
        self.sistema.gerar = lambda pergunta, trechos: self.fail("não deveria chamar o LLM")

    def test_autoria_cita_metadados_sem_usar_passagem_ou_similaridade(self):
        resultado = self.sistema.perguntar("Who wrote Pride and Prejudice?")
        self.assertIn("Austen, Jane", resultado["resposta"])
        self.assertIn("[1]", resultado["resposta"])
        self.assertEqual(resultado["citados"], [1])
        self.assertEqual(resultado["trechos"][0]["id"], 1342)
        self.assertEqual(resultado["trechos"][0]["tipo_fonte"], "metadados")
        self.assertNotIn("similaridade", resultado["trechos"][0])
        with redirect_stdout(io.StringIO()) as saida:
            imprimir(resultado, True)
        self.assertIn("metadados", saida.getvalue().lower())
        self.assertNotIn("similaridade", saida.getvalue().lower())

    def test_autoria_ambigua_nao_atribui_colaborador_ao_autor_original(self):
        casos = [
            {"autores": ""},
            {"autores": "Anonymous"},
            {"autores": "Author, Ren\ufffd"},
            {"autores": "Writer [Editor]"},
            {"autores": "Original Author; Adapter"},
            {"titulo": "A play based on Pride and Prejudice", "autores": "Adapter"},
        ]
        for alteracoes in casos:
            with self.subTest(alteracoes=alteracoes):
                livro = {**self.livro, **alteracoes}
                self.sistema.indice_lexical = IndiceFalso([livro])
                resultado = self.sistema.perguntar(f"Who wrote {livro['titulo']}?")
                self.assertEqual(resultado["resposta"], SEM_CONTEXTO)
                self.assertEqual(resultado["citados"], [])

    def test_edicoes_com_atribuicoes_divergentes_provocam_abstencao(self):
        self.sistema.indice_lexical = IndiceFalso([
            self.livro, {**self.livro, "id": 2, "autores": "Other, Author"},
        ])
        resultado = self.sistema.perguntar("Who wrote Pride and Prejudice?")
        self.assertEqual(resultado["resposta"], SEM_CONTEXTO)
        self.assertEqual(resultado["citados"], [])

    def test_edicoes_com_mesma_atribuicao_podem_citar_uma_edicao(self):
        self.sistema.indice_lexical = IndiceFalso([
            self.livro, {**self.livro, "id": 2},
        ])
        self.assertEqual(self.sistema.perguntar("Who wrote Pride and Prejudice?")["citados"], [1])

    def test_titulo_ausente_recorre_a_busca_normal(self):
        trecho = {**self.livro, "texto": "A passage", "similaridade": 0.8}
        self.sistema.min_sim = 0.4
        self.sistema.recuperar = lambda pergunta: [trecho]
        self.sistema.gerar = lambda pergunta, trechos: "A supported answer [1]."
        resultado = self.sistema.perguntar("Who wrote a different title?")
        self.assertEqual(resultado["resposta"], "A supported answer [1].")
        self.assertEqual(resultado["trechos"], [trecho])

    def test_sem_llm_e_pergunta_de_enredo_preservam_recuperacao(self):
        trecho = {**self.livro, "texto": "A passage", "similaridade": 0.8}
        self.sistema.min_sim = 0.4
        self.sistema.recuperar = lambda pergunta: [trecho]
        self.sistema.lm = None
        resultado = self.sistema.perguntar("Who wrote Pride and Prejudice?")
        self.assertEqual(resultado["resposta"], "")
        self.assertEqual(resultado["trechos"], [trecho])
        self.sistema.lm = object()
        self.sistema.gerar = lambda pergunta, trechos: "A plot answer [1]."
        resultado = self.sistema.perguntar("What happens in Pride and Prejudice?")
        self.assertEqual(resultado["resposta"], "A plot answer [1].")
        self.assertEqual(resultado["trechos"], [trecho])


if __name__ == "__main__":
    unittest.main()
