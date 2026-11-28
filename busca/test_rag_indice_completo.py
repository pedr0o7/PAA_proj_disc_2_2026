import io
import re
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

    def buscar_frase(self, frase, limite):
        return [p for p in self.passagens if frase.casefold() in p.get("texto", "").casefold()][:limite]


class TokenizerFalso:
    def __call__(self, texto, **kwargs):
        return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", texto)]}

    def num_special_tokens_to_add(self, pair=False):
        return 2


class ModeloFalso:
    tokenizer = TokenizerFalso()
    max_seq_length = 128

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

    def test_embedding_considera_conteudo_no_fim_da_passagem(self):
        class ModeloJanelas:
            tokenizer = TokenizerFalso()
            max_seq_length = 8

            def encode(self, textos, **kwargs):
                return np.array([[1.0, 0.0] if "target" in texto else [0.0, 1.0]
                                 for texto in textos], dtype=np.float32)

        sistema = SistemaRAG.__new__(SistemaRAG)
        sistema.k = 1
        sistema.indice_lexical = IndiceFalso([
            {"id": 1, "titulo": "Book", "autores": "Author",
             "texto": "one two three four five six seven eight target"},
        ])
        sistema.modelo_emb = ModeloJanelas()
        sistema._vetor = lambda pergunta: np.array([1.0, 0.0], dtype=np.float32)
        self.assertEqual(sistema.recuperar("target")[0]["similaridade"], 1.0)

    def test_deduplica_passagens_sobrepostas_entre_edicoes(self):
        class ModeloQuatro:
            tokenizer = TokenizerFalso()
            max_seq_length = 128

            def encode(self, textos, **kwargs):
                vetores = [[1.0, 0.0], [0.95, 0.3], [0.9, 0.4], [0.85, 0.5]]
                return np.array(vetores[:len(textos)], dtype=np.float32)

        cena_base = (
            "Captain Ahab stood upon the quarter-deck and spoke to Starbuck, Stubb, and Flask "
            "about the white whale Moby Dick that dismasted him on a previous voyage"
        )
        cena_edicao_2 = (
            "spoke to Starbuck, Stubb, and Flask about the white whale Moby Dick that "
            "dismasted him on a previous voyage across the southern ocean"
        )
        sistema = SistemaRAG.__new__(SistemaRAG)
        sistema.k = 3
        sistema.indice_lexical = IndiceFalso([
            {"id": 2489, "titulo": "Moby Dick; Or, The Whale", "autores": "Melville", "texto": cena_base},
            {"id": 15, "titulo": "Moby-Dick; or, The Whale", "autores": "Melville", "texto": cena_edicao_2},
            {"id": 2701, "titulo": "Moby Dick; Or, The Whale", "autores": "Melville",
             "texto": "He was a moody captain with a wrinkled brow and a crooked jaw on the Pequod."},
            {"id": 2489, "titulo": "Moby Dick; Or, The Whale", "autores": "Melville",
             "texto": "Gold doubloon nailed to the mainmast as a reward for sighting the whale."},
        ])
        sistema.modelo_emb = ModeloQuatro()
        sistema._vetor = lambda pergunta: np.array([1.0, 0.0], dtype=np.float32)

        achados = sistema.recuperar("Who is Captain Ahab?")
        self.assertEqual([p["id"] for p in achados], [2489, 2701, 2489])
        self.assertNotIn(15, [p["id"] for p in achados])


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

    def test_titulo_ausente_nao_gera_resposta_de_memoria(self):
        resultado = self.sistema.perguntar("Who wrote a different title?")
        self.assertEqual(resultado["resposta"], SEM_CONTEXTO)
        self.assertEqual(resultado["citados"], [])
        self.assertIn("título exato", resultado["aviso"])

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


class CitacaoTest(unittest.TestCase):
    def setUp(self):
        self.frase = "nothing contributes so much to tranquillise the mind as a steady purpose"
        self.trecho = {"id": 84, "titulo": "Frankenstein", "autores": "Mary Shelley",
                       "url": "https://www.gutenberg.org/ebooks/84",
                       "texto": "The narrator says that " + self.frase + " during the voyage."}
        self.sistema = SistemaRAG.__new__(SistemaRAG)
        self.sistema.indice_lexical = IndiceFalso([self.trecho])
        self.sistema.k = 3
        self.sistema.min_sim = 0.99
        self.sistema.lm = object()
        self.sistema.gerar = lambda pergunta, trechos: self.fail("não deveria chamar o LLM")

    def test_frase_literal_cita_trecho_comprovado_sem_alegar_origem(self):
        resultado = self.sistema.perguntar(f"Where does the phrase {self.frase} come from?")
        self.assertEqual(resultado["citados"], [1])
        self.assertEqual(resultado["trechos"][0]["id"], 84)
        self.assertEqual(resultado["trechos"][0]["tipo_fonte"], "correspondencia_literal")
        self.assertIn("does not establish", resultado["resposta"])
        self.assertNotIn("similaridade", resultado["trechos"][0])

    def test_pergunta_alternativa_sobre_livro_encontra_frase(self):
        resultado = self.sistema.perguntar(f"Which book contains the quote {self.frase}?")
        self.assertEqual(resultado["citados"], [1])
        self.assertEqual(resultado["trechos"][0]["id"], 84)

    def test_parafrase_mostra_correspondencia_com_ressalva(self):
        pergunta = ("Where does the phrase nothing contributes so much to calming the mind "
                    "as a firm purpose come from?")
        resultado = self.sistema.perguntar(pergunta)
        self.assertEqual(resultado["citados"], [1])
        self.assertEqual(resultado["trechos"][0]["id"], 84)
        self.assertEqual(resultado["trechos"][0]["tipo_fonte"], "correspondencia_aproximada")
        self.assertIn("similar", resultado["resposta"].lower())
        self.assertIn("does not establish", resultado["resposta"])

    def test_frase_ausente_nao_cita_candidato_irrelevante(self):
        pergunta = "Where does the phrase unicorns conquer galaxies with magic umbrellas come from?"
        resultado = self.sistema.perguntar(pergunta)
        self.assertEqual(resultado["resposta"], SEM_CONTEXTO)
        self.assertEqual(resultado["citados"], [])
        self.assertIn("correspondência", resultado["aviso"])

    def test_frase_curta_nao_inventa_origem(self):
        resultado = self.sistema.perguntar("Where does the phrase magic come from?")
        self.assertEqual(resultado["resposta"], SEM_CONTEXTO)
        self.assertEqual(resultado["citados"], [])
        self.assertIn("curta", resultado["aviso"])

    def test_negacao_e_tratada_apenas_como_redacao_semelhante(self):
        pergunta = ("Where does the phrase nothing contributes so much to tranquillise the mind "
                    "as a steady purpose not come from?")
        resultado = self.sistema.perguntar(pergunta)
        self.assertEqual(resultado["trechos"][0]["tipo_fonte"], "correspondencia_aproximada")
        self.assertIn("does not establish", resultado["resposta"])

    def test_sem_llm_exibe_fonte_literal_mas_nao_gera_resposta(self):
        self.sistema.lm = None
        resultado = self.sistema.perguntar(f"Where does the phrase {self.frase} come from?")
        self.assertEqual(resultado["resposta"], "")
        self.assertEqual(resultado["citados"], [])
        self.assertEqual(resultado["trechos"][0]["tipo_fonte"], "correspondencia_literal")
        with redirect_stdout(io.StringIO()) as saida:
            imprimir(resultado, True)
        self.assertNotIn("modelo não confirmou", saida.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
