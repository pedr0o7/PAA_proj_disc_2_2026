import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import numpy as np

import avaliar
import benchmark
import rag
from infraestrutura.arquivos import gravar_atomico


class TestCacheEmbeddings(unittest.TestCase):
    def test_pede_consentimento_antes_de_carregar_modelo_para_cache_antigo(self):
        with tempfile.TemporaryDirectory() as diretorio, \
             patch.object(rag, "CACHE", Path(diretorio)), \
             patch.object(rag, "preparar_acesso_hub", return_value=True), \
             patch.object(rag, "SentenceTransformer") as modelo:
            (Path(diretorio) / "passagens_titulo_1_42_1.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "--regenerar-cache"):
                rag.SistemaRAG(n=1, llm=None)
            modelo.assert_not_called()

    def test_reusa_somente_mesmas_passagens_e_revisao(self):
        with tempfile.TemporaryDirectory() as diretorio, patch.object(rag, "CACHE", Path(diretorio)):
            sistema = rag.SistemaRAG.__new__(rag.SistemaRAG)
            sistema.modelo_emb = Mock()
            sistema.modelo_emb.get_sentence_embedding_dimension.return_value = 2
            sistema.modelo_emb.encode.return_value = np.array([[1.0, 2.0]], dtype=np.float32)
            sistema.passagens = [{"texto": "um", "titulo": "t", "url": "u"}]
            primeiro = sistema._embeddings()
            np.testing.assert_array_equal(sistema._embeddings(), primeiro)
            self.assertEqual(sistema.modelo_emb.encode.call_count, 1)
            sistema.passagens = [{"texto": "dois", "titulo": "t", "url": "u"}]
            sistema._embeddings()
            with patch.object(rag, "REVISAO_EMB", "outra-revisao"):
                sistema._embeddings()
            self.assertEqual(sistema.modelo_emb.encode.call_count, 3)
            self.assertEqual(len(list(Path(diretorio).glob("embeddings_v1_*.npz"))), 3)

    def test_rejeita_quantidade_incompativel(self):
        with tempfile.TemporaryDirectory() as diretorio, patch.object(rag, "CACHE", Path(diretorio)):
            sistema = rag.SistemaRAG.__new__(rag.SistemaRAG)
            sistema.modelo_emb = Mock()
            sistema.modelo_emb.get_sentence_embedding_dimension.return_value = 2
            sistema.modelo_emb.encode.return_value = np.array([[1.0, 2.0]], dtype=np.float32)
            sistema.passagens = [{"texto": "um", "titulo": "t", "url": "u"}]
            sistema._embeddings()
            caminho = next(Path(diretorio).glob("*.npz"))
            with np.load(caminho, allow_pickle=False) as registro:
                metadados = registro["metadados"]
            np.savez(caminho, metadados=metadados, embeddings=np.zeros((2, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "embeddings inválido ou incompatível"):
                sistema._embeddings()

    def test_rejeita_largura_incompativel_com_modelo(self):
        with tempfile.TemporaryDirectory() as diretorio, patch.object(rag, "CACHE", Path(diretorio)):
            sistema = rag.SistemaRAG.__new__(rag.SistemaRAG)
            sistema.modelo_emb = Mock()
            sistema.modelo_emb.get_sentence_embedding_dimension.return_value = 2
            sistema.modelo_emb.encode.return_value = np.array([[1.0, 2.0]], dtype=np.float32)
            sistema.passagens = [{"texto": "um", "titulo": "t", "url": "u"}]
            sistema._embeddings()
            caminho = next(Path(diretorio).glob("*.npz"))
            with np.load(caminho, allow_pickle=False) as registro:
                metadados = registro["metadados"]
            np.savez(caminho, metadados=metadados, embeddings=np.zeros((1, 3), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "embeddings inválido ou incompatível"):
                sistema._embeddings()

    def test_rejeita_npz_corrompido_ou_sem_chaves(self):
        with tempfile.TemporaryDirectory() as diretorio, patch.object(rag, "CACHE", Path(diretorio)):
            sistema = rag.SistemaRAG.__new__(rag.SistemaRAG)
            sistema.modelo_emb = Mock()
            sistema.modelo_emb.get_sentence_embedding_dimension.return_value = 2
            sistema.modelo_emb.encode.return_value = np.array([[1.0, 2.0]], dtype=np.float32)
            sistema.passagens = [{"texto": "um", "titulo": "t", "url": "u"}]
            sistema._embeddings()
            caminho = next(Path(diretorio).glob("*.npz"))

            for dados in (b"", b"nao e zip", b"PK\x03\x04corrompido"):
                caminho.write_bytes(dados)
                with self.assertRaisesRegex(ValueError, "embeddings inválido ou incompatível"):
                    sistema._embeddings()

            np.savez(caminho, embeddings=np.zeros((1, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "embeddings inválido ou incompatível"):
                sistema._embeddings()

            np.savez(caminho, metadados=np.array(["a", "b"]), embeddings=np.zeros((1, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "embeddings inválido ou incompatível"):
                sistema._embeddings()

            np.savez(caminho, metadados="{invalido", embeddings=np.zeros((1, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "embeddings inválido ou incompatível"):
                sistema._embeddings()

            np.savez(caminho, metadados=np.array(123), embeddings=np.zeros((1, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "embeddings inválido ou incompatível"):
                sistema._embeddings()

            caminho.unlink()
            sistema._embeddings()
            with caminho.open("rb") as f, np.load(f, allow_pickle=False) as reg:
                meta_ok = reg["metadados"]
            np.savez(caminho, metadados=meta_ok, embeddings=np.zeros((1, 0), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "embeddings inválido ou incompatível"):
                sistema._embeddings()

            with zipfile.ZipFile(caminho, "w") as zf:
                zf.writestr("metadados.npy", b"bytes sem cabecalho npy")
                zf.writestr("embeddings.npy", b"bytes sem cabecalho npy")
            with self.assertRaisesRegex(ValueError, "embeddings inválido ou incompatível"):
                sistema._embeddings()

            np.savez(caminho, metadados=meta_ok, embeddings=np.zeros((1, 2), dtype=np.float32))
            with zipfile.ZipFile(caminho, "r") as zin:
                meta_bytes = zin.read("metadados.npy")
                emb_bytes = zin.read("embeddings.npy")
            with zipfile.ZipFile(caminho, "w") as zf:
                zf.writestr("metadados.npy", meta_bytes)
                zf.writestr("embeddings.npy", b"bytes sem cabecalho npy")
            with self.assertRaisesRegex(ValueError, "embeddings inválido ou incompatível"):
                sistema._embeddings()
            with zipfile.ZipFile(caminho, "w") as zf:
                zf.writestr("metadados.npy", meta_bytes)
                zf.writestr("embeddings.npy", emb_bytes[:20])
            with self.assertRaisesRegex(ValueError, "embeddings inválido ou incompatível"):
                sistema._embeddings()

    def test_gravar_atomico_cria_diretorios_pai(self):
        with tempfile.TemporaryDirectory() as diretorio:
            alvo = Path(diretorio) / "a" / "b" / "arquivo.bin"
            gravar_atomico(alvo, lambda f: f.write(b"ok"))
            self.assertEqual(alvo.read_bytes(), b"ok")

    def test_sistema_rag_propaga_offline_apos_fallback_de_rede(self):
        passagens = [{"texto": "um", "titulo": "t", "url": "u"}]
        emb = np.array([[1.0, 0.0]], dtype=np.float32)
        modelo_emb = Mock()
        modelo_emb.get_sentence_embedding_dimension.return_value = 2
        modelo_emb.encode.return_value = emb
        tok = Mock()
        lm = Mock()
        lm.eval.return_value = lm
        chamadas_emb = []
        chamadas_tok = []

        def fake_st(*args, **kwargs):
            chamadas_emb.append(kwargs["local_files_only"])
            if not kwargs["local_files_only"]:
                raise httpx.ConnectError("sem rede")
            return modelo_emb

        def fake_tok(*args, **kwargs):
            chamadas_tok.append(kwargs["local_files_only"])
            return tok

        with tempfile.TemporaryDirectory() as diretorio, \
             patch.object(rag, "CACHE", Path(diretorio) / "cache"), \
             patch.object(rag, "preparar_acesso_hub", return_value=False), \
             patch.object(rag, "SentenceTransformer", side_effect=fake_st), \
             patch.object(rag, "carregar_passagens", return_value=(passagens, True)) as carregar_p, \
             patch.object(rag.AutoTokenizer, "from_pretrained", side_effect=fake_tok), \
             patch.object(rag.AutoModelForCausalLM, "from_pretrained", return_value=lm) as carregar_lm, \
             self.assertLogs("busca.hub", level="WARNING"):
            sistema = rag.SistemaRAG(n=1, llm="modelo/teste")

        self.assertTrue(sistema.offline)
        self.assertEqual(chamadas_emb, [False, True])
        self.assertTrue(carregar_p.call_args.kwargs["offline"])
        self.assertEqual(chamadas_tok, [True])
        self.assertTrue(carregar_lm.call_args.kwargs["local_files_only"])

    def test_sistema_rag_propaga_offline_do_corpus_para_llm(self):
        passagens = [{"texto": "um", "titulo": "t", "url": "u"}]
        modelo_emb = Mock()
        modelo_emb.get_sentence_embedding_dimension.return_value = 2
        modelo_emb.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
        lm = Mock()
        lm.eval.return_value = lm

        with tempfile.TemporaryDirectory() as diretorio, \
             patch.object(rag, "CACHE", Path(diretorio)), \
             patch.object(rag, "preparar_acesso_hub", return_value=False), \
             patch.object(rag, "SentenceTransformer", return_value=modelo_emb), \
             patch.object(rag, "carregar_passagens", return_value=(passagens, True)), \
             patch.object(rag.AutoTokenizer, "from_pretrained", return_value=Mock()) as carregar_tok, \
             patch.object(rag.AutoModelForCausalLM, "from_pretrained", return_value=lm) as carregar_lm:
            sistema = rag.SistemaRAG(n=1, llm="modelo/teste")

        self.assertTrue(sistema.offline)
        self.assertTrue(carregar_tok.call_args.kwargs["local_files_only"])
        self.assertTrue(carregar_lm.call_args.kwargs["local_files_only"])

    def test_avaliar_usa_revisao_emb_e_atualiza_offline(self):
        sistema = Mock(offline=False, tok=Mock(), lm=Mock())
        itens = [{"pergunta": "p", "referencia": "r", "titulo_esperado": "t"}]
        resultados = [{"resposta": "resp", "trechos": [{"texto": "ctx"}]}]
        df = Mock()
        df.columns = ["faithfulness"]
        col = Mock()
        col.dropna.return_value = Mock(empty=False)
        col.mean.return_value = 0.5
        col.isna.return_value = Mock(sum=Mock(return_value=0))
        df.__getitem__ = Mock(return_value=col)
        chamadas_hf_emb = []

        class FakeHFEmbeddings:
            def __init__(self, **kwargs):
                chamadas_hf_emb.append(kwargs)
                if not kwargs["model_kwargs"]["local_files_only"]:
                    raise httpx.ConnectError("sem rede")

        fake_lc_hf = types.ModuleType("langchain_huggingface")
        fake_lc_hf.HuggingFaceEmbeddings = FakeHFEmbeddings
        fake_lc_hf.HuggingFacePipeline = lambda **kwargs: Mock()
        fake_ragas = types.ModuleType("ragas")
        fake_ragas.EvaluationDataset = lambda x: x
        fake_ragas.SingleTurnSample = lambda **kwargs: kwargs
        fake_ragas.evaluate = lambda *args, **kwargs: Mock(to_pandas=lambda: df)
        fake_ragas_emb = types.ModuleType("ragas.embeddings")
        fake_ragas_emb.LangchainEmbeddingsWrapper = lambda x: x
        fake_ragas_llms = types.ModuleType("ragas.llms")
        fake_ragas_llms.LangchainLLMWrapper = lambda x: x
        fake_ragas_metrics = types.ModuleType("ragas.metrics")
        for nome in ("Faithfulness", "LLMContextPrecisionWithReference", "LLMContextRecall",
                     "ResponseRelevancy", "SemanticSimilarity"):
            setattr(fake_ragas_metrics, nome, lambda **kwargs: nome)
        fake_ragas_rc = types.ModuleType("ragas.run_config")
        fake_ragas_rc.RunConfig = lambda **kwargs: kwargs

        with patch.dict(sys.modules, {
            "langchain_huggingface": fake_lc_hf,
            "ragas": fake_ragas,
            "ragas.embeddings": fake_ragas_emb,
            "ragas.llms": fake_ragas_llms,
            "ragas.metrics": fake_ragas_metrics,
            "ragas.run_config": fake_ragas_rc,
        }), patch("transformers.pipeline", return_value=Mock()), \
             self.assertLogs("busca.hub", level="WARNING"):
            avaliar.avaliar_ragas(sistema, itens, resultados, rag.LLM)

        self.assertTrue(sistema.offline)
        self.assertEqual(chamadas_hf_emb[0]["model_name"], rag.MODELO_EMB)
        self.assertEqual(chamadas_hf_emb[0]["model_kwargs"], {"revision": rag.REVISAO_EMB, "local_files_only": False})
        self.assertEqual(chamadas_hf_emb[1]["model_kwargs"], {"revision": rag.REVISAO_EMB, "local_files_only": True})

    def test_benchmark_usa_revisao_emb_para_modelo_padrao(self):
        fake_st_mod = types.ModuleType("sentence_transformers")
        chamadas = []

        class FakeST:
            def __init__(self, *args, **kwargs):
                chamadas.append((args, kwargs))
                self.max_seq_length = 0

            def encode(self, textos, **kwargs):
                return np.ones((len(textos), 2), dtype=np.float32)

        fake_st_mod.SentenceTransformer = FakeST
        passagens = ["p1", "p2"]
        consultas = [("c1", 0)]
        with patch.dict(sys.modules, {"sentence_transformers": fake_st_mod}), \
             patch("acesso_hf.preparar_acesso_hub", return_value=True):
            benchmark.metodo_sentence(passagens, consultas, 1, rag.MODELO_EMB)
            benchmark.metodo_sentence(passagens, consultas, 1, "outro/modelo")
            benchmark.metodo_sentence(passagens, consultas, 1, rag.MODELO_EMB, revisao="")
            benchmark.metodo_sentence(passagens, consultas, 1, "outro/modelo", revisao="   ")

        self.assertEqual(chamadas[0][1]["revision"], rag.REVISAO_EMB)
        self.assertTrue(chamadas[0][1]["local_files_only"])
        self.assertNotIn("revision", chamadas[1][1])
        self.assertEqual(chamadas[2][1]["revision"], rag.REVISAO_EMB)
        self.assertNotIn("revision", chamadas[3][1])

    def test_benchmark_preserva_offline_entre_execucoes(self):
        fake_st_mod = types.ModuleType("sentence_transformers")
        chamadas_st = []

        class FakeST:
            def __init__(self, *args, **kwargs):
                chamadas_st.append(kwargs["local_files_only"])
                if not kwargs["local_files_only"]:
                    raise httpx.ConnectError("sem rede")
                self.max_seq_length = 0

            def encode(self, textos, **kwargs):
                return np.ones((len(textos), 2), dtype=np.float32)

        fake_st_mod.SentenceTransformer = FakeST
        passagens = ["p1", "p2"]
        consultas = [("c1", 0)]

        with patch.dict(sys.modules, {"sentence_transformers": fake_st_mod}), \
             patch("infraestrutura.huggingface.HF_HUB_OFFLINE", False), \
             patch("acesso_hf.preparar_acesso_hub", return_value=False) as preparar:
            with self.assertLogs("busca.hub", level="WARNING"):
                *_, offline = benchmark.metodo_sentence(
                    passagens, consultas, 1, rag.MODELO_EMB,
                    offline=None,
                )
            self.assertTrue(offline)
            *_, offline_seguinte = benchmark.metodo_sentence(
                passagens, consultas, 1, rag.MODELO_EMB,
                offline=offline,
            )
            self.assertTrue(offline_seguinte)
            preparar.assert_called_once()
        self.assertEqual(chamadas_st, [False, True, True])

        chamadas_st.clear()
        with patch.dict(sys.modules, {"sentence_transformers": fake_st_mod}), \
             patch("infraestrutura.huggingface.HF_HUB_OFFLINE", False), \
             patch("acesso_hf.preparar_acesso_hub", return_value=False) as preparar, \
             patch.object(benchmark, "amostrar_passagens", return_value=passagens), \
             patch.object(benchmark, "criar_consultas", return_value=consultas), \
             patch.object(benchmark, "metodo_tfidf", return_value=([[0]], {})), \
             patch.object(benchmark, "metodo_word2vec", return_value=([[0]], {})), \
             patch.object(benchmark.argparse, "ArgumentParser") as parser:
            argumentos = parser.return_value.parse_args.return_value
            argumentos.tamanhos = [10, 20]
            argumentos.consultas = 1
            argumentos.k = 1
            argumentos.semente = 42
            argumentos.modelo = rag.MODELO_EMB
            argumentos.revisao = None
            argumentos.saida = Mock()
            with self.assertLogs("busca.hub", level="WARNING"):
                benchmark.main()

        preparar.assert_called_once()
        self.assertEqual(chamadas_st, [False, True, True])
        salvo = json.loads(argumentos.saida.write_text.call_args.args[0])
        self.assertEqual(len(salvo), 2)


if __name__ == "__main__":
    unittest.main()
