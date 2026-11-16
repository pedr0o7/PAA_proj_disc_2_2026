import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import rag


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
            sistema.modelo_emb.encode.return_value = np.array([[1.0, 2.0]], dtype=np.float32)
            sistema.passagens = [{"texto": "um", "titulo": "t", "url": "u"}]
            sistema._embeddings()
            caminho = next(Path(diretorio).glob("*.npz"))
            with np.load(caminho, allow_pickle=False) as registro:
                metadados = registro["metadados"]
            np.savez(caminho, metadados=metadados, embeddings=np.zeros((2, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "embeddings inválido ou incompatível"):
                sistema._embeddings()


if __name__ == "__main__":
    unittest.main()
