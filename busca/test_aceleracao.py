import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from dotenv import dotenv_values

import rag
from infraestrutura import aceleracao


class TestConfiguracaoAceleracao(unittest.TestCase):
    def test_env_example_define_variaveis_portabilidade_com_padrao_auto(self):
        exemplo = Path(__file__).resolve().parent.parent / ".env.example"
        valores = dotenv_values(exemplo)
        for chave in ("EMB_BACKEND", "LLM_BACKEND", "DISPOSITIVO_IA", "LLM_QUANTIZACAO"):
            self.assertEqual(valores.get(chave), "auto")
        self.assertEqual(valores.get("LLM_MODEL"), "")

    def test_ler_config_env_normaliza_e_trata_valores_invalidos(self):
        with self.assertLogs("busca.aceleracao", level="WARNING"):
            cfg = aceleracao.ler_config_env({
                "EMB_BACKEND": "  ONNX ",
                "LLM_BACKEND": "invalido",
                "DISPOSITIVO_IA": "gpu",
                "LLM_QUANTIZACAO": "INT8",
            })
        self.assertEqual(cfg["emb_backend"], "onnx")
        self.assertEqual(cfg["llm_backend"], "auto")
        self.assertEqual(cfg["dispositivo"], "GPU")
        self.assertEqual(cfg["llm_quantizacao"], "int8")

    def test_auto_intel_com_openvino_gpu(self):
        caps = {
            "cpu_vendor": "intel",
            "openvino": True,
            "openvino_gpu": True,
            "nncf": True,
            "onnxruntime": True,
            "optimum_onnx": True,
            "onnx_gpu_provider": None,
            "torch_device": "cpu",
            "bitsandbytes": False,
        }
        cfg = aceleracao.resolver_config_execucao(env={}, capacidades=caps, usar_llm=True)
        self.assertEqual(cfg["emb_backend"], "openvino")
        self.assertEqual(cfg["emb_dispositivo"], "GPU")
        self.assertEqual(cfg["llm_backend"], "openvino")
        self.assertEqual(cfg["llm_dispositivo"], "GPU")
        self.assertEqual(cfg["llm_quantizacao"], "int8")

        cfg_int4 = aceleracao.resolver_config_execucao(
            env={"LLM_QUANTIZACAO": "int4"}, capacidades=caps, usar_llm=True
        )
        self.assertEqual(cfg_int4["llm_quantizacao"], "int4")

        cfg_fp16 = aceleracao.resolver_config_execucao(
            env={"LLM_QUANTIZACAO": "fp16"}, capacidades=caps, usar_llm=True
        )
        self.assertEqual(cfg_fp16["llm_quantizacao"], "fp16")

    def test_auto_amd_com_onnx_e_directml(self):
        caps = {
            "cpu_vendor": "amd",
            "openvino": False,
            "openvino_gpu": False,
            "nncf": False,
            "onnxruntime": True,
            "optimum_onnx": True,
            "onnx_gpu_provider": "DmlExecutionProvider",
            "torch_device": "cpu",
            "bitsandbytes": False,
        }
        cfg = aceleracao.resolver_config_execucao(env={}, capacidades=caps, usar_llm=True)
        self.assertEqual(cfg["emb_backend"], "onnx")
        self.assertEqual(cfg["emb_dispositivo"], "GPU")
        self.assertEqual(cfg["emb_provider"], "DmlExecutionProvider")
        self.assertEqual(cfg["llm_backend"], "onnx")
        self.assertEqual(cfg["llm_dispositivo"], "GPU")

    def test_auto_cpu_only_sem_bibliotecas_opcionais_usa_torch(self):
        caps = {
            "cpu_vendor": "desconhecido",
            "openvino": False,
            "openvino_gpu": False,
            "nncf": False,
            "onnxruntime": False,
            "optimum_onnx": False,
            "onnx_gpu_provider": None,
            "torch_device": "cpu",
            "bitsandbytes": False,
        }
        cfg = aceleracao.resolver_config_execucao(env={}, capacidades=caps, usar_llm=True)
        self.assertEqual(cfg["emb_backend"], "torch")
        self.assertEqual(cfg["emb_dispositivo"], "CPU")
        self.assertEqual(cfg["llm_backend"], "torch")
        self.assertEqual(cfg["llm_dispositivo"], "CPU")
        self.assertEqual(cfg["llm_quantizacao"], "fp32")

    def test_backend_explicito_indisponivel_faz_fallback_para_torch(self):
        caps = {
            "cpu_vendor": "amd",
            "openvino": False,
            "openvino_gpu": False,
            "nncf": False,
            "onnxruntime": False,
            "optimum_onnx": False,
            "onnx_gpu_provider": None,
            "torch_device": "cpu",
            "bitsandbytes": False,
        }
        cfg = aceleracao.resolver_config_execucao(
            env={"EMB_BACKEND": "openvino", "LLM_BACKEND": "onnx", "DISPOSITIVO_IA": "GPU"},
            capacidades=caps,
            usar_llm=True,
        )
        self.assertEqual(cfg["emb_backend"], "torch")
        self.assertEqual(cfg["llm_backend"], "torch")
        self.assertEqual(cfg["llm_dispositivo"], "CPU")
        self.assertGreaterEqual(len(cfg["fallbacks"]), 2)

    def test_carregar_embeddings_faz_fallback_silencioso_quando_backend_falha_em_runtime(self):
        chamadas = []

        def fake_st(nome, **kwargs):
            chamadas.append(kwargs)
            if kwargs.get("backend") == "onnx":
                raise RuntimeError("onnxruntime falhou")
            return Mock()

        cfg = {
            "emb_backend": "onnx",
            "emb_dispositivo": "CPU",
            "emb_provider": "CPUExecutionProvider",
            "torch_device": "cpu",
            "fallbacks": [],
        }

        def fake_carregar(desc, fn, *args, retornar_offline=False, **kwargs):
            res = fn(*args, **kwargs)
            return (res, False) if retornar_offline else res

        with self.assertLogs("busca.aceleracao", level="WARNING"):
            modelo, offline = aceleracao.carregar_modelo_embeddings(
                "modelo/teste", "rev1", False, cfg, carregar_fn=fake_carregar, st_cls=fake_st
            )
        self.assertIsNotNone(modelo)
        self.assertFalse(offline)
        self.assertEqual(cfg["emb_backend"], "torch")
        self.assertEqual(cfg["emb_dispositivo"], "CPU")
        self.assertEqual(len(chamadas), 2)
        self.assertEqual(chamadas[0]["backend"], "onnx")
        self.assertEqual(chamadas[1]["device"], "cpu")


class TestNormalizacaoCacheEDominio(unittest.TestCase):
    def test_normalizar_vetores_garante_norma_unitaria_e_float32(self):
        v = aceleracao.normalizar_vetores([[3.0, 4.0], [0.0, 0.0]])
        self.assertEqual(v.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(v[0])), 1.0, places=5)
        np.testing.assert_array_equal(v[1], np.array([0.0, 0.0], dtype=np.float32))

    def test_cache_embeddings_preserva_metadados_torch_e_isola_backend_alternativo(self):
        with tempfile.TemporaryDirectory() as diretorio, patch.object(rag, "CACHE", Path(diretorio)):
            sistema = rag.SistemaRAG.__new__(rag.SistemaRAG)
            sistema.modelo_emb = Mock()
            sistema.modelo_emb.get_sentence_embedding_dimension.return_value = 2
            sistema.modelo_emb.encode.return_value = np.array([[3.0, 4.0]], dtype=np.float32)
            sistema.passagens = [{"texto": "um", "titulo": "t", "autores": "", "url": "u"}]

            sistema.config_execucao = {"emb_backend": "torch"}
            emb_torch = sistema._embeddings()
            self.assertAlmostEqual(float(np.linalg.norm(emb_torch[0])), 1.0, places=5)

            sistema.config_execucao = {"emb_backend": "onnx"}
            sistema._embeddings()

            arquivos = sorted(Path(diretorio).glob("embeddings_v1_*.npz"))
            self.assertEqual(len(arquivos), 2)
            metas = []
            for arq in arquivos:
                with np.load(arq, allow_pickle=False) as reg:
                    metas.append(json.loads(str(reg["metadados"].item())))
            self.assertTrue(any("backend" not in m for m in metas))
            self.assertTrue(any(m.get("backend") == "onnx" for m in metas))

    def test_bm25_argpartition_produz_mesmo_ranking_que_argsort(self):
        sistema = rag.SistemaRAG.__new__(rag.SistemaRAG)
        sistema.indice_lexical = None
        sistema.k = 2
        sistema.passagens = [
            {"id": i, "titulo": f"Book {i}", "autores": "Author", "texto": f"passage {i}", "url": f"u{i}"}
            for i in range(12)
        ]
        emb = aceleracao.normalizar_vetores(np.eye(12, 4, dtype=np.float32) + 0.1)
        sistema.emb = emb
        sistema._vetor = lambda p: emb[3]

        fake_indice = Mock()
        fake_indice.search.return_value = Mock(keys=np.array([0, 3, 5, 1, 2, 4]))
        sistema.indice = fake_indice

        scores = np.array([0.1, 0.4, 0.2, 0.95, 0.3, 0.8, 0.0, 0.5, 0.7, 0.6, 0.15, 0.25], dtype=np.float32)
        fake_bm25 = Mock()
        fake_bm25.get_scores.return_value = scores
        sistema.bm25 = fake_bm25

        achados = sistema.recuperar("query")
        self.assertEqual([a["id"] for a in achados], [3, 5])

    def test_prompt_sistema_aderente_ao_project_gutenberg(self):
        self.assertIn("Project Gutenberg", rag.PROMPT_SISTEMA)
        self.assertIn(rag.SEM_CONTEXTO, rag.PROMPT_SISTEMA)
        self.assertIn("[1]", rag.PROMPT_SISTEMA)


class TestWrappersOpenVINO(unittest.TestCase):
    def test_modelo_embeddings_openvino_mean_pooling_e_normalizacao(self):
        fake_compiled = Mock()
        fake_inp_ids = Mock()
        fake_inp_ids.any_name = "input_ids"
        fake_inp_mask = Mock()
        fake_inp_mask.any_name = "attention_mask"
        fake_compiled.inputs = [fake_inp_ids, fake_inp_mask]
        fake_compiled.return_value = {0: np.array([[[3.0, 0.0], [0.0, 4.0]]], dtype=np.float32)}

        def fake_tok(lote, **kwargs):
            return {
                "input_ids": np.array([[10, 20]], dtype=np.int64),
                "attention_mask": np.array([[1, 1]], dtype=np.int64),
            }

        modelo = aceleracao.ModeloEmbeddingsOpenVINO(fake_compiled, fake_tok, dimensao=2)
        self.assertEqual(modelo.get_sentence_embedding_dimension(), 2)
        vet = modelo.encode("texto", normalize_embeddings=True)
        self.assertEqual(vet.shape, (2,))
        self.assertEqual(vet.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(vet)), 1.0, places=5)
        np.testing.assert_allclose(vet, np.array([0.6, 0.8], dtype=np.float32), atol=1e-5)

    def test_modelo_llm_openvino_genai_generate_compat_com_completar(self):
        fake_tok = Mock()
        fake_tok.decode.return_value = "Prompt"
        fake_tok.encode.return_value = [101, 102]
        fake_pipe = Mock()
        fake_pipe.generate.return_value = "Resposta [1]"
        fake_cfg = Mock(apply_chat_template=True)
        fake_pipe.get_generation_config.return_value = fake_cfg

        with patch.dict("sys.modules", {"openvino_genai": None}):
            lm = aceleracao.ModeloLLMOpenVINOGenAI("pasta", "GPU", fake_tok, pipe=fake_pipe)
            entrada = torch.tensor([[1, 2, 3]], dtype=torch.long)
            saida = lm.generate(input_ids=entrada, max_new_tokens=50, do_sample=False, repetition_penalty=1.1)
        self.assertEqual(saida.shape, (1, 5))
        self.assertEqual(saida[0, 3:].tolist(), [101, 102])
        self.assertFalse(fake_cfg.apply_chat_template)

    def test_exportar_llm_openvino_reutiliza_cache_e_exige_em_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            destino = Path(tmp) / "ov_modelo"
            with self.assertRaises(FileNotFoundError):
                aceleracao.exportar_llm_openvino("m/teste", "int4", destino=destino, offline=True)
            destino.mkdir()
            (destino / "openvino_model.xml").write_text("<xml/>", encoding="utf-8")
            (destino / "openvino_model.bin").write_bytes(b"bin")
            self.assertEqual(
                aceleracao.exportar_llm_openvino("m/teste", "int4", destino=destino, offline=True),
                destino,
            )

    def test_mean_pooling_respeita_attention_mask_com_padding_e_bate_com_torch(self):
        token_emb = np.array([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[4.0, 3.0], [999.0, -999.0], [-999.0, 999.0]],
        ], dtype=np.float32)
        mask = np.array([[1, 1, 1], [1, 0, 0]], dtype=np.int64)
        ids = np.array([[1, 2, 3], [4, 0, 0]], dtype=np.int64)

        def fake_tok(lote, **kwargs):
            return {"input_ids": ids, "attention_mask": mask}

        # Referência exata em PyTorch
        t_emb = torch.from_numpy(token_emb)
        t_mask = torch.from_numpy(mask).unsqueeze(-1).float()
        ref_torch = torch.nn.functional.normalize((t_emb * t_mask).sum(dim=1) / t_mask.sum(dim=1), p=2, dim=1).numpy()

        # Wrapper OpenVINO
        fake_ov = Mock()
        inp_ids, inp_mask = Mock(any_name="input_ids"), Mock(any_name="attention_mask")
        fake_ov.inputs = [inp_ids, inp_mask]
        fake_ov.return_value = {0: token_emb}
        modelo_ov = aceleracao.ModeloEmbeddingsOpenVINO(fake_ov, fake_tok, dimensao=2)

        # Wrapper ONNX
        fake_ort = Mock()
        out_meta = Mock(shape=[None, None, 2])
        inp_ort_ids, inp_ort_mask = Mock(), Mock()
        inp_ort_ids.name = "input_ids"
        inp_ort_mask.name = "attention_mask"
        fake_ort.get_inputs.return_value = [inp_ort_ids, inp_ort_mask]
        fake_ort.get_outputs.return_value = [out_meta]
        fake_ort.run.return_value = [token_emb]
        modelo_onnx = aceleracao.ModeloEmbeddingsONNX(fake_ort, fake_tok, dimensao=2)

        frases = ["Captain Ahab pursued the white whale", "Pride and Prejudice"]
        for wrapper in (modelo_ov, modelo_onnx):
            vet = wrapper.encode(frases, normalize_embeddings=True)
            self.assertEqual(vet.dtype, np.float32)
            self.assertEqual(vet.shape, (2, 2))
            sims = np.sum(vet * ref_torch, axis=1)
            self.assertGreaterEqual(float(sims[0]), 0.999999)
            self.assertGreaterEqual(float(sims[1]), 0.999999)

    def test_wrappers_integram_com_cache_embeddings_e_completar_do_rag(self):
        fake_ov = Mock()
        inp_ids, inp_mask = Mock(any_name="input_ids"), Mock(any_name="attention_mask")
        fake_ov.inputs = [inp_ids, inp_mask]
        fake_ov.return_value = {0: np.array([[[3.0, 4.0], [3.0, 4.0]]], dtype=np.float32)}

        def fake_tok_emb(lote, **kwargs):
            return {
                "input_ids": np.array([[10, 20]], dtype=np.int64),
                "attention_mask": np.array([[1, 1]], dtype=np.int64),
            }

        modelo_emb = aceleracao.ModeloEmbeddingsOpenVINO(fake_ov, fake_tok_emb, dimensao=2)
        with tempfile.TemporaryDirectory() as diretorio, patch.object(rag, "CACHE", Path(diretorio)):
            sistema = rag.SistemaRAG.__new__(rag.SistemaRAG)
            sistema.modelo_emb = modelo_emb
            sistema.config_execucao = {"emb_backend": "openvino"}
            sistema.passagens = [{
                "id": 2701, "titulo": "Moby-Dick", "autores": "Melville, Herman",
                "url": "https://www.gutenberg.org/ebooks/2701", "texto": "Call me Ishmael.",
            }]
            emb1 = sistema._embeddings()
            emb2 = sistema._embeddings()
            self.assertEqual(emb1.shape, (1, 2))
            self.assertEqual(emb1.dtype, np.float32)
            self.assertAlmostEqual(float(np.linalg.norm(emb1[0])), 1.0, places=5)
            np.testing.assert_array_equal(emb1, emb2)

        fake_tok_lm = Mock()
        fake_tok_lm.apply_chat_template.return_value = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
        fake_tok_lm.decode.side_effect = ["Prompt", "Captain Ahab commands the Pequod. [1]"]
        fake_tok_lm.encode.return_value = [101, 102, 103]
        fake_pipe = Mock()
        fake_pipe.generate.return_value = "Captain Ahab commands the Pequod. [1]"

        sistema.tok = fake_tok_lm
        sistema.lm = aceleracao.ModeloLLMOpenVINOGenAI("pasta", "GPU", fake_tok_lm, pipe=fake_pipe)
        resposta = sistema.completar([{"role": "user", "content": "Who is Captain Ahab?"}], 50)
        self.assertEqual(resposta, "Captain Ahab commands the Pequod. [1]")

    def test_carregar_llm_faz_fallback_silencioso_para_torch_e_registra_diagnostico(self):
        cfg = {
            "emb_backend": "openvino",
            "emb_dispositivo": "GPU",
            "llm_backend": "openvino",
            "llm_dispositivo": "GPU",
            "llm_quantizacao": "int4",
            "torch_device": "cpu",
            "fallbacks": [],
        }
        fake_lm = Mock()
        fake_lm.eval.return_value = fake_lm
        fake_auto = Mock()
        fake_auto.from_pretrained.return_value = fake_lm

        def fake_carregar(desc, fn, *args, retornar_offline=False, **kwargs):
            res = fn(*args, **kwargs)
            return (res, False) if retornar_offline else res

        with patch.object(aceleracao, "exportar_llm_openvino", side_effect=RuntimeError("IR indisponível")), \
             self.assertLogs("busca.aceleracao", level="WARNING"):
            lm, offline = aceleracao.carregar_modelo_llm(
                "HuggingFaceTB/SmolLM2-1.7B-Instruct", False, cfg,
                carregar_fn=fake_carregar, auto_model_cls=fake_auto,
            )

        self.assertIs(lm, fake_lm)
        self.assertFalse(offline)
        self.assertEqual(cfg["llm_backend"], "torch")
        self.assertEqual(cfg["llm_dispositivo"], "CPU")
        self.assertEqual(cfg["llm_quantizacao"], "fp32")
        self.assertEqual(len(cfg["fallbacks"]), 1)
        self.assertIn("LLM=torch/CPU(fp32)", aceleracao.resumo_execucao(cfg))

    def test_auto_openvino_reutiliza_cache_int4_se_int8_ausente(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            pasta_int8 = raiz / "ov_int8"
            pasta_int4 = raiz / "ov_int4"
            pasta_int4.mkdir()
            (pasta_int4 / "openvino_model.xml").write_text("<xml/>", encoding="utf-8")
            (pasta_int4 / "openvino_model.bin").write_bytes(b"bin")
            cfg = {
                "emb_backend": "openvino", "emb_dispositivo": "GPU",
                "llm_backend": "openvino", "llm_dispositivo": "GPU",
                "llm_quantizacao": "int8", "torch_device": "cpu",
                "solicitado": {"llm_quantizacao": "auto"}, "fallbacks": [],
            }
            fake_tok = Mock()
            with patch.object(aceleracao, "_modulo_disponivel", return_value=True), \
                 patch.object(aceleracao, "caminho_cache_llm_openvino",
                              side_effect=lambda n, q: pasta_int8 if q == "int8" else pasta_int4), \
                 patch.object(aceleracao, "exportar_llm_openvino", return_value=pasta_int4) as exp, \
                 patch.object(aceleracao, "ModeloLLMOpenVINOGenAI", return_value=Mock(eval=lambda: "ok")) as cls, \
                 patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok):
                lm, _ = aceleracao.carregar_modelo_llm("HuggingFaceTB/SmolLM2-1.7B-Instruct", True, cfg)
            self.assertEqual(lm, "ok")
            self.assertEqual(cfg["llm_quantizacao"], "int4")
            exp.assert_called_once_with("HuggingFaceTB/SmolLM2-1.7B-Instruct", quantizacao="int4", offline=True)
            cls.assert_called_once()


class TestCorrecoesRevisaoPR(unittest.TestCase):
    def test_openvino_gpu_respeita_fp32_explicito_e_onnx_reporta_fp32(self):
        caps_ov = {
            "cpu_vendor": "intel", "openvino": True, "openvino_gpu": True, "nncf": True,
            "onnxruntime": False, "optimum_onnx": False, "onnx_gpu_provider": None,
            "torch_device": "cpu", "bitsandbytes": False,
        }
        cfg_ov = aceleracao.resolver_config_execucao(
            env={"LLM_BACKEND": "openvino", "DISPOSITIVO_IA": "GPU", "LLM_QUANTIZACAO": "fp32"},
            capacidades=caps_ov, usar_llm=True,
        )
        self.assertEqual(cfg_ov["llm_quantizacao"], "fp32")

        caps_onnx = {
            "cpu_vendor": "amd", "openvino": False, "openvino_gpu": False, "nncf": False,
            "onnxruntime": True, "optimum_onnx": True, "onnx_gpu_provider": "DmlExecutionProvider",
            "torch_device": "cpu", "bitsandbytes": False,
        }
        cfg_onnx = aceleracao.resolver_config_execucao(
            env={"LLM_BACKEND": "onnx", "DISPOSITIVO_IA": "GPU", "LLM_QUANTIZACAO": "fp16"},
            capacidades=caps_onnx, usar_llm=True,
        )
        self.assertEqual(cfg_onnx["llm_quantizacao"], "fp32")
        self.assertTrue(any("fp16" in f and "ONNX" in f for f in cfg_onnx["fallbacks"]))

    def test_onnx_directml_configura_session_options_sequencial(self):
        fake_ort = Mock()
        fake_opts = Mock()
        fake_ort.SessionOptions.return_value = fake_opts
        fake_ort.ExecutionMode.ORT_SEQUENTIAL = "SEQ"
        fake_sess = Mock()
        inp_ids, inp_mask = Mock(), Mock()
        inp_ids.name, inp_mask.name = "input_ids", "attention_mask"
        fake_sess.get_inputs.return_value = [inp_ids, inp_mask]
        fake_sess.get_outputs.return_value = [Mock(shape=[None, None, 2])]
        fake_ort.InferenceSession.return_value = fake_sess

        def fake_carregar(desc, fn, *args, retornar_offline=False, **kwargs):
            return (Mock() if fn is not None else None), False

        with patch.dict("sys.modules", {"onnxruntime": fake_ort}), \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=Mock()):
            modelo, _ = aceleracao.ModeloEmbeddingsONNX.from_pretrained(
                "m/emb", provider="DmlExecutionProvider", offline=False, carregar_fn=fake_carregar
            )
        self.assertIsInstance(modelo, aceleracao.ModeloEmbeddingsONNX)
        self.assertFalse(fake_opts.enable_mem_pattern)
        self.assertEqual(fake_opts.execution_mode, "SEQ")
        self.assertIn("sess_options", fake_ort.InferenceSession.call_args.kwargs)

    def test_exportar_llm_openvino_isola_tmp_e_respeita_publicacao_concorrente(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            destino = Path(tmpdir) / "ov_modelo"

            def simular_concorrencia(cmd, check=True):
                tmp_ir = Path(cmd[-1])
                tmp_ir.mkdir(parents=True, exist_ok=True)
                (tmp_ir / "openvino_model.xml").write_text("<tmp/>", encoding="utf-8")
                (tmp_ir / "openvino_model.bin").write_bytes(b"tmp")
                # Simula outro processo publicando o destino válido antes do rename
                destino.mkdir(parents=True, exist_ok=True)
                (destino / "openvino_model.xml").write_text("<vencedor/>", encoding="utf-8")
                (destino / "openvino_model.bin").write_bytes(b"vencedor")

            with patch.object(aceleracao.shutil, "which", return_value="uv"), \
                 patch.object(aceleracao.subprocess, "run", side_effect=simular_concorrencia):
                res = aceleracao.exportar_llm_openvino("m/teste", "int8", destino=destino, offline=False)

            self.assertEqual(res, destino)
            self.assertEqual((destino / "openvino_model.xml").read_text(encoding="utf-8"), "<vencedor/>")
            restantes = [p.name for p in Path(tmpdir).iterdir()]
            self.assertEqual(restantes, ["ov_modelo"])

    def test_embeddings_404_faz_fallback_para_torch_e_401_propaga(self):
        from huggingface_hub.errors import HfHubHTTPError

        err_404 = HfHubHTTPError("404 Not Found", response=Mock(status_code=404))
        err_401 = HfHubHTTPError("401 Unauthorized", response=Mock(status_code=401))

        def fake_st(nome, **kwargs):
            return Mock(name="torch_emb")

        fake_st.__name__ = "SentenceTransformer"

        def fake_carregar(desc, fn, *args, retornar_offline=False, **kwargs):
            return fn(*args, **kwargs), False

        for backend in ("openvino", "onnx"):
            cfg = {
                "emb_backend": backend, "emb_dispositivo": "GPU",
                "emb_provider": "CPUExecutionProvider", "torch_device": "cpu", "fallbacks": [],
            }
            cls_alvo = aceleracao.ModeloEmbeddingsOpenVINO if backend == "openvino" else aceleracao.ModeloEmbeddingsONNX
            with patch.object(cls_alvo, "from_pretrained", side_effect=err_404), \
                 self.assertLogs("busca.aceleracao", level="WARNING"):
                modelo, _ = aceleracao.carregar_modelo_embeddings(
                    "m/sem-ir", "rev", False, cfg, carregar_fn=fake_carregar, st_cls=fake_st
                )
            self.assertEqual(cfg["emb_backend"], "torch")
            self.assertIsNotNone(modelo)

            cfg_auth = {
                "emb_backend": backend, "emb_dispositivo": "GPU",
                "emb_provider": "CPUExecutionProvider", "torch_device": "cpu", "fallbacks": [],
            }
            with patch.object(cls_alvo, "from_pretrained", side_effect=err_401):
                with self.assertRaises(HfHubHTTPError):
                    aceleracao.carregar_modelo_embeddings(
                        "m/privado", "rev", False, cfg_auth, carregar_fn=fake_carregar, st_cls=fake_st
                    )

    def test_torch_cuda_int8_usa_bitsandbytes_e_fallback_converte_para_cpu_fp32(self):
        cfg_bnb = {
            "llm_backend": "torch", "llm_dispositivo": "GPU",
            "llm_quantizacao": "int8", "torch_device": "cuda", "fallbacks": [],
        }
        fake_bnb_cls = Mock()
        fake_lm = Mock()
        fake_lm.eval.return_value = fake_lm
        fake_auto = Mock()
        fake_auto.from_pretrained.return_value = fake_lm

        def fake_carregar(desc, fn, *args, retornar_offline=False, **kwargs):
            return fn(*args, **kwargs), False

        with patch("transformers.BitsAndBytesConfig", fake_bnb_cls):
            lm, _ = aceleracao.carregar_modelo_llm(
                "m/llm", False, cfg_bnb, carregar_fn=fake_carregar, auto_model_cls=fake_auto
            )
        self.assertIs(lm, fake_lm)
        fake_bnb_cls.assert_called_once_with(load_in_8bit=True)

        # Falha em .to("cuda") deve converter efetivamente para CPU/fp32
        cfg_fb = {
            "llm_backend": "torch", "llm_dispositivo": "GPU",
            "llm_quantizacao": "fp16", "torch_device": "cuda", "fallbacks": [],
        }
        lm_gpu = Mock()
        lm_cpu = Mock()
        lm_cpu.eval.return_value = lm_cpu

        def fake_to(*args, **kwargs):
            if args == ("cuda",):
                raise RuntimeError("CUDA OOM")
            return lm_cpu

        lm_gpu.to.side_effect = fake_to
        fake_auto_fb = Mock()
        fake_auto_fb.from_pretrained.return_value = lm_gpu
        with self.assertLogs("busca.aceleracao", level="WARNING"):
            lm_res, _ = aceleracao.carregar_modelo_llm(
                "m/llm", False, cfg_fb, carregar_fn=fake_carregar, auto_model_cls=fake_auto_fb
            )
        self.assertIs(lm_res, lm_cpu)
        lm_gpu.to.assert_any_call("cpu", dtype=torch.float32)
        self.assertEqual(cfg_fb["llm_dispositivo"], "CPU")
        self.assertEqual(cfg_fb["llm_quantizacao"], "fp32")

    def test_caminho_cache_ir_evita_colisao_de_slug_e_diferencia_revisao(self):
        c1 = aceleracao.caminho_cache_llm_openvino("a/b_c", "int8")
        c2 = aceleracao.caminho_cache_llm_openvino("a_b/c", "int8")
        c3 = aceleracao.caminho_cache_llm_openvino("a/b_c", "int8", revisao="rev-1")
        c4 = aceleracao.caminho_cache_llm_openvino("a/b_c", "int8", revisao="rev-2")
        self.assertNotEqual(c1, c2)
        self.assertNotEqual(c1, c3)
        self.assertNotEqual(c3, c4)

    def test_exportar_llm_openvino_grava_metadados_e_reexporta_se_revisao_divergir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            destino = Path(tmpdir) / "ov_modelo"
            destino.mkdir()
            (destino / "openvino_model.xml").write_text("<antigo/>", encoding="utf-8")
            (destino / "openvino_model.bin").write_bytes(b"antigo")
            (destino / "metadados_ir.json").write_text(
                json.dumps({"modelo": "m/teste", "quantizacao": "int8", "revisao": "rev-1"}),
                encoding="utf-8",
            )

            cmds = []

            def fake_run(cmd, check=True):
                cmds.append(cmd)
                tmp_ir = Path(cmd[-1])
                tmp_ir.mkdir(parents=True, exist_ok=True)
                (tmp_ir / "openvino_model.xml").write_text("<novo/>", encoding="utf-8")
                (tmp_ir / "openvino_model.bin").write_bytes(b"novo")

            with patch.object(aceleracao.shutil, "which", return_value="uv"), \
                 patch.object(aceleracao.subprocess, "run", side_effect=fake_run):
                res = aceleracao.exportar_llm_openvino(
                    "m/teste", "int8", destino=destino, offline=False, revisao="rev-2"
                )
            self.assertEqual(res, destino)
            self.assertIn("--revision", cmds[0])
            self.assertIn("rev-2", cmds[0])
            meta = json.loads((destino / "metadados_ir.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["revisao"], "rev-2")

    def test_avaliar_ragas_aceita_wrapper_openvino_genai_sem_transformers_pipeline(self):
        import types
        import sys
        import avaliar

        fake_tok = Mock()
        fake_tok.apply_chat_template.return_value = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
        fake_tok.decode.side_effect = ["Prompt", '```json\n{"verdict": 1}\n```']
        fake_pipe = Mock()
        fake_pipe.generate.return_value = '```json\n{"verdict": 1}\n```'
        fake_tok.encode.return_value = [10, 11]
        lm_ov = aceleracao.ModeloLLMOpenVINOGenAI("pasta", "GPU", fake_tok, pipe=fake_pipe)

        lc_llm = avaliar._criar_llm_langchain_openvino(fake_tok, lm_ov)
        self.assertEqual(lc_llm._llm_type, "openvino_genai")
        self.assertEqual(lc_llm._call("Evaluate"), '{"verdict": 1}')

        sistema = Mock(offline=False, tok=fake_tok, lm=lm_ov, config_execucao={"llm_modelo": "Qwen/Qwen2.5-1.5B-Instruct"})
        itens = [{"pergunta": "p", "referencia": "r", "titulo_esperado": "t"}]
        resultados = [{"resposta": "resp [1]", "trechos": [{"texto": "ctx"}]}]
        df = Mock()
        df.columns = ["faithfulness"]
        col = Mock()
        col.dropna.return_value = Mock(empty=False)
        col.mean.return_value = 0.8
        col.isna.return_value = Mock(sum=Mock(return_value=0))
        df.__getitem__ = Mock(return_value=col)

        fake_lc_hf = types.ModuleType("langchain_huggingface")
        fake_lc_hf.HuggingFaceEmbeddings = lambda **kwargs: Mock()
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
        }), patch("transformers.pipeline", side_effect=AssertionError("não deveria chamar pipeline")):
            scores, nans = avaliar.avaliar_ragas(sistema, itens, resultados, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(scores["faithfulness"], 0.8)
        self.assertEqual(nans["faithfulness"], 0)


if __name__ == "__main__":
    unittest.main()
