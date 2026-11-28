"""Detecção de hardware, resolução de backends (.env), wrappers nativos e fallback seguro para PyTorch."""
import argparse
import importlib.util
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from infraestrutura.configuracao_hf import configurar_huggingface
from infraestrutura.huggingface import carregar_modelo

LOG = logging.getLogger("busca.aceleracao")
RAIZ = Path(__file__).resolve().parent.parent

BACKENDS_VALIDOS = ("auto", "openvino", "onnx", "torch")
DISPOSITIVOS_VALIDOS = ("auto", "CPU", "GPU")
QUANTIZACOES_VALIDAS = ("auto", "fp32", "fp16", "int8", "int4")

ONNX_GPU_PROVIDERS = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "ROCMExecutionProvider",
    "OpenVINOExecutionProvider",
)


def _modulo_disponivel(nome):
    try:
        return importlib.util.find_spec(nome) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _normalizar_opcao(valor, validos, padrao="auto", maiusculas=False, nome_var=""):
    bruto = (valor or "").strip()
    if not bruto:
        return padrao
    canonico = bruto.upper() if maiusculas else bruto.lower()
    if canonico not in validos:
        LOG.warning("Valor inválido para %s=%r; usando %r.", nome_var or "opção", valor, padrao)
        return padrao
    return canonico


def ler_config_env(env=None):
    """Lê e valida as variáveis de portabilidade de hardware após carregar o .env."""
    if env is None:
        configurar_huggingface()
        env = os.environ
    return {
        "emb_backend": _normalizar_opcao(env.get("EMB_BACKEND"), BACKENDS_VALIDOS, nome_var="EMB_BACKEND"),
        "llm_backend": _normalizar_opcao(env.get("LLM_BACKEND"), BACKENDS_VALIDOS, nome_var="LLM_BACKEND"),
        "dispositivo": _normalizar_opcao(
            env.get("DISPOSITIVO_IA"), DISPOSITIVOS_VALIDOS, maiusculas=True, nome_var="DISPOSITIVO_IA"
        ),
        "llm_quantizacao": _normalizar_opcao(
            env.get("LLM_QUANTIZACAO"), QUANTIZACOES_VALIDAS, nome_var="LLM_QUANTIZACAO"
        ),
    }


def detectar_fabricante_cpu():
    """Identifica o fabricante da CPU ('intel', 'amd' ou 'desconhecido') de forma portável."""
    textos = [
        platform.processor() or "",
        os.environ.get("PROCESSOR_IDENTIFIER", ""),
    ]
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            textos.append(cpuinfo.read_text(encoding="utf-8", errors="ignore")[:4096])
        except OSError:
            pass
    combinado = " ".join(textos).lower()
    if "intel" in combinado or "genuineintel" in combinado:
        return "intel"
    if "amd" in combinado or "authenticamd" in combinado or "ryzen" in combinado:
        return "amd"
    return "desconhecido"


def detectar_capacidades():
    """Inspeciona runtimes nativos, bibliotecas opcionais e dispositivos disponíveis."""
    caps = {
        "cpu_vendor": detectar_fabricante_cpu(),
        "openvino": False,
        "openvino_genai": _modulo_disponivel("openvino_genai"),
        "openvino_gpu": False,
        "nncf": _modulo_disponivel("nncf"),
        "onnxruntime": False,
        "optimum_onnx": _modulo_disponivel("optimum.onnxruntime"),
        "onnx_gpu_provider": None,
        "torch_device": "cpu",
        "bitsandbytes": _modulo_disponivel("bitsandbytes"),
    }

    if _modulo_disponivel("openvino"):
        try:
            import openvino as ov

            dispositivos = list(ov.Core().available_devices)
            caps["openvino"] = True
            caps["openvino_gpu"] = any(d == "GPU" or d.startswith("GPU.") for d in dispositivos)
        except Exception:
            caps["openvino"] = False

    if _modulo_disponivel("onnxruntime"):
        try:
            import onnxruntime as ort

            providers = set(ort.get_available_providers())
            caps["onnxruntime"] = True
            for provider in ONNX_GPU_PROVIDERS:
                if provider in providers:
                    caps["onnx_gpu_provider"] = provider
                    break
        except Exception:
            caps["onnxruntime"] = False

    try:
        if torch.cuda.is_available():
            caps["torch_device"] = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            caps["torch_device"] = "xpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            caps["torch_device"] = "mps"
    except Exception:
        caps["torch_device"] = "cpu"

    return caps


def _resolver_torch_device(dispositivo_solicitado, caps, fallbacks, contexto):
    torch_dev = caps.get("torch_device", "cpu")
    if dispositivo_solicitado == "CPU":
        return "CPU", "cpu"
    if torch_dev != "cpu":
        return "GPU", torch_dev
    if dispositivo_solicitado == "GPU":
        fallbacks.append(f"{contexto}: GPU PyTorch indisponível; usando CPU.")
    return "CPU", "cpu"


def _resolver_embeddings(solicitado, dispositivo_solicitado, caps, fallbacks):
    tem_ov = bool(caps.get("openvino"))
    tem_onnx = bool(caps.get("onnxruntime"))

    if solicitado == "auto":
        if caps.get("cpu_vendor") == "intel" and tem_ov:
            escolhido = "openvino"
        elif tem_onnx:
            escolhido = "onnx"
        else:
            escolhido = "torch"
    else:
        escolhido = solicitado

    if escolhido == "openvino":
        if tem_ov:
            if dispositivo_solicitado != "CPU" and caps.get("openvino_gpu"):
                return "openvino", "GPU", None
            if dispositivo_solicitado == "GPU":
                fallbacks.append("embeddings: GPU OpenVINO indisponível; usando CPU.")
            return "openvino", "CPU", None
        fallbacks.append("embeddings: openvino indisponível; fallback para torch.")

    if escolhido == "onnx":
        if tem_onnx:
            gpu_prov = caps.get("onnx_gpu_provider")
            if dispositivo_solicitado != "CPU" and gpu_prov:
                return "onnx", "GPU", gpu_prov
            if dispositivo_solicitado == "GPU":
                fallbacks.append("embeddings: provider GPU ONNX indisponível; usando CPUExecutionProvider.")
            return "onnx", "CPU", "CPUExecutionProvider"
        fallbacks.append("embeddings: onnxruntime indisponível; fallback para torch.")

    disp_label, _ = _resolver_torch_device(dispositivo_solicitado, caps, fallbacks, "embeddings")
    return "torch", disp_label, None


def _resolver_llm(solicitado, dispositivo_solicitado, quant_solicitada, caps, fallbacks):
    tem_ov = bool(caps.get("openvino")) and bool(caps.get("openvino_genai", caps.get("openvino")))
    tem_onnx = bool(caps.get("onnxruntime") and caps.get("optimum_onnx"))

    if solicitado == "auto":
        if caps.get("cpu_vendor") == "intel" and tem_ov:
            escolhido = "openvino"
        elif tem_onnx:
            escolhido = "onnx"
        else:
            escolhido = "torch"
    else:
        escolhido = solicitado

    if escolhido == "openvino":
        if tem_ov:
            usar_gpu = dispositivo_solicitado != "CPU" and bool(caps.get("openvino_gpu"))
            if dispositivo_solicitado == "GPU" and not usar_gpu:
                fallbacks.append("llm: GPU OpenVINO indisponível; usando CPU.")
            disp = "GPU" if usar_gpu else "CPU"
            pode_quant = bool(caps.get("openvino_genai")) or bool(caps.get("nncf"))
            if quant_solicitada in ("auto", "int8") and pode_quant:
                quant = "int8"
            elif quant_solicitada == "int4" and pode_quant:
                quant = "int4"
            elif quant_solicitada == "fp32":
                quant = "fp32"
            elif quant_solicitada == "fp16" or (quant_solicitada == "auto" and usar_gpu):
                quant = "fp16"
            else:
                quant = "fp32" if disp == "CPU" else "fp16"
                if quant_solicitada in ("int4", "int8") and not pode_quant:
                    fallbacks.append(f"llm: suporte à quantização {quant_solicitada} ausente; usando {quant}.")
            return "openvino", disp, quant, None
        fallbacks.append("llm: openvino/openvino-genai indisponível; fallback para torch.")

    if escolhido == "onnx":
        if tem_onnx:
            gpu_prov = caps.get("onnx_gpu_provider")
            usar_gpu = dispositivo_solicitado != "CPU" and bool(gpu_prov)
            if dispositivo_solicitado == "GPU" and not usar_gpu:
                fallbacks.append("llm: provider GPU ONNX indisponível; usando CPUExecutionProvider.")
            disp = "GPU" if usar_gpu else "CPU"
            prov = gpu_prov if usar_gpu else "CPUExecutionProvider"
            quant = "fp32"
            if quant_solicitada in ("fp16", "int4", "int8"):
                fallbacks.append(f"llm: precisão/quantização {quant_solicitada} direta indisponível para ONNX; usando fp32.")
            return "onnx", disp, quant, prov
        fallbacks.append("llm: onnxruntime/optimum-onnxruntime indisponível; fallback para torch.")

    disp, torch_dev = _resolver_torch_device(dispositivo_solicitado, caps, fallbacks, "llm")
    if quant_solicitada == "int4":
        if torch_dev == "cuda" and caps.get("bitsandbytes"):
            quant = "int4"
        else:
            fallbacks.append("llm: int4 em torch requer CUDA + bitsandbytes; usando fallback.")
            quant = "int8" if torch_dev == "cpu" else "fp16"
    elif quant_solicitada == "int8":
        if torch_dev == "cpu":
            quant = "int8"
        elif caps.get("bitsandbytes") and torch_dev == "cuda":
            quant = "int8"
        else:
            fallbacks.append("llm: int8 indisponível neste dispositivo; usando fp16/fp32.")
            quant = "fp16" if disp == "GPU" else "fp32"
    elif quant_solicitada == "fp16":
        if disp == "GPU":
            quant = "fp16"
        else:
            fallbacks.append("llm: fp16 em CPU torch pode ser instável; usando fp32.")
            quant = "fp32"
    elif quant_solicitada == "fp32":
        quant = "fp32"
    else:
        quant = "fp16" if disp == "GPU" else "fp32"
    return "torch", disp, quant, None


def resolver_config_execucao(env=None, capacidades=None, usar_llm=True):
    """Resolve políticas 'auto' de backend, dispositivo e quantização com fallback seguro."""
    solicitado = ler_config_env(env)
    caps = detectar_capacidades() if capacidades is None else capacidades
    fallbacks = []

    emb_backend, emb_disp, emb_prov = _resolver_embeddings(
        solicitado["emb_backend"], solicitado["dispositivo"], caps, fallbacks
    )

    if usar_llm:
        llm_backend, llm_disp, llm_quant, llm_prov = _resolver_llm(
            solicitado["llm_backend"], solicitado["dispositivo"], solicitado["llm_quantizacao"], caps, fallbacks
        )
    else:
        llm_backend, llm_disp, llm_quant, llm_prov = None, None, None, None

    for aviso in fallbacks:
        LOG.info("Portabilidade: %s", aviso)

    return {
        "emb_backend": emb_backend,
        "emb_dispositivo": emb_disp,
        "emb_provider": emb_prov,
        "llm_backend": llm_backend,
        "llm_dispositivo": llm_disp,
        "llm_quantizacao": llm_quant,
        "llm_provider": llm_prov,
        "hardware_cpu": caps.get("cpu_vendor", "desconhecido"),
        "torch_device": caps.get("torch_device", "cpu"),
        "solicitado": solicitado,
        "fallbacks": fallbacks,
    }


def normalizar_vetores(vetores):
    """Normaliza vetores em L2 e garante dtype float32 independente do backend."""
    arr = np.asarray(vetores, dtype=np.float32)
    normas = np.linalg.norm(arr, axis=-1, keepdims=True)
    normas = np.where(normas == 0, 1.0, normas).astype(np.float32)
    return (arr / normas).astype(np.float32, copy=False)


class _ModeloEmbeddingsBase:
    """Wrapper base compatível com a interface SentenceTransformer usada no RAG."""

    def __init__(self, tokenizer, input_names, dimensao, max_seq_length=128):
        self.tokenizer = tokenizer
        self._input_names = set(input_names)
        self._dimensao = int(dimensao)
        self.max_seq_length = max_seq_length

    def get_sentence_embedding_dimension(self):
        return self._dimensao

    def _executar_lote(self, entradas):
        raise NotImplementedError

    def encode(self, textos, batch_size=64, normalize_embeddings=True, show_progress_bar=False):
        unico = isinstance(textos, str)
        lista = [textos] if unico else list(textos)
        if not lista:
            return np.empty((0, self._dimensao), dtype=np.float32)
        blocos = []
        for inicio in range(0, len(lista), max(1, int(batch_size))):
            lote = lista[inicio:inicio + max(1, int(batch_size))]
            enc = self.tokenizer(
                lote,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="np",
            )
            entradas = {k: np.asarray(v) for k, v in enc.items() if k in self._input_names}
            token_emb = np.asarray(self._executar_lote(entradas), dtype=np.float32)
            mascara = enc["attention_mask"][:, :, None].astype(np.float32)
            soma = (token_emb * mascara).sum(axis=1)
            contagem = np.clip(mascara.sum(axis=1), 1e-9, None)
            blocos.append((soma / contagem).astype(np.float32))
        resultado = np.vstack(blocos).astype(np.float32, copy=False)
        if normalize_embeddings:
            resultado = normalizar_vetores(resultado)
        return resultado[0] if unico else resultado


class ModeloEmbeddingsOpenVINO(_ModeloEmbeddingsBase):
    """Executa embeddings via openvino.Core (CPU ou GPU Intel Iris Xe/Arc) sem optimum-intel."""

    def __init__(self, compiled_model, tokenizer, dimensao=None, max_seq_length=128):
        self._compiled = compiled_model
        input_names = [inp.any_name for inp in compiled_model.inputs]
        if dimensao is None:
            forma = compiled_model.output(0).partial_shape[-1]
            dimensao = int(forma.get_length()) if forma.is_static else 384
        super().__init__(tokenizer, input_names, dimensao, max_seq_length=max_seq_length)

    def _executar_lote(self, entradas):
        return self._compiled(entradas)[0]

    @classmethod
    def from_pretrained(cls, modelo_nome, revisao=None, dispositivo="CPU", offline=False,
                        carregar_fn=carregar_modelo):
        import openvino as ov
        from transformers import AutoTokenizer

        kw_hub = {"local_files_only": offline, "retornar_offline": True}
        if revisao is not None:
            kw_hub["revision"] = revisao

        tokenizer, offline = carregar_fn(
            f"o tokenizer de embeddings {modelo_nome}",
            AutoTokenizer.from_pretrained,
            modelo_nome,
            **kw_hub,
        )
        xml_path, offline = carregar_fn(
            f"os pesos OpenVINO XML de {modelo_nome}",
            hf_hub_download,
            repo_id=modelo_nome,
            filename="openvino/openvino_model.xml",
            **kw_hub,
        )
        _, offline = carregar_fn(
            f"os pesos OpenVINO BIN de {modelo_nome}",
            hf_hub_download,
            repo_id=modelo_nome,
            filename="openvino/openvino_model.bin",
            **kw_hub,
        )
        compiled = ov.Core().compile_model(str(xml_path), dispositivo)
        return cls(compiled, tokenizer), offline


class ModeloEmbeddingsONNX(_ModeloEmbeddingsBase):
    """Executa embeddings via onnxruntime.InferenceSession sem depender de optimum-onnx."""

    def __init__(self, session, tokenizer, dimensao=None, max_seq_length=128):
        self._session = session
        input_names = [inp.name for inp in session.get_inputs()]
        if dimensao is None:
            ultima = session.get_outputs()[0].shape[-1]
            dimensao = int(ultima) if isinstance(ultima, int) else 384
        super().__init__(tokenizer, input_names, dimensao, max_seq_length=max_seq_length)

    def _executar_lote(self, entradas):
        return self._session.run(None, entradas)[0]

    @classmethod
    def from_pretrained(cls, modelo_nome, revisao=None, provider="CPUExecutionProvider",
                        offline=False, carregar_fn=carregar_modelo):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        kw_hub = {"local_files_only": offline, "retornar_offline": True}
        if revisao is not None:
            kw_hub["revision"] = revisao

        tokenizer, offline = carregar_fn(
            f"o tokenizer de embeddings {modelo_nome}",
            AutoTokenizer.from_pretrained,
            modelo_nome,
            **kw_hub,
        )
        onnx_path, offline = carregar_fn(
            f"o modelo ONNX de {modelo_nome}",
            hf_hub_download,
            repo_id=modelo_nome,
            filename="onnx/model.onnx",
            **kw_hub,
        )
        providers = [provider, "CPUExecutionProvider"] if provider != "CPUExecutionProvider" else ["CPUExecutionProvider"]
        sess_kwargs = {"providers": providers}
        if provider == "DmlExecutionProvider":
            sess_options = ort.SessionOptions()
            sess_options.enable_mem_pattern = False
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_kwargs["sess_options"] = sess_options
        session = ort.InferenceSession(str(onnx_path), **sess_kwargs)
        return cls(session, tokenizer), offline


def caminho_cache_llm_openvino(llm_nome, quantizacao="int4"):
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", llm_nome).strip("_")
    return RAIZ / "busca" / "cache_indice" / f"ov_{slug}_{quantizacao}"


def _ir_valido(pasta):
    return (pasta / "openvino_model.xml").is_file() and (pasta / "openvino_model.bin").is_file()


def exportar_llm_openvino(llm_nome, quantizacao="int4", destino=None, offline=False):
    """Exporta e quantiza (1x) o LLM oficial para OpenVINO IR via `uv run` isolado."""
    destino = Path(destino) if destino is not None else caminho_cache_llm_openvino(llm_nome, quantizacao)
    if _ir_valido(destino):
        return destino
    if offline:
        raise FileNotFoundError(
            f"Modelo OpenVINO IR ausente em {destino} no modo offline."
        )
    uv_bin = shutil.which("uv")
    if not uv_bin:
        raise FileNotFoundError("Executável 'uv' não encontrado para exportar o modelo OpenVINO isoladamente.")

    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp_base = Path(tempfile.mkdtemp(prefix=f"{destino.name}.tmp.", dir=destino.parent))
    tmp = tmp_base / "ir"

    peso = quantizacao if quantizacao in ("int4", "int8", "fp16", "fp32") else "int4"
    cmd = [
        uv_bin, "run", "--python", "3.12",
        "--with", "optimum-intel[openvino]==2.2.0",
        "--with", "transformers==5.5.4",
        "--with", "huggingface-hub>=1.5,<1.22",
        "--with", "nncf>=2.19",
        "optimum-cli", "export", "openvino",
        "--model", llm_nome,
        "--weight-format", peso,
        "--task", "text-generation-with-past",
        str(tmp),
    ]
    LOG.info("Exportando e quantizando %s (%s) para OpenVINO IR em %s...", llm_nome, peso, destino)
    try:
        subprocess.run(cmd, check=True)
        if not _ir_valido(tmp):
            raise RuntimeError(f"Exportação OpenVINO não gerou openvino_model.xml/.bin em {tmp}.")
        if _ir_valido(destino):
            return destino
        try:
            tmp.rename(destino)
        except OSError:
            if _ir_valido(destino):
                return destino
            if destino.exists():
                shutil.rmtree(destino, ignore_errors=True)
                tmp.rename(destino)
            else:
                raise
    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)
    return destino


class ModeloLLMOpenVINOGenAI:
    """Wrapper para openvino_genai.LLMPipeline compatível com SistemaRAG.completar()."""

    def __init__(self, pasta_ir, dispositivo, tokenizer, pipe=None):
        self._tokenizer = tokenizer
        self.dispositivo = dispositivo
        if pipe is not None:
            self._pipe = pipe
        else:
            import openvino_genai as ov_genai

            self._pipe = ov_genai.LLMPipeline(str(pasta_ir), dispositivo)

    def eval(self):
        return self

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=200,
                 do_sample=False, repetition_penalty=1.1, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids é obrigatório para generate().")
        prompt = self._tokenizer.decode(input_ids[0], skip_special_tokens=False)
        try:
            import openvino_genai as ov_genai

            cfg = ov_genai.GenerationConfig()
        except ImportError:
            cfg = self._pipe.get_generation_config()
        cfg.max_new_tokens = int(max_new_tokens)
        cfg.do_sample = bool(do_sample)
        cfg.repetition_penalty = float(repetition_penalty)
        if hasattr(cfg, "apply_chat_template"):
            cfg.apply_chat_template = False

        resposta = self._pipe.generate(prompt, cfg)
        texto = resposta if isinstance(resposta, str) else str(resposta)
        novos = self._tokenizer.encode(texto, add_special_tokens=False)
        novos_tensor = torch.tensor([novos], dtype=input_ids.dtype)
        return torch.cat([input_ids, novos_tensor], dim=1)


def _device_torch(config_execucao, dispositivo_label):
    if dispositivo_label == "GPU":
        dev = config_execucao.get("torch_device", "cpu")
        if dev != "cpu":
            return dev
    return "cpu"


def _eh_erro_http_404(exc):
    return getattr(getattr(exc, "response", None), "status_code", None) == 404


def carregar_modelo_embeddings(modelo_nome, revisao, offline, config_execucao,
                               carregar_fn=carregar_modelo, st_cls=None):
    """Carrega embeddings via OpenVINO/ONNX nativos ou fallback silencioso para PyTorch."""
    usa_st_real = st_cls is None or getattr(st_cls, "__name__", "") == "SentenceTransformer"
    if st_cls is None:
        from sentence_transformers import SentenceTransformer as st_cls

    backend = config_execucao["emb_backend"]
    disp = config_execucao["emb_dispositivo"]
    kwargs_base = {"local_files_only": offline, "retornar_offline": True}
    if revisao is not None:
        kwargs_base["revision"] = revisao

    if backend == "openvino" and usa_st_real:
        try:
            return ModeloEmbeddingsOpenVINO.from_pretrained(
                modelo_nome, revisao=revisao, dispositivo=disp, offline=offline, carregar_fn=carregar_fn
            )
        except Exception as exc:
            if isinstance(exc, HfHubHTTPError) and not _eh_erro_http_404(exc):
                raise
            motivo = f"embeddings: falha ao carregar OpenVINO/{disp} ({exc}); fallback para torch/CPU."
            LOG.warning(motivo)
            config_execucao["fallbacks"].append(motivo)
            config_execucao["emb_backend"] = "torch"
            config_execucao["emb_dispositivo"] = "CPU"
            config_execucao["emb_provider"] = None
    elif backend == "onnx" and usa_st_real:
        try:
            prov = config_execucao.get("emb_provider") or "CPUExecutionProvider"
            return ModeloEmbeddingsONNX.from_pretrained(
                modelo_nome, revisao=revisao, provider=prov, offline=offline, carregar_fn=carregar_fn
            )
        except Exception as exc:
            if isinstance(exc, HfHubHTTPError) and not _eh_erro_http_404(exc):
                raise
            motivo = f"embeddings: falha ao carregar ONNX/{disp} ({exc}); fallback para torch/CPU."
            LOG.warning(motivo)
            config_execucao["fallbacks"].append(motivo)
            config_execucao["emb_backend"] = "torch"
            config_execucao["emb_dispositivo"] = "CPU"
            config_execucao["emb_provider"] = None
    elif backend != "torch":
        # Caminho preservado para mocks/classes de teste customizadas
        kwargs_opt = {**kwargs_base, "backend": backend}
        if backend == "onnx" and config_execucao.get("emb_provider"):
            kwargs_opt["model_kwargs"] = {"provider": config_execucao["emb_provider"]}
        try:
            return carregar_fn(
                f"o modelo de embeddings {modelo_nome} ({backend}/{disp})",
                st_cls,
                modelo_nome,
                **kwargs_opt,
            )
        except Exception as exc:
            if isinstance(exc, HfHubHTTPError) and not _eh_erro_http_404(exc):
                raise
            motivo = f"embeddings: falha ao carregar com {backend}/{disp} ({exc}); fallback para torch/CPU."
            LOG.warning(motivo)
            config_execucao["fallbacks"].append(motivo)
            config_execucao["emb_backend"] = "torch"
            config_execucao["emb_dispositivo"] = "CPU"
            config_execucao["emb_provider"] = None

    device_torch = _device_torch(config_execucao, config_execucao["emb_dispositivo"])
    return carregar_fn(
        f"o modelo de embeddings {modelo_nome}",
        st_cls,
        modelo_nome,
        device=device_torch,
        **kwargs_base,
    )


def carregar_modelo_llm(llm_nome, offline, config_execucao,
                        carregar_fn=carregar_modelo, auto_model_cls=None):
    """Carrega LLM via openvino_genai.LLMPipeline, ORT ou fallback silencioso para torch."""
    if auto_model_cls is None:
        from transformers import AutoModelForCausalLM as auto_model_cls

    backend = config_execucao["llm_backend"]
    disp = config_execucao["llm_dispositivo"]
    quant = config_execucao["llm_quantizacao"]

    if backend == "openvino":
        if _modulo_disponivel("openvino_genai"):
            try:
                from transformers import AutoTokenizer

                solicitado_quant = config_execucao.get("solicitado", {}).get("llm_quantizacao", "auto")
                if solicitado_quant == "auto" and quant == "int8":
                    cache_int8 = caminho_cache_llm_openvino(llm_nome, "int8")
                    cache_int4 = caminho_cache_llm_openvino(llm_nome, "int4")
                    int8_pronto = (cache_int8 / "openvino_model.xml").is_file()
                    int4_pronto = (cache_int4 / "openvino_model.xml").is_file()
                    pouco_espaco = shutil.disk_usage(RAIZ).free < 4 * 1024**3
                    if not int8_pronto and (int4_pronto or pouco_espaco) and int4_pronto:
                        aviso = "llm: cache int8 ausente; reutilizando cache int4 existente no modo auto."
                        LOG.info("Portabilidade: %s", aviso)
                        config_execucao.setdefault("fallbacks", []).append(aviso)
                        config_execucao["llm_quantizacao"] = "int4"
                        quant = "int4"

                pasta_ir = exportar_llm_openvino(llm_nome, quantizacao=quant, offline=offline)
                tok = AutoTokenizer.from_pretrained(str(pasta_ir), local_files_only=True)
                modelo = ModeloLLMOpenVINOGenAI(pasta_ir, disp, tok)
                return modelo.eval(), offline
            except HfHubHTTPError:
                raise
            except Exception as exc:
                motivo = f"llm: falha ao carregar OpenVINO GenAI ({exc}); fallback para torch/CPU fp32."
                LOG.warning(motivo)
                config_execucao["fallbacks"].append(motivo)
                config_execucao["llm_backend"] = "torch"
                config_execucao["llm_dispositivo"] = "CPU"
                config_execucao["llm_quantizacao"] = "fp32"
                backend, disp, quant = "torch", "CPU", "fp32"
        else:
            try:
                from optimum.intel.openvino import OVModelForCausalLM, OVWeightQuantizationConfig

                kwargs_ov = {
                    "export": True,
                    "device": disp,
                    "local_files_only": offline,
                    "retornar_offline": True,
                }
                if quant in ("int4", "int8"):
                    kwargs_ov["quantization_config"] = OVWeightQuantizationConfig(bits=4 if quant == "int4" else 8)
                lm, offline = carregar_fn(
                    f"o modelo de geração {llm_nome} (openvino/{disp}/{quant})",
                    OVModelForCausalLM.from_pretrained,
                    llm_nome,
                    **kwargs_ov,
                )
                return lm.eval(), offline
            except HfHubHTTPError:
                raise
            except Exception as exc:
                motivo = f"llm: falha ao carregar OpenVINO ({exc}); fallback para torch/CPU fp32."
                LOG.warning(motivo)
                config_execucao["fallbacks"].append(motivo)
                config_execucao["llm_backend"] = "torch"
                config_execucao["llm_dispositivo"] = "CPU"
                config_execucao["llm_quantizacao"] = "fp32"
                backend, disp, quant = "torch", "CPU", "fp32"

    if backend == "onnx":
        try:
            from optimum.onnxruntime import ORTModelForCausalLM

            kwargs_ort = {
                "export": True,
                "provider": config_execucao.get("llm_provider") or "CPUExecutionProvider",
                "local_files_only": offline,
                "retornar_offline": True,
            }
            lm, offline = carregar_fn(
                f"o modelo de geração {llm_nome} (onnx/{disp}/{quant})",
                ORTModelForCausalLM.from_pretrained,
                llm_nome,
                **kwargs_ort,
            )
            return lm.eval(), offline
        except HfHubHTTPError:
            raise
        except Exception as exc:
            motivo = f"llm: falha ao carregar ONNX ({exc}); fallback para torch/CPU fp32."
            LOG.warning(motivo)
            config_execucao["fallbacks"].append(motivo)
            config_execucao["llm_backend"] = "torch"
            config_execucao["llm_dispositivo"] = "CPU"
            config_execucao["llm_quantizacao"] = "fp32"
            backend, disp, quant = "torch", "CPU", "fp32"

    torch_dev = _device_torch(config_execucao, disp)
    if quant in ("int4", "int8") and torch_dev == "cuda":
        try:
            from transformers import BitsAndBytesConfig

            bnb = BitsAndBytesConfig(load_in_4bit=True) if quant == "int4" else BitsAndBytesConfig(load_in_8bit=True)
            lm, offline = carregar_fn(
                f"o modelo de geração {llm_nome}",
                auto_model_cls.from_pretrained,
                llm_nome,
                quantization_config=bnb,
                device_map="auto",
                local_files_only=offline,
                retornar_offline=True,
            )
            return lm.eval(), offline
        except HfHubHTTPError:
            raise
        except Exception as exc:
            motivo = f"llm: falha no {quant} torch ({exc}); fallback para fp32."
            LOG.warning(motivo)
            config_execucao["fallbacks"].append(motivo)
            config_execucao["llm_quantizacao"] = "fp32"
            quant = "fp32"

    dtype = torch.float16 if (quant == "fp16" and torch_dev != "cpu") else torch.float32
    lm, offline = carregar_fn(
        f"o modelo de geração {llm_nome}",
        auto_model_cls.from_pretrained,
        llm_nome,
        dtype=dtype,
        local_files_only=offline,
        retornar_offline=True,
    )
    if torch_dev != "cpu" and hasattr(lm, "to"):
        try:
            lm = lm.to(torch_dev)
        except Exception as exc:
            lm = lm.to("cpu", dtype=torch.float32)
            motivo = f"llm: falha ao mover modelo para {torch_dev} ({exc}); usando CPU."
            LOG.warning(motivo)
            config_execucao["fallbacks"].append(motivo)
            config_execucao["llm_dispositivo"] = "CPU"
            config_execucao["llm_quantizacao"] = "fp32"
    elif quant == "int8" and torch_dev == "cpu":
        try:
            lm = torch.ao.quantization.quantize_dynamic(lm, {torch.nn.Linear}, dtype=torch.qint8)
        except Exception as exc:
            motivo = f"llm: quantização dinâmica int8 falhou ({exc}); usando fp32."
            LOG.warning(motivo)
            config_execucao["fallbacks"].append(motivo)
            config_execucao["llm_quantizacao"] = "fp32"

    return lm.eval(), offline


def resumo_execucao(config_execucao):
    """Formata resumo conciso para logs e UI."""
    if not config_execucao:
        return ""
    partes = [f"EMB={config_execucao['emb_backend']}/{config_execucao['emb_dispositivo']}"]
    if config_execucao.get("llm_backend"):
        partes.append(
            f"LLM={config_execucao['llm_backend']}/{config_execucao['llm_dispositivo']}"
            f"({config_execucao['llm_quantizacao']})"
        )
    return " | ".join(partes)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Utilitário de aceleração e exportação OpenVINO IR.")
    ap.add_argument("--exportar-llm", required=True, help="ID do modelo HF (ex.: HuggingFaceTB/SmolLM2-1.7B-Instruct)")
    ap.add_argument("--quant", default="int4", choices=["int4", "int8", "fp16", "fp32"])
    ap.add_argument("--destino", type=Path, default=None)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    saida = exportar_llm_openvino(a.exportar_llm, quantizacao=a.quant, destino=a.destino, offline=a.offline)
    print(f"Modelo OpenVINO IR pronto em: {saida}")


if __name__ == "__main__":
    main()
