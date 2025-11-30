import os

os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"

from typing import Optional, Tuple, List, Union
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import torch
import vllm
# import flash_attn_2_cuda_vllm
# from flash_attn.flash_attn_interface import flash_attn_gpu
# from vllm import _custom_ops as ops
from vllm.model_executor.models.qwen3 import (AttentionType, extract_layer_index)

from vllm.v1.attention.backends.flash_attn import (FlashAttentionImpl,
                                                   FlashAttentionMetadata,
                                                   FlashAttentionBackend,
                                                   cascade_attention, is_quantized_kv_cache, flash_attn_supports_fp8)
from vllm.vllm_flash_attn.flash_attn_interface import DEFAULT_FA_VERSION, maybe_contiguous
from vllm.attention.utils.fa_utils import (is_flash_attn_varlen_func_available)
from vllm.entrypoints.llm import *
from vllm.v1.engine.llm_engine import *

if is_flash_attn_varlen_func_available():
    from vllm.attention.utils.fa_utils import (reshape_and_cache_flash)
import flash_attn_2_cuda_vllm
from swaa_config import SWAAConfig


def flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q,
        cu_seqlens_q,
        max_seqlen_k,
        cu_seqlens_k=None,  # only used for non-paged prefill
        seqused_k=None,
        q_v=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size: Optional[List[int]] = None,
        keep_first=0,
        force_fa_decode=False,
        softcap=0.0,  # 0.0 means deactivated
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        block_table=None,
        return_softmax_lse=False,
        out=None,
        # FA3 Only
        scheduler_metadata=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        num_splits: int = 0,
        # Version selector
        fa_version: int = DEFAULT_FA_VERSION,
        s_aux=None,
):
    """
    Flash Attention forward function supporting variable sequence lengths.
    Includes custom parameters for Sliding Window Attention Adaptation (SWAA).

    Arguments:
        ... (standard FA arguments) ...
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        keep_first: Number of initial tokens to always attend to (sink tokens).
        force_fa_decode: Whether to disable sliding window in decoding phase.
        ...
    """
    assert cu_seqlens_k is not None or seqused_k is not None, \
        "cu_seqlens_k or seqused_k must be provided"
    assert cu_seqlens_k is None or seqused_k is None, \
        "cu_seqlens_k and seqused_k cannot be provided at the same time"
    assert block_table is None or seqused_k is not None, \
        "seqused_k must be provided if block_table is provided"

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    # custom op does not support non-tuple input
    real_window_size: Tuple[int, int]
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)

    if fa_version == 2:
        if scheduler_metadata is not None and q_descale is not None \
                and k_descale is not None and v_descale is not None:
            raise NotImplementedError(
                "FA2 does not support scheduler_metadata, q_descale, "
                "k_descale, v_descale"
            )
        if s_aux is not None:
            raise NotImplementedError("FA2 does not support s_aux")
        if num_splits > 1:
            raise NotImplementedError("FA2 does not support num_splits > 1")
        out_new = flash_attn_2_cuda_vllm.varlen_fwd(
            # torch.ops.flash_attn.varlen_fwd(
            q, k, v,
            out,
            cu_seqlens_q,
            # cu_seqlens_k not used since we use seqused_k, but flash_api.cpp
            # still wants it so we pass all zeros
            dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
            seqused_k,
            None,  # leftpad_k
            block_table,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            real_window_size[0],
            real_window_size[1],
            softcap,
            keep_first,
            force_fa_decode,
            return_softmax_lse and dropout_p > 0,
            None,
        )
    elif fa_version == 3:
        raise ValueError(f"Unsupported FA version: {fa_version}")
    else:
        raise ValueError(f"Unsupported FA version: {fa_version}")
    return out_new


class FlashAttentionImplSWAA(FlashAttentionImpl):
    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            logits_soft_cap: Optional[float] = None,
            attn_type: AttentionType = AttentionType.DECODER,
            kv_sharing_target_layer_name: Optional[str] = None,
            sinks: Optional[torch.Tensor] = None,
            swaa_config: SWAAConfig = None
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # FlashAttentionBackend.validate_head_size(head_size)

        self.attn_type = attn_type
        self.vllm_flash_attn_version = 2
        if is_quantized_kv_cache(self.kv_cache_dtype) \
                and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device.")

        self.sinks = sinks
        if self.sinks is not None:
            assert self.vllm_flash_attn_version == 3, (
                "Sinks are only supported in FlashAttention 3")
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer")

        # Sliding Window Attention Adaptation config
        self.sliding_window_size = swaa_config.sliding_window_size
        self.keep_first = swaa_config.keep_first
        self.force_fa_decode = swaa_config.force_fa_decode
        self.non_sliding_layers = swaa_config.non_sliding_layers

        # Print SWAA config if debug environment variable is set
        if os.environ.get("SWAA_DEBUG", "0") == "1":
            print(
                "FlashAttentionImplSWAA initialized with sliding_window_size={}, keep_first={}, force_fa_decode={}, non_sliding_layers={}".format(
                    self.sliding_window_size, self.keep_first, self.force_fa_decode, self.non_sliding_layers))

    def forward(
            self,
            layer: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: FlashAttentionMetadata,
            output: Optional[torch.Tensor] = None,
            output_scale: Optional[torch.Tensor] = None,
            output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention."""
        assert output is not None, "Output tensor must be provided."

        start_time_attention = time.perf_counter()
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for FlashAttentionImpl")

        if attn_metadata is None:
            # Profiling run.
            return output

        attn_type = self.attn_type

        # Get layer index
        layer_index = extract_layer_index(layer.layer_name)

        if layer_index == 0:
            # Debugging code removed for cleaner output.
            pass

        # IMPORTANT! (vLLM warning)
        # Minimize the PyTorch ops in this method as much as possible.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(query[:num_actual_tokens],
                                                   key[:num_actual_tokens],
                                                   value[:num_actual_tokens],
                                                   output[:num_actual_tokens],
                                                   attn_metadata, layer)

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(0)

        # key and value may be None in the case of cross attention. They are
        # calculated once based on the output from the encoder and then cached
        # in KV cache.
        if (self.kv_sharing_target_layer_name is None and key is not None
                and value is not None):
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            # queries are quantized in the attention layer
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype)
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            # Determine sliding_window based on non_sliding_layers and config
            if layer_index in self.non_sliding_layers or self.sliding_window_size is None:
                sliding_window = [-1, 0]  # Disable sliding window
            else:
                sliding_window = [self.sliding_window_size, 0]  # Enable sliding window

            # Determine force_fa_decode
            if isinstance(self.force_fa_decode, list):
                force_fa_decode = True if layer_index in self.force_fa_decode else False
            else:
                force_fa_decode = self.force_fa_decode

            start_time_flash_varlen = time.perf_counter()
            flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=sliding_window,
                keep_first=self.keep_first,
                force_fa_decode=force_fa_decode,
                block_table=block_table,
                softcap=self.logits_soft_cap,
                scheduler_metadata=None,  # scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                q_descale=None,  # layer._q_scale.expand(descale_shape),
                k_descale=None,  # layer._k_scale.expand(descale_shape),
                v_descale=None,  # layer._v_scale.expand(descale_shape),
                num_splits=attn_metadata.max_num_splits,
                s_aux=self.sinks,
            )
            end_time = time.perf_counter()

            # Log performance details if debug is enabled
            if layer_index == 10 and max_seqlen_k % 100 == 0 and os.environ.get("SWAA_DEBUG", "0") == "1":
                print("flash_attn_varlen_func time:", round(end_time - start_time_flash_varlen, 6),
                      " total attention time:", round(end_time - start_time_attention, 6),
                      " max seq len q:", max_seqlen_q, "max seq len k:", max_seqlen_k)

            return output

        # Cascade attention (rare case).
        print("Using cascade attention")
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
        )
        return output


class LLMSWAA(LLM):
    def __init__(
            self,
            model: str,
            *,
            runner: RunnerOption = "auto",
            convert: ConvertOption = "auto",
            tokenizer: Optional[str] = None,
            tokenizer_mode: TokenizerMode = "auto",
            skip_tokenizer_init: bool = False,
            trust_remote_code: bool = False,
            allowed_local_media_path: str = "",
            allowed_media_domains: Optional[list[str]] = None,
            tensor_parallel_size: int = 1,
            dtype: ModelDType = "auto",
            quantization: Optional[QuantizationMethods] = None,
            revision: Optional[str] = None,
            tokenizer_revision: Optional[str] = None,
            seed: Optional[int] = None,
            gpu_memory_utilization: float = 0.9,
            swap_space: float = 4,
            cpu_offload_gb: float = 0,
            enforce_eager: bool = False,
            disable_custom_all_reduce: bool = False,
            hf_token: Optional[Union[bool, str]] = None,
            hf_overrides: Optional[HfOverrides] = None,
            mm_processor_kwargs: Optional[dict[str, Any]] = None,
            pooler_config: Optional[PoolerConfig] = None,
            override_pooler_config: Optional[PoolerConfig] = None,
            structured_outputs_config: Optional[Union[dict[
                str, Any], StructuredOutputsConfig]] = None,
            kv_cache_memory_bytes: Optional[int] = None,
            compilation_config: Optional[Union[int, dict[str, Any],
            CompilationConfig]] = None,
            logits_processors: Optional[list[Union[str,
            type[LogitsProcessor]]]] = None,
            swaa_config: SWAAConfig = None,
            **kwargs: Any,
    ) -> None:
        """LLM constructor."""

        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            # if the worker_cls is not qualified string name,
            # we serialize it using cloudpickle to avoid pickling issues
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        if "kv_transfer_config" in kwargs and isinstance(
                kwargs["kv_transfer_config"], dict):
            from vllm.config.kv_transfer import KVTransferConfig
            raw_config_dict = kwargs["kv_transfer_config"]
            try:
                kwargs["kv_transfer_config"] = KVTransferConfig(
                    **raw_config_dict)
            except ValidationError as e:
                logger.error(
                    "Failed to convert 'kv_transfer_config' dict to "
                    "KVTransferConfig object. Dict: %s. Error: %s",
                    raw_config_dict, e)
                # Consider re-raising a more specific vLLM error or ValueError
                # to provide better context to the user.
                raise ValueError(
                    f"Invalid 'kv_transfer_config' provided: {e}") from e

        if hf_overrides is None:
            hf_overrides = {}

        if compilation_config is not None:
            if isinstance(compilation_config, int):
                compilation_config_instance = CompilationConfig(
                    level=compilation_config)
            elif isinstance(compilation_config, dict):
                compilation_config_instance = CompilationConfig(
                    **{
                        k: v
                        for k, v in compilation_config.items()
                        if is_init_field(CompilationConfig, k)
                    })
            else:
                compilation_config_instance = compilation_config
        else:
            compilation_config_instance = CompilationConfig()

        if structured_outputs_config is not None:
            if isinstance(structured_outputs_config, dict):
                structured_outputs_instance = StructuredOutputsConfig(
                    **{
                        k: v
                        for k, v in structured_outputs_config.items()
                        if is_init_field(StructuredOutputsConfig, k)
                    })
            else:
                structured_outputs_instance = structured_outputs_config
        else:
            structured_outputs_instance = StructuredOutputsConfig()

        engine_args = EngineArgs(
            model=model,
            runner=runner,
            convert=convert,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            allowed_local_media_path=allowed_local_media_path,
            allowed_media_domains=allowed_media_domains,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            disable_custom_all_reduce=disable_custom_all_reduce,
            hf_token=hf_token,
            hf_overrides=hf_overrides,
            mm_processor_kwargs=mm_processor_kwargs,
            pooler_config=pooler_config,
            override_pooler_config=override_pooler_config,
            structured_outputs_config=structured_outputs_instance,
            compilation_config=compilation_config_instance,
            logits_processors=logits_processors,
            **kwargs,
        )

        log_non_default_args(engine_args)

        engine_args.swaa_config = swaa_config
        print("You are using LLMSWAA with swaa_config:", swaa_config)

        # Create the Engine (autoselects V0 vs V1)
        self.llm_engine = LLMEngineSWAA.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS)
        self.engine_class = type(self.llm_engine)

        self.request_counter = Counter()
        self.default_sampling_params: dict[str, Any] | None = None

        supported_tasks = self.llm_engine.get_supported_tasks()
        logger.info("Supported tasks: %s", supported_tasks)
        self.supported_tasks = supported_tasks

        self.model_config = self.llm_engine.model_config
        self.processor = self.llm_engine.processor
        self.io_processor = self.llm_engine.io_processor


class LLMEngineSWAA(LLMEngine):
    @classmethod
    def from_engine_args(
            cls,
            engine_args: EngineArgs,
            usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
            stat_loggers: Optional[list[StatLoggerFactory]] = None,
            enable_multiprocessing: bool = False,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        vllm_config.swaa_config = engine_args.swaa_config

        if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.debug("Enabling multiprocessing for LLMEngine.")
            enable_multiprocessing = True

        # Create the LLMEngine.
        return cls(vllm_config=vllm_config,
                   executor_class=executor_class,
                   log_stats=not engine_args.disable_log_stats,
                   usage_context=usage_context,
                   stat_loggers=stat_loggers,
                   multiprocess_mode=enable_multiprocessing)


def from_engine_args_swaa(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
        enable_multiprocessing: bool = False,
) -> "LLMEngine":
    """Creates an LLM engine from the engine arguments."""

    # Create the engine configs.
    vllm_config = engine_args.create_engine_config(usage_context)
    executor_class = Executor.get_class(vllm_config)

    # Inject SWAA config into vllm_config so it can be accessed by Models
    vllm_config.swaa_config = engine_args.swaa_config

    if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
        logger.debug("Enabling multiprocessing for LLMEngine.")
        enable_multiprocessing = True

    # Create the LLMEngine.
    return cls(vllm_config=vllm_config,
               executor_class=executor_class,
               log_stats=not engine_args.disable_log_stats,
               usage_context=usage_context,
               stat_loggers=stat_loggers,
               multiprocess_mode=enable_multiprocessing)


@staticmethod
def get_impl_cls_swaa() -> type["FlashAttentionImpl"]:
    return FlashAttentionImplSWAA


def hack_vllm_swaa():
    from vllm_0110_swaa_models import Qwen3ModelSWAA, Qwen2ModelSWAA, LlamaModelSWAA, Qwen3MoeModelSWAA

    # replace get_impl_cls with get_impl_cls_swaa to use custom implementation
    FlashAttentionBackend.get_impl_cls = get_impl_cls_swaa

    # Replace model classes with SWAA-enabled versions
    # vllm.model_executor.models.transformers.TransformersBase = TransformersBaseSWAA (Removed/Commented out)
    # vllm.model_executor.models.transformers.TransformersForCausalLM = TransformersForCausalLMSWAA (Removed/Commented out)

    vllm.model_executor.models.qwen3.Qwen3Model = Qwen3ModelSWAA
    vllm.model_executor.models.qwen2.Qwen2Model = Qwen2ModelSWAA
    vllm.model_executor.models.llama.LlamaModel = LlamaModelSWAA
    vllm.model_executor.models.qwen3_moe.Qwen3MoeModel = Qwen3MoeModelSWAA

    # replace from_engine_args with from_engine_args_swaa
    # EngineArgs.from_engine_args = classmethod(from_engine_args_swaa) (Replaced with LLMEngineSWAA.from_engine_args patch)

    # replace LLM with LLMSWAA
    vllm.LLM = LLMSWAA