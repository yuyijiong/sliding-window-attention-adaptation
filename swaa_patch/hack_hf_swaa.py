import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from typing import Optional,Union
from functools import partial
import inspect
import os
import torch
import transformers
from transformers.models.qwen3.modeling_qwen3 import (Cache, apply_rotary_pos_emb, FlashAttentionKwargs, Unpack,
                                                      BaseModelOutputWithPast,
                                                      CausalLMOutputWithPast,Callable,ALL_ATTENTION_FUNCTIONS,eager_attention_forward)
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Attention,Qwen3DecoderLayer
from transformers.models.llama.modeling_llama import LlamaAttention,LlamaForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM,Qwen3MoeModel,Qwen3MoeAttention,MoeModelOutputWithPast,MoeCausalLMOutputWithPast,load_balancing_loss_func
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM,Qwen2Attention,Qwen2DecoderLayer
from transformers.utils import (
    logging,
)
from transformers.modeling_utils import flash_attention_forward,is_flash_attn_2_available
from transformers.modeling_flash_attention_utils import _process_flash_attention_kwargs,_hf_api_to_flash_mapping
from transformers.models.gemma2 import Gemma2Config
from transformers.models.gemma3 import Gemma3Config
from .swaa_config import SWAAConfig

logger = logging.get_logger(__name__)

def _lazy_define_process_function_swaa(flash_function):
    """
    Depending on the version and kernel some features are not supported. Due to limitations in
    `torch.compile`, we opt to statically type which (optional) kwarg parameters are supported
    within `_process_flash_attention_kwargs`.

    NOTE: While all supported kwargs are marked as `True`, everything else is marked as `False`.
          This might be confusing for kwargs that we use in any case, e.g. `is_causal`.
    """

    flash_parameters = inspect.signature(flash_function).parameters
    process_parameters = inspect.signature(_process_flash_attention_kwargs_swaa).parameters

    supports_mapping = {}
    for param in process_parameters:
        fa_param = _hf_api_to_flash_mapping.get(param, param)
        supports_mapping[fa_param] = fa_param in flash_parameters

    return partial(_process_flash_attention_kwargs_swaa, supports_mapping=supports_mapping)

def _process_flash_attention_kwargs_swaa(
    query_length: int,
    key_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    s_aux: Optional[torch.Tensor] = None,
    keep_first: Optional[int] = None,
    force_fa_decode: Optional[bool] = None,
    supports_mapping: Optional[dict[str, bool]] = None,
    **kwargs,
):
    """
    Returns a set of kwargs that are passed down to the according flash attention function based on
    requested features and whether it is supported - depends on the version and kernel implementation
    which is dynamically configured at `lazy_import_flash_attention`. The (un)supported features can be
    inspected in `supports_mapping`, see `_lazy_define_process_function` for more details.

    Args:
        query_length (`int`):
            Length of the query states
        key_length (`int`):
            Length of the key states
        is_causal (`bool`):
            Whether we perform causal (decoder) attention or full attention.
        dropout (`float`):
            Attention dropout.
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to `1 / sqrt(head_dim)`.
        sliding_window (`int`, *optional*):
            The size of the sliding window, i.e. we look at a max of `sliding_window` tokens back.
        use_top_left_mask (`bool`):
            Deprecated behavior of older versions of flash attention requiring different masking.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
        s_aux (`torch.Tensor`, *optional*):
            Attention sink auxiliary that adds a `bias` to the attention calculation via an additional head.
        keep_first (`int`, *optional*):
            Number of initial tokens to always attend to when using sliding window attention.
        force_fa_decode (`bool`, *optional*):
            Whether to automatically only allow sliding window attention to be applied to the prefilling stage.
    Return:
        flash_kwargs (`dict`):
            A dict of kwargs that are requested and supported.
    """
    flash_kwargs = {
        "causal": is_causal and not (use_top_left_mask and query_length == 1),
        "softmax_scale": softmax_scale,
    }

    if supports_mapping["dropout_p"]:
        flash_kwargs["dropout_p"] = dropout

    if supports_mapping["window_size"] and sliding_window is not None and key_length > sliding_window:
        # The flash attention API sets inclusive boundaries, i.e. (4, 0) would take 4 tokens to the left
        # and the current token for a total size of 5. However, we usually define our window sizes by
        # their total window size (when causal). Encoder models as of now seldom use SWA and when they
        # do, they have a custom workaround (e.g. ModernBERT) which would align with this symmetric logic, i.e.
        # for a total of `2*sliding_window + 1`.
        flash_kwargs["window_size"] = (sliding_window - 1, sliding_window - 1)

    if supports_mapping["deterministic"]:
        flash_kwargs["deterministic"] = (
            deterministic if deterministic is not None else os.getenv("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
        )

    if supports_mapping["softcap"] and softcap is not None:
        flash_kwargs["softcap"] = softcap

    # Only within kernel implementation atm
    if supports_mapping["s_aux"] and s_aux is not None:
        flash_kwargs["s_aux"] = s_aux

    # Get keep_first from kwargs and pass to flash_kwargs
    if supports_mapping["keep_first"] and keep_first is not None:
        flash_kwargs["keep_first"] = keep_first

    # Get force_fa_decode from kwargs and pass to flash_kwargs
    if supports_mapping["force_fa_decode"] and force_fa_decode is not None:
        flash_kwargs["force_fa_decode"] = force_fa_decode

    return flash_kwargs

def attention_forward_swaa(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache],
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:

    q_len = hidden_states.shape[1]
    batch_size = hidden_states.shape[0]
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Extract sliding_window_size, keep_first, force_fa_decode, non_sliding_layers from self.config.swaa_config
    swaa_config:SWAAConfig = self.config.swaa_config if hasattr(self.config, "swaa_config") else SWAAConfig()

    sliding_window_size=swaa_config.sliding_window_size
    non_sliding_layers=swaa_config.non_sliding_layers
    force_fa_decode=swaa_config.force_fa_decode
    keep_first=swaa_config.keep_first

    # Disable sliding window if the current layer is in non_sliding_layers
    if int(self.layer_idx) in non_sliding_layers:
        sliding_window_size=None

    if isinstance(swaa_config.force_fa_decode,list):
        force_fa_decode= int(self.layer_idx) in swaa_config.force_fa_decode

    # Print SWAA configuration if environment variable SWAA_DEBUG is '1'
    if os.environ.get("SWAA_DEBUG", "0") == "1":
        print(
            "Attention initialized with sliding_window_size={}, keep_first={}, prefill_slide={}, non_sliding_layers={}".format(
                sliding_window_size, keep_first, force_fa_decode, non_sliding_layers
            ))


    # Get prompt_length from kwargs
    prompt_length = kwargs.get("prompt_length", None)
    if force_fa_decode and prompt_length is None and sliding_window_size is not None and self.training:
        raise ValueError("prompt_length must be provided in training when force_fa_decode=True and sliding_window_size is not None.")

    # shape (batch_size, num_attention_heads, seq_length, head_dim)
    if self.config.model_type in ["qwen3","qwen3_moe"]:
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    elif self.config.model_type in ["llama","qwen2"]:
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    else:
        raise ValueError("Unsupported model type: {}".format(self.config.model_type))

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx,
                                                          cache_kwargs)

    if prompt_length is not None and force_fa_decode and sliding_window_size is not None:
        if batch_size > 1:
            raise NotImplementedError("batch size > 1 is not supported when force_fa_decode=True in training.")
        # Training phase with provided prompt_length and force_fa_decode=True:
        # Attention is calculated in two parts.
        # Part 1 (prefill) requires SWA; Part 2 (decoding) does not require SWA.
        # Split query_states
        query_states_prompt = query_states[:, :, :prompt_length, :]
        query_states_answer = query_states[:, :, prompt_length:, :]
        key_states_prompt = key_states[:, :, :prompt_length, :]
        value_states_prompt = value_states[:, :, :prompt_length, :]

        # Calculate attention for the prompt part using sliding window
        attn_output_prefill,_ = flash_attention_forward(
            self,
            query_states_prompt,
            key_states_prompt,
            value_states_prompt,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window_size,
            keep_first=keep_first,
            force_fa_decode=False,
            **kwargs,)

        # Calculate attention for the answer part using full attention
        attn_output_decode,_ = flash_attention_forward(
            self,
            query_states_answer,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=None,
            **kwargs,
        )

        # Concatenate outputs
        attn_output = torch.cat([attn_output_prefill, attn_output_decode], dim=1)

    else:
        # inference, or training without force_fa_decode
        attn_output, _ = flash_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window_size,
            keep_first=keep_first,
            force_fa_decode=force_fa_decode,
            **kwargs,
        )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None

def hybrid_causallm_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
) -> CausalLMOutputWithPast:

    # Get prompt_length
    prompt_length = kwargs.pop("prompt_length", None)
    # If labels are provided and prompt_length is None, calculate prompt_length:
    # prompt_length is the count of -100 in labels[0]
    if labels is not None and prompt_length is None:
        prompt_length = (labels[0] == -100).nonzero(as_tuple=False).shape[0] if labels[0].numel() > 0 else None

    if use_cache and past_key_values is None:
        print("Warning: use_cache is True but past_key_values is None.")

    outputs: BaseModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        prompt_length=prompt_length,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def hybrid_causallm_forward_moe(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
) -> MoeCausalLMOutputWithPast:

    # Get prompt_length
    prompt_length = kwargs.pop("prompt_length", None)
    # If labels are provided and prompt_length is None, calculate prompt_length:
    # prompt_length is the count of -100 in labels[0]
    if labels is not None and prompt_length is None:
        prompt_length = (labels[0] == -100).nonzero(as_tuple=False).shape[0] if labels[0].numel() > 0 else None

    if use_cache and past_key_values is None:
        print("Warning: use_cache is True but past_key_values is None.")

    output_router_logits = (
        output_router_logits if output_router_logits is not None else self.config.output_router_logits
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs: MoeModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_router_logits=output_router_logits,
        cache_position=cache_position,
        prompt_length=prompt_length,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

    aux_loss = None
    if output_router_logits:
        aux_loss = load_balancing_loss_func(
            outputs.router_logits,
            self.num_experts,
            self.num_experts_per_tok,
            attention_mask,
        )
        if labels is not None:
            loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

    return MoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
    )


def hack_hf_swaa(training=False):
    if not is_flash_attn_2_available():
        raise ImportError("Flash Attention 2 is not available. "
                          "Please install customized flash-attn to use Sliding Window Attention Adaptation.")

    transformers.modeling_flash_attention_utils._process_flash_attention_kwargs=_process_flash_attention_kwargs_swaa
    #transformers.modeling_flash_attention_utils._lazy_define_process_function=_lazy_define_process_function_swaa


    Qwen3Attention.forward = attention_forward_swaa
    Qwen3MoeAttention.forward = attention_forward_swaa
    LlamaAttention.forward = attention_forward_swaa
    Qwen2Attention.forward= attention_forward_swaa

    if training:
        Qwen3ForCausalLM.forward = hybrid_causallm_forward
        Qwen3MoeForCausalLM.forward = hybrid_causallm_forward_moe
        LlamaForCausalLM.forward = hybrid_causallm_forward
        Qwen2ForCausalLM.forward = hybrid_causallm_forward

    print("Hacked Qwen3, Qwen3Moe, Qwen2, and Llama models to use customized attention for SWAA.")