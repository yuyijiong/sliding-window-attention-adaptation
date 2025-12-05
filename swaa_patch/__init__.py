# 1. Import Configuration

from .swaa_config import SWAAConfig


# 2. Import Hugging Face Hack
# This module depends on transformers/torch.
from .hack_hf_swaa import hack_hf_swaa


# 3. Import vLLM Hack (Optional Dependency)
# This module strictly depends on vllm. We wrap it in a try-except block
# to prevent the package from crashing if the user does not have vllm installed.
try:
    from .hack_vllm_0110_swaa import hack_vllm_swaa, LLMSWAA
except ImportError:
    # If vllm is not installed (or compatible), define dummy objects.
    # These will only raise errors if the user explicitly tries to use them.

    def hack_vllm_swaa(*args, **kwargs):
        raise ImportError(
            "Cannot call 'hack_vllm_swaa'. 'vllm' does not appear to be installed "
            "in your environment, or the version is incompatible.\n"
            "Please ensure vllm is installed to use this feature."
        )


__all__ = [
    "hack_hf_swaa",
    "hack_vllm_swaa",
    "SWAAConfig"
]