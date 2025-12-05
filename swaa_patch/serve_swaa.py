import sys
# Append parent directory to sys.path
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
import uvloop
import argparse
from typing import Optional
import json
# Import vLLM modules
import vllm.envs as envs
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.config import VllmConfig

# Import custom Hack code
from hack_vllm_0110_swaa import hack_vllm_swaa, SWAAConfig

# ==========================================
# 1. Apply Core Hacks (Replace Model/Attention classes)
# ==========================================
print("[SWAA] Applying Model and Attention hacks...")
hack_vllm_swaa()

# ==========================================
# 2. Patch AsyncEngineArgs
# ==========================================
# Patch AsyncEngineArgs to pass SWAA CLI arguments to VllmConfig.

# Save original methods
_original_from_cli_args = AsyncEngineArgs.from_cli_args
_original_create_engine_config = AsyncEngineArgs.create_engine_config


def patched_from_cli_args(cls, args: argparse.Namespace):
    # 1. Call original method to create engine_args object
    engine_args = _original_from_cli_args(args)

    # 2. Manually bind SWAA parameters from CLI args to the engine_args instance
    if hasattr(args, "sliding_window_size"):
        # Check if sliding_window_size is defined before creating SWAAConfig
        if args.sliding_window_size is not None or hasattr(args, 'non_sliding_layers'):
            force_fa_decode_bool = True if isinstance(args.force_fa_decode,
                                                      str) and 'true' in args.force_fa_decode.lower() else False

            # If it's a list string from CLI, it's parsed as a list in __main__
            non_sliding_layers_list = args.non_sliding_layers if isinstance(args.non_sliding_layers, list) else []

            swaa_config = SWAAConfig(
                sliding_window_size=args.sliding_window_size,
                keep_first=args.keep_first,
                force_fa_decode=force_fa_decode_bool,
                non_sliding_layers=non_sliding_layers_list
            )
            # Dynamically bind to the instance
            engine_args.swaa_config = swaa_config
            print(f"[SWAA] SWAA Config loaded into EngineArgs: {swaa_config}")
        else:
            print("[SWAA] SWAA options present but sliding_window_size is None. SWAA is likely inactive.")

    return engine_args


def patched_create_engine_config(self, *args, **kwargs):
    # 1. Call original method to create vllm_config
    vllm_config = _original_create_engine_config(self, *args, **kwargs)

    # 2. Check if self (engine_args) has swaa_config and inject it into vllm_config
    if hasattr(self, "swaa_config"):
        vllm_config.swaa_config = self.swaa_config
        print("[SWAA] SWAA Config injected into VllmConfig")
    else:
        # This warning is less critical if SWAA is intended to be off.
        pass

    return vllm_config


# Apply Patch
AsyncEngineArgs.from_cli_args = classmethod(patched_from_cli_args)
AsyncEngineArgs.create_engine_config = patched_create_engine_config


# ==========================================
# 3. Custom Argument Parser
# ==========================================
def add_swaa_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("SWAA Options")
    group.add_argument("--sliding-window-size", type=int, default=None,
                       help="Override model's sliding window size.")
    group.add_argument("--keep-first", type=int, default=0,
                       help="Number of initial tokens to keep (sink tokens).")
    # Type is str to handle CLI input flexibly
    group.add_argument("--force-fa-decode", type=str, default="False",
                       help="Set to 'True' to force full attention during decoding (default: False).")
    parser.add_argument('--non-sliding-layers', type=str, default='',
                        help="List of layer indices (as a string, e.g., '[1, 3, 5]') that should use full attention.")
    return parser


# ==========================================
# 4. Main Entry Point (Mimics vllm.entrypoints.openai.api_server)
# ==========================================
if __name__ == "__main__":
    # Initialize parser
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server with SWAA support."
    )
    # Add native vLLM arguments
    parser = make_arg_parser(parser)
    # Add SWAA arguments
    parser = add_swaa_args(parser)

    args = parser.parse_args()

    # Process non-sliding-layers argument (converting from string to list of ints)
    non_sliding_layers_list = []

    if args.non_sliding_layers:
        # Case A: Input is empty string ""
        if args.non_sliding_layers.strip() == "":
            non_sliding_layers_list = []

        # Case B: Input is a list string "[1, 3, 5]" (common for JSON/CLI)
        elif args.non_sliding_layers.strip().startswith("["):
            try:
                non_sliding_layers_list = json.loads(args.non_sliding_layers)
                if not isinstance(non_sliding_layers_list, list) or not all(
                        isinstance(x, int) for x in non_sliding_layers_list):
                    raise TypeError("Parsed list elements are not all integers.")
            except (json.JSONDecodeError, TypeError):
                print(f"Error parsing non-sliding-layers list: {args.non_sliding_layers}. Falling back to empty list.")
                non_sliding_layers_list = []

        # Case C: Input is space or comma-separated string "1 3 5" (if manually run)
        else:
            try:
                cleaned_str = args.non_sliding_layers.replace(",", " ")
                non_sliding_layers_list = [int(x) for x in cleaned_str.split()]
            except ValueError:
                print(
                    f"Error parsing non-sliding-layers string: {args.non_sliding_layers}. Falling back to empty list.")
                non_sliding_layers_list = []

    # Assign the processed list back to args
    args.non_sliding_layers = non_sliding_layers_list

    validate_parsed_serve_args(args)

    # Start Server
    # run_server calls AsyncEngineArgs.from_cli_args(args), triggering our patch
    uvloop.run(run_server(args))