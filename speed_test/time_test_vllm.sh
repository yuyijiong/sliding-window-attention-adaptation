export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/share/models/Qwen3-4B-Thinking-2507"
# export HF_ENDPOINT="https://hf-mirror.com"
# nohup bash time_test_vllm.sh > time_test_vllm.log 2>&1 &

vllm bench sweep serve \
    --serve-cmd "python ../Patch/serve_swaa.py --enforce-eager --max-model-len 138000 --max-num-batched-tokens 16384 --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --host 0.0.0.0 --port 5001 --served-model-name qwen3-4b --model $MODEL_PATH" \
    --bench-cmd "vllm bench serve --ready-check-timeout-sec 60 --port 5001 --host 0.0.0.0 --num-prompts 100 --model $MODEL_PATH --served-model-name qwen3-4b --backend vllm --endpoint /v1/completions --dataset-name random" \
    --serve-params ./serve_hparams2.json \
    --bench-params ./bench_hparams.json \
    -o vllm_bench_results/ \
    --num-runs 1 \
    --after-bench-cmd "sleep 1"