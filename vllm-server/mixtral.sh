CUDA_VISIBLE_DEVICES=0,1 python vllm_api.py \
 --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
 --host 0.0.0.0 \
 --served-model-name mixtral \
 --dtype float16 \
 --port 23100 \
 --disable-log-requests \
 --trust-remote-code \
 --tp 2