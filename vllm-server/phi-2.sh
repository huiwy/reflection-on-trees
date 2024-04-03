CUDA_VISIBLE_DEVICES=2 python vllm_api.py \
 --model microsoft/phi-2 \
 --host 0.0.0.0 \
 --served-model-name phi-2 \
 --dtype float16 \
 --port 23100 \
 --disable-log-requests \
 --trust-remote-code