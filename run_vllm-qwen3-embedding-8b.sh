#!/bin/bash

PORT=1827   # sudo lsof -i:1825
export CUDA_VISIBLE_DEVICES=2
model_path="/nfs2/zdy_download/Qwen/Qwen3-Embedding-8B"
model_name="Qwen3-Embedding-8B"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="log/vllm"
LOG_FILE="${LOG_DIR}/${model_name}_${TIMESTAMP}.log"

RED='\033[0;31m'
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
NC='\033[0m' # No Color

check_port() {
    local port=$1
    if sudo ss -tlnp | grep -q ":${port} " || \
       sudo ss -tlnp | grep -q ":${port}\$"; then
        return 1
    fi
    return 0
}

echo -e "\n${GREEN}检测端口: $PORT | 当前时间: $(date +"%Y-%m-%d %H:%M:%S") ${NC}"

    # --enable-auto-tool-choice \
    # --tool-call-parser qwen3_xml \

    #     nohup python -m vllm.entrypoints.openai.api_server \
    # --model $model_path --port $PORT \
    # --dtype bfloat16 \
    # --gpu-memory-utilization 0.55 \
    # --trust-remote-code \
    # --served-model-name $model_name \
    # --max-model-len 32768  >> ${LOG_FILE} 2>&1 &

if check_port $PORT; then
    echo -e "${GREEN}✓ 端口 $PORT 可用，准备启动vLLM服务...${NC}"
    echo -e "${ORANGE} Use CUDA device ${CUDA_VISIBLE_DEVICES}. ${NC}\n"
    echo -e "${ORANGE} Log file: ${LOG_FILE} ${NC}\n"

    # 清理旧日志（保留最近7天）     
    find ${LOG_DIR} -name "vllm_${model_name}_*.log" -mtime +7 -delete

    #     --tensor-parallel-size 2 \
    # --gpu-memory-utilization 0.35 \
    nohup python -m vllm.entrypoints.openai.api_server \
    --model $model_path --port $PORT \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --served-model-name $model_name \
    --seed 42 \
    --max-model-len 8192 >> ${LOG_FILE} 2>&1 &

else
    echo -e "${RED}✗ 端口 $PORT 已被占用！${NC}"
    exit 1
fi
