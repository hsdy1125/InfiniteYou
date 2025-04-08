#!/bin/bash
#$ -N Destylization_InfiniteYou
#$ -o logs/Destylization_InfDe.out
#$ -e logs/Destylization_InfDe.err
#$ -l tmem=80G          # 请求 GPU 卡附带 80G 显存
#$ -l gpu=true          # 声明需要使用 GPU
#$ -l h_rt=20:00:00     # 最长运行时间
#$ -q gpu.q
#$ -cwd

# === 环境激活 ===
export PATH=/share/apps/anaconda3-2022-5/bin:$PATH
source /share/apps/anaconda3-2022-5/bin/activate 
export PATH=/share/apps/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/cuda-11.8/lib64:$LD_LIBRARY_PATH 
conda activate infiniteYou

# === 路径参数 ===
JSON_FILE="/SAN/intelsys/IdPreservProject/epic/dataset/sub_8emotion_ipDe.json"
DATASET_ROOT="/SAN/intelsys/IdPreservProject/epic"
SAVE_ROOT="${DATASET_ROOT}/dataset/ipPairdImg"
PROMPT="photorealistic, real person, neutral expression, neutral emotion"

# === 进入工作目录 ===
cd /SAN/intelsys/IdPreservProject

# === 获取 JSON 长度 ===
NUM_SAMPLES=$(jq length "$JSON_FILE")

echo "📄 开始处理 $NUM_SAMPLES 条样本..."

# === 遍历样本 ===
for ((i=0; i<$NUM_SAMPLES; i++)); do
  IMG_PATH=$(jq -r ".[$i].img_path" "$JSON_FILE")
  ID_NAME=$(basename "$IMG_PATH" .png)
  ID_IMAGE="${DATASET_ROOT}/${IMG_PATH}"
  OUT_DIR="${SAVE_ROOT}"
  EMOTION="${ID_NAME}_InfDe"

  echo "➡️ [$i/$NUM_SAMPLES] Processing $ID_NAME"

  python /SAN/intelsys/IdPreservProject/InfiniteYou/test.py \
    --emotion "$EMOTION" \
    --id_image "$ID_IMAGE" \
    --out_results_dir "$OUT_DIR" \
    --prompt "$PROMPT"
done

echo "✅ 所有处理完成"

#qsub /SAN/intelsys/IdPreservProject/InfiniteYou/run_infiniteyou_all.sh
#Your job 5427712 ("Destylization_InfiniteYou") has been submitted