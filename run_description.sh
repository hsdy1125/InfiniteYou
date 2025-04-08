#!/bin/bash

# === 参数设定 ===
DESC_FILE="/SAN/intelsys/IdPreservProject/InfiniteYou/description/API_4o_Prompt7_P17_D1.txt"
ID_IMAGE="epic/dataset/FormatPhoto/P17.png"
OUT_DIR_BASE="./results/API_4o_Prompt7_P17_D1"
PYTHON_SCRIPT="python InfiniteYou/test.py"

# === 遍历文本中所有情绪:prompt ===
while IFS=: read -r EMOTION RAW_PROMPT
do
  # 去除前后的空格
  EMOTION=$(echo "$EMOTION" | xargs)
  PROMPT=$(echo "$RAW_PROMPT" | xargs)

  # 如果行为空或不是8个情绪之一则跳过
  if [[ -z "$EMOTION" || -z "$PROMPT" ]]; then
    continue
  fi

  # 输出目录按情绪分类
  OUT_DIR="${OUT_DIR_BASE}"
  echo "🔄 Running for emotion: $EMOTION"

  $PYTHON_SCRIPT \
    --emotion "$EMOTION" \
    --id_image "$ID_IMAGE" \
    --out_results_dir "$OUT_DIR" \
    --prompt "$PROMPT"

done < "$DESC_FILE"

# bash /SAN/intelsys/IdPreservProject/InfiniteYou/run_description.sh