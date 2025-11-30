#!/bin/bash
# This script runs the full Adam v21.0 upgrade pipeline.
# It assumes 'setup_env.sh' has been run and the
# venv is active: 'source venv-tinker/bin/activate'.

echo "--- [Adam v21.0 Upgrade] STARTING ---"

# --- Step 1: Verify Connection ---
echo "\\n Verifying Tinker API Connection..."
python tinker_upgrade/check_connection.py

# --- Step 2: Stage 1 (Tool Use) Data Gen ---
echo "\\n Generating Stage 1 'Tool Use' (Neo4j) *expanded* dataset..."
# This script will be modified to generate the expanded dataset
python tinker_upgrade/stage1_tool_use_gen.py

# --- Step 3: Stage 1 (Tool Use) Training ---
echo "\\n Starting Stage 1 (Neo4j Cypher Agent) Training Job..."
python tinker_upgrade/stage1_train_cypher.py
# OUTPUT: 'adam_cypher_lora_v1'

# --- Step 4: Stage 2 (Distillation) ---
echo "\\n Starting Stage 2 'Distillation'..."
echo "  [4a] Generating Teacher data from Qwen-235B-MoE..."
python tinker_upgrade/stage2_create_data.py # NEW SCRIPT
echo "  [4b] Training Student (Llama-8B) on distilled data..."
python tinker_upgrade/stage2_train_student.py # NEW SCRIPT
# OUTPUT: 'adam_distilled_mind_v1'

# --- Step 5: Stage 3 (DPO) ---
echo "\\n Starting Stage 3 'DPO Alignment'..."
echo "  [5a] Generating *expanded* DPO preference dataset..."
python tinker_upgrade/stage3_dpo_prep.py
echo "  [5b] Training 'Soul' adapter via DPO..."
python tinker_upgrade/stage3_train_dpo.py # NEW SCRIPT
# OUTPUT: 'adam_aligned_soul_v1'

echo "\\n--- [Adam v21.0 Upgrade] FULL TRAINING PIPELINE COMPLETE ---"
