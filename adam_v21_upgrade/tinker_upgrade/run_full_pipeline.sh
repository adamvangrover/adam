#!/bin/bash
# This script runs the full Adam v21.0 upgrade pipeline.
# It assumes you have already run 'setup_env.sh' and
# activated the virtual environment with 'source venv-tinker/bin/activate'.
echo "--- [Adam v21.0 Upgrade] STARTING ---"
# --- Step 1: Verify Connection ---
echo "\n Verifying Tinker API Connection..."
python tinker_upgrade/check_connection.py
# --- Step 2: Stage 1 (Tool Use) Data Gen ---
echo "\n Generating Stage 1 'Tool Use' (Neo4j) *expanded* dataset..."
# This script is now assumed to generate the expanded dataset from Sec 2.2
python tinker_upgrade/stage1_tool_use_gen.py
# --- Step 3: Stage 1 (Tool Use) Training ---
echo "\n Starting Stage 1 (Neo4j Cypher Agent) Training Job..."
python tinker_upgrade/stage1_train_cypher.py
# OUTPUT: 'adam_cypher_lora_v1' adapter
# --- Step 4: Stage 2 (Distillation) ---
echo "\n Starting Stage 2 'Distillation'..."
echo "  [4a] Generating Teacher data from Qwen-235B-MoE..."
# This is the new, fully-implemented script from Sec 3.3
python tinker_upgrade/stage2_create_data.py
echo "  [4b] Training Student (Llama-8B) on distilled data..."
# This is the new, fully-implemented script from Sec 3.4
python tinker_upgrade/stage2_train_student.py
# OUTPUT: 'adam_distilled_mind_v1' adapter
# --- Step 5: Stage 3 (DPO) ---
echo "\n Starting Stage 3 'DPO Alignment'..."
echo "  [5a] Generating *expanded* DPO preference dataset..."
# This script is now assumed to generate the expanded dataset from Sec 4.2
python tinker_upgrade/stage3_dpo_prep.py
echo "  [5b] Training 'Soul' adapter via DPO..."
# This is the new, fully-implemented script from Sec 4.3
python tinker_upgrade/stage3_train_dpo.py
# OUTPUT: 'adam_aligned_soul_v1' adapter
echo "\n--- [Adam v21.0 Upgrade] FULL TRAINING PIPELINE COMPLETE ---"
echo "All three adapters have been trained and saved to Tinker cloud storage."
