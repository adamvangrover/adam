#!/bin/bash
# This script runs the full Adam v21.0 upgrade pipeline.
# It assumes 'setup_env.sh' has been run and the
# venv is active: 'source venv-tinker/bin/activate'.

# --- Logging Start ---
echo "================================================="
echo "  ADAM v21.0 FULL TRAINING PIPELINE (START)    "
echo "================================================="
echo "Timestamp: $(date)"
echo ""

# --- Step 1: Verify Connection ---
echo "--- [Step 1/6] Verifying Tinker API Connection... ---"
python tinker_upgrade/check_connection.py
echo "--- [Step 1/6] Verification Complete ---"
echo ""

# --- Step 2: Stage 1 (Tool Use) Data Gen ---
echo "--- [Step 2/6] Generating Stage 1 'Tool Use' (Neo4j) Dataset... ---"
python tinker_upgrade/stage1_tool_use_gen.py
echo "--- [Step 2/6] Data Generation Complete ---"
echo ""

# --- Step 3: Stage 1 (Tool Use) Training ---
echo "--- [Step 3/6] Starting Stage 1 (Neo4j Cypher Agent) Training... ---"
python tinker_upgrade/stage1_train_cypher.py
echo "--- [Step 3/6] Stage 1 Training Complete ---"
echo "✅ OUTPUT: 'adam_cypher_lora_v1' (The Hands)"
echo ""

# --- Step 4: Stage 2 (Distillation) ---
echo "--- [Step 4/6] Starting Stage 2 'Cognitive Distillation'... ---"
echo "  [4a] Generating Teacher data from Qwen-235B-MoE..."
python tinker_upgrade/stage2_create_data.py
echo "  [4b] Training Student (Llama-8B) on distilled data..."
python tinker_upgrade/stage2_train_student.py
echo "--- [Step 4/6] Stage 2 Distillation Complete ---"
echo "✅ OUTPUT: 'adam_distilled_mind_v1' (The Mind)"
echo ""

# --- Step 5: Stage 3 (DPO) ---
echo "--- [Step 5/6] Starting Stage 3 'DPO Alignment'... ---"
echo "  [5a] Generating DPO preference dataset..."
python tinker_upgrade/stage3_dpo_prep.py
echo "  [5b] Training 'Soul' adapter via DPO..."
python tinker_upgrade/stage3_train_dpo.py
echo "--- [Step 5/6] Stage 3 DPO Alignment Complete ---"
echo "✅ OUTPUT: 'adam_aligned_soul_v1' (The Soul)"
echo ""

# --- Step 6: Merge Adapters for Production ---
echo "--- [Step 6/6] Merging 'Mind' and 'Soul' Adapters... ---"
python tinker_upgrade/merge_adapters.py
echo "--- [Step 6/6] Adapter Merge Complete ---"
echo "✅ FINAL OUTPUT: 'adam_final_agent_lora' (Merged Mind + Soul)"
echo ""


# --- Final Summary ---
echo "================================================="
echo "   ADAM v21.0 FULL TRAINING PIPELINE (COMPLETE)  "
echo "================================================="
echo "Timestamp: $(date)"
echo ""
echo "Final Deployable Artifacts:"
echo "-------------------------------------------------"
echo "1. adam_cypher_lora_v1"
echo "   - ROLE: Specialist Tool Use Agent (The Hands)"
echo "   - USAGE: Load on-demand for Neo4j database queries."
echo ""
echo "2. adam_final_agent_lora"
echo "   - ROLE: Composed Reasoning & Alignment Agent (Mind + Soul)"
echo "   - USAGE: Primary adapter for analysis, reasoning, and final response generation."
echo "-------------------------------------------------"
echo "All adapters have been trained and saved to Tinker cloud storage."
echo "Use the 'download_adapters.py' script to retrieve them for local deployment."
echo ""
