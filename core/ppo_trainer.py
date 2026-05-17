import json

import torch
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


def get_rewards(queries, responses):
    """
    Reward function that parses the response as JSON and penalizes if it violates
    the MarketMayhemLedger schema constraints.
    """
    rewards = []
    for response in responses:
        try:
            parsed = json.loads(response)
            # Check for the required 'data_points' array
            if "data_points" not in parsed or not isinstance(
                parsed["data_points"], list
            ):
                rewards.append(torch.tensor(-1.0))
                continue

            is_valid = True
            for point in parsed["data_points"]:
                if not isinstance(point, dict):
                    is_valid = False
                    break
                # Check for required keys in each data point
                if not all(
                    key in point
                    for key in [
                        "variable_node",
                        "market_level_value",
                        "primary_model_target",
                    ]
                ):
                    is_valid = False
                    break

                # Check enum value for primary_model_target
                if point["primary_model_target"] not in ["PD", "LGD", "DCF", "EV"]:
                    is_valid = False
                    break

            if is_valid:
                rewards.append(torch.tensor(1.0))
            else:
                rewards.append(torch.tensor(-1.0))

        except json.JSONDecodeError:
            # Penalize heavily if the output is not valid JSON
            rewards.append(torch.tensor(-1.0))

    return rewards


# Mock dataset and data collator for the seed
your_prompt_dataset = [
    {"input_ids": torch.tensor([1, 2, 3]), "query": "Test query 1"},
    {"input_ids": torch.tensor([4, 5, 6]), "query": "Test query 2"},
]


def your_data_collator(data):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in data]),
        "query": [x["query"] for x in data],
    }


# 1. Configuration for the Harness
config = PPOConfig(
    model_name="mistralai/Mistral-7B-v0.1",
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=4,
    optimize_cuda_cache=True,
)

# 2. Setup LoRA to handle the 4-model memory problem
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# 3. Load the Actor and Critic (Value Head) together
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    peft_config=lora_config,
    device_map="auto",
    load_in_8bit=True,  # Requires bitsandbytes
)

# The Reference model is automatically handled by trl
# (it disables the LoRA adapters to get reference probabilities)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# 4. Initialize the PPO Trainer
ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=None,  # Automatically inferred via PEFT
    tokenizer=tokenizer,
    dataset=your_prompt_dataset,  # Needs to be defined
    data_collator=your_data_collator,
)

# 5. The Core RLHF Loop
for _epoch, batch in enumerate(ppo_trainer.dataloader):
    query_tensors = batch["input_ids"]

    # A. Generate Responses (Rollout)
    # We pass a list of tensors to ppo_trainer.generate as expected
    response_tensors = ppo_trainer.generate(list(query_tensors), max_new_tokens=128)
    batch["response"] = tokenizer.batch_decode(response_tensors)

    # B. Calculate Rewards
    # (In reality, you pass batch["response"] through a separate Reward Model API or function)
    # For this seed, we assume a function get_rewards() returns a list of float tensors
    rewards = get_rewards(batch["query"], batch["response"])

    # C. PPO Step (Calculates Advantages, KL Divergence, and updates weights)
    stats = ppo_trainer.step(list(query_tensors), response_tensors, rewards)

    # D. Telemetry & Provenance Logging
    ppo_trainer.log_stats(stats, batch, rewards)

# 6. Save the aligned model
ppo_trainer.save_pretrained("my-aligned-chatbot")
