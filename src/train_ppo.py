import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from tqdm import tqdm
import wandb
import os
from collections import defaultdict
import numpy as np

from src.ppo_planner import LLMAgent_Planner_PPO
from src.Environment import Environment
from src.task_sampler import PPOTaskSampler
from src.ppo_reward import PPORewardFunction
from src.utils import _tokenizer as global_tokenizer

# --- Improved Configuration ---
ppo_config_params = {
    "model_name": "path/to/your/sft_tuned_model",
    "learning_rate": 1.41e-5,
    "batch_size": 16,  # Further reduced for stability
    "mini_batch_size": 2,  # Smaller mini-batches
    "gradient_accumulation_steps": 4,  # Compensate with more accumulation
    "ppo_epochs": 3,  # Reduced epochs per batch
    "log_with": "wandb",
    "max_grad_norm": 0.5,  # More conservative clipping
    "vf_coef": 0.25,  # Higher value function coefficient
    "cliprange": 0.15,  # More conservative clipping
    "cliprange_value": 0.15,
    "gamma": 0.95,  # Slightly lower discount for shorter episodes
    "lam": 0.9,  # GAE lambda
    "target_kl": 0.005,  # More conservative KL target
    "optimize_cuda_cache": True,
    "remove_unused_columns": False,
    "dataloader_pin_memory": False,  # Reduce memory usage
}

config = PPOConfig(**ppo_config_params)

# Initialize wandb
wandb.init(project="adapt-ppo", config=ppo_config_params)

# --- Model Loading with Memory Optimization ---
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    use_cache=False  # Disable KV cache to save memory
)

ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    use_cache=False
)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Initialize PPO Trainer
ppo_trainer = PPOTrainer(
    config=config, 
    model=policy_model, 
    ref_model=ref_model, 
    tokenizer=tokenizer
)

# Initialize PPO Planner Agent
ppo_planner_agent = LLMAgent_Planner_PPO(
    persona_id="PPO_Agent",
    policy_model=ppo_trainer.model,
    critic_model=ppo_trainer.critic,
    tokenizer_instance=tokenizer,
    temperature_planner=0.7,
    no_ask_option=False,
    user_info_with_summary=False
)

# Initialize task sampler
task_sampler = PPOTaskSampler(
    config_path="cfg/split0_run_config_train.json",
    split="train"
)

print("Task distribution:", task_sampler.get_task_distribution())

# Initialize reward function
reward_function = PPORewardFunction(tokenizer, lambda_penalty=0.01)

# --- Training Loop with Best Practices ---
max_ppo_steps = 1000
max_episode_length = 30  # Reduced for stability
validation_frequency = 50
checkpoint_frequency = 100

# Metrics tracking
metrics_history = defaultdict(list)
best_avg_reward = float('-inf')

for ppo_step in tqdm(range(max_ppo_steps)):
    # Clear GPU cache periodically
    if ppo_step % 10 == 0:
        torch.cuda.empty_cache()
    
    batch_queries = []
    batch_responses = []
    batch_rewards = []
    step_metrics = defaultdict(list)
    
    # Collect batch episodes
    for episode_idx in range(config.batch_size):
        try:
            # Sample task and scene
            current_task, persona_id, scene_data = task_sampler.sample_task_and_scene()
            
            # Initialize environment
            env = Environment(scene_data)
            rollout_past = []
            
            # Update agent persona and reset
            ppo_planner_agent.persona_id = persona_id
            ppo_planner_agent.reset()
            
            episode_queries = []
            episode_responses = []
            episode_step_rewards = []
            
            for step in range(max_episode_length):
                try:
                    query_tensor, response_tokens, value, interaction_data = ppo_planner_agent.collect_rollout_step(
                        env, rollout_past, current_task
                    )
                    
                    # Improved tensor handling
                    query_tensor = query_tensor.squeeze(0) if len(query_tensor.shape) > 1 else query_tensor
                    response_tokens = response_tokens.squeeze(0) if len(response_tokens.shape) > 1 else response_tokens
                    
                    # Ensure tensors are on correct device and detached
                    query_tensor = query_tensor.detach().to(ppo_trainer.accelerator.device)
                    response_tokens = response_tokens.detach().to(ppo_trainer.accelerator.device)
                    
                    step_reward = reward_function.calculate_step_reward(interaction_data, env, rollout_past)
                    
                    episode_queries.append(query_tensor.clone())  # Clone to avoid references
                    episode_responses.append(response_tokens.clone())
                    episode_step_rewards.append(step_reward)
                    
                    rollout_past.append(interaction_data)
                    
                    if interaction_data["action_enum"] == "done":
                        break
                        
                except Exception as e:
                    print(f"Warning: Episode {episode_idx} step {step} failed: {e}")
                    break

            # Calculate episode rewards if we have valid rollout
            if rollout_past and len(episode_queries) > 0:
                rollout_data = {
                    "task": current_task,
                    "persona_id": persona_id,
                    "rollout": rollout_past,
                    "finished": rollout_past[-1]["action_enum"] == "done",
                    "sim_steps": len(rollout_past),
                    "episode_length": len(rollout_past)
                }
                
                final_rewards = reward_function.calculate_episode_rewards(rollout_data, episode_step_rewards)
                
                # Collect metrics
                reward_info = reward_function.get_reward_info(rollout_data)
                step_metrics['episode_reward'].append(sum(final_rewards))
                step_metrics['task_completion'].append(reward_info['task_completion'])
                step_metrics['preferences_satisfied'].append(reward_info['preferences_satisfied'])
                step_metrics['num_questions'].append(reward_info['num_questions'])
                step_metrics['episode_length'].append(len(rollout_past))
                
                # Add to batch
                for q, r, reward in zip(episode_queries, episode_responses, final_rewards):
                    batch_queries.append(q)
                    batch_responses.append(r)
                    batch_rewards.append(torch.tensor(reward, dtype=torch.float32, device=ppo_trainer.accelerator.device))
                    
        except Exception as e:
            print(f"Warning: Episode {episode_idx} failed completely: {e}")
            continue
    
    # Perform PPO update if we have valid data
    if len(batch_queries) > 0:
        try:
            stats = ppo_trainer.step(batch_queries, batch_responses, batch_rewards)
            
            # Log comprehensive metrics
            if step_metrics:
                avg_reward = np.mean(step_metrics['episode_reward'])
                avg_task_completion = np.mean(step_metrics['task_completion'])
                avg_preferences = np.mean(step_metrics['preferences_satisfied'])
                avg_questions = np.mean(step_metrics['num_questions'])
                avg_length = np.mean(step_metrics['episode_length'])
                
                metrics_history['avg_reward'].append(avg_reward)
                metrics_history['task_completion'].append(avg_task_completion)
                metrics_history['preferences_satisfied'].append(avg_preferences)
                metrics_history['num_questions'].append(avg_questions)
                
                # Log to wandb
                wandb.log({
                    "step": ppo_step,
                    "avg_reward": avg_reward,
                    "task_completion": avg_task_completion,
                    "preferences_satisfied": avg_preferences,
                    "num_questions": avg_questions,
                    "episode_length": avg_length,
                    "kl_div": stats.get("objective/kl", 0),
                    "policy_loss": stats.get("ppo/loss/policy", 0),
                    "value_loss": stats.get("ppo/loss/value", 0),
                })
                
                if ppo_step % 10 == 0:
                    print(f"Step {ppo_step}: Reward={avg_reward:.3f}, Task={avg_task_completion:.3f}, Pref={avg_preferences:.3f}, Q={avg_questions:.1f}")
                
                # Early stopping based on KL divergence
                if stats.get("objective/kl", 0) > config.target_kl * 10:  # 10x threshold for early stop
                    print(f"Early stopping: KL divergence too high ({stats.get('objective/kl', 0):.4f})")
                    break
                    
                # Save best model
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    ppo_trainer.save_pretrained("best_ppo_model")
                    print(f"New best model saved with reward: {avg_reward:.3f}")
                    
        except Exception as e:
            print(f"Warning: PPO step {ppo_step} failed: {e}")
            continue
    
    # Save checkpoint periodically
    if ppo_step % checkpoint_frequency == 0 and ppo_step > 0:
        checkpoint_path = f"checkpoints/ppo_step_{ppo_step}"
        os.makedirs(checkpoint_path, exist_ok=True)
        ppo_trainer.save_pretrained(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

# Save final model
ppo_trainer.save_pretrained("final_ppo_model")
wandb.finish()

print("Training completed!")
print(f"Best average reward achieved: {best_avg_reward:.3f}")