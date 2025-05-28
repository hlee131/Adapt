#!/usr/bin/env python3
"""
PPO Training Script for ADAPT
"""

import argparse
import os
import sys
import torch
import wandb
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

from src.ppo_planner import LLMAgent_Planner_PPO
from src.Environment import Environment
from src.task_sampler import PPOTaskSampler
from src.ppo_reward import PPORewardFunction
from src.utils import _tokenizer as global_tokenizer


def setup_wandb(args):
    """Initialize wandb with proper configuration"""
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=f"ppo_split{args.split}_{args.run_name}" if args.run_name else None,
            config=vars(args),
            mode="online" if not args.wandb_offline else "offline"
        )
    else:
        print("Warning: No wandb project specified. Logging disabled.")


def get_ppo_config(args):
    return PPOConfig(
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_ppo_epochs=args.ppo_epochs,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        cliprange=args.cliprange,
        cliprange_value=args.cliprange_value,
        gamma=args.gamma,
        lam=args.lam,
        kl_coef=args.target_kl,
        optimize_cuda_cache=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="wandb" if args.wandb_project else None,
    )


def load_models(config, args):
    """Load policy and reference models"""
    print(f"Loading models from {args.model_name}")
    
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "use_cache": False,
    }
    
    if args.use_4bit:
        model_kwargs.update({
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_quant_type": "nf4",
        })
    
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name, **model_kwargs
    )
    
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name, **model_kwargs
    )
    
    return policy_model, ref_model


def setup_tokenizer(model_name):
    """Setup tokenizer with proper padding token"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def save_checkpoint(ppo_trainer, step, checkpoint_dir, best_reward=None):
    """Save model checkpoint"""
    checkpoint_path = Path(checkpoint_dir) / f"ppo_step_{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    ppo_trainer.save_pretrained(str(checkpoint_path))
    
    metadata = {
        "step": step,
        "best_reward": best_reward,
    }
    
    with open(checkpoint_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved checkpoint to {checkpoint_path}")
    return str(checkpoint_path)


def train_ppo(args):
    """Main PPO training loop"""
    
    # Setup
    setup_wandb(args)
    config = get_ppo_config(args)
    
    # Load models and tokenizer
    policy_model, ref_model = load_models(config, args)
    tokenizer = setup_tokenizer(args.model_name)
    
    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        args=config,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=ref_model,
    )

    # Initialize components
    ppo_planner_agent = LLMAgent_Planner_PPO(
        persona_id="PPO_Agent",
        policy_model=ppo_trainer.model,
        critic_model=ppo_trainer.critic,
        tokenizer_instance=tokenizer,
        temperature_planner=args.temperature_planner,
        no_ask_option=args.no_ask_option,
        user_info_with_summary=args.user_info_with_summary
    )
    
    task_sampler = PPOTaskSampler(
        config_path=args.config_path,
        split="train"
    )
    
    reward_function = PPORewardFunction(tokenizer, lambda_penalty=args.reward_lambda)
    
    print(f"Task distribution: {task_sampler.get_task_distribution()}")
    
    # Training metrics
    metrics_history = defaultdict(list)
    best_avg_reward = float('-inf')
    
    # Training loop
    for ppo_step in tqdm(range(args.max_steps), desc="PPO Training"):
        # Memory cleanup
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
                
                # Initialize environment and agent
                env = Environment(scene_data)
                rollout_past = []
                ppo_planner_agent.persona_id = persona_id
                ppo_planner_agent.reset()
                
                episode_queries = []
                episode_responses = []
                episode_step_rewards = []
                
                # Episode rollout
                for step in range(args.max_episode_length):
                    try:
                        query_tensor, response_tokens, value, interaction_data = ppo_planner_agent.collect_rollout_step(
                            env, rollout_past, current_task
                        )
                        
                        # Tensor handling
                        query_tensor = query_tensor.squeeze(0) if len(query_tensor.shape) > 1 else query_tensor
                        response_tokens = response_tokens.squeeze(0) if len(response_tokens.shape) > 1 else response_tokens
                        
                        query_tensor = query_tensor.detach().to(ppo_trainer.accelerator.device)
                        response_tokens = response_tokens.detach().to(ppo_trainer.accelerator.device)
                        
                        step_reward = reward_function.calculate_step_reward(interaction_data, env, rollout_past)
                        
                        episode_queries.append(query_tensor.clone())
                        episode_responses.append(response_tokens.clone())
                        episode_step_rewards.append(step_reward)
                        rollout_past.append(interaction_data)
                        
                        if interaction_data["action_enum"] == "done":
                            break
                            
                    except Exception as e:
                        print(f"Warning: Episode {episode_idx} step {step} failed: {e}")
                        break

                # Calculate episode rewards
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
        
        # PPO update
        if len(batch_queries) > 0:
            try:
                stats = ppo_trainer.step(batch_queries, batch_responses, batch_rewards)
                
                # Log metrics
                if step_metrics:
                    avg_reward = np.mean(step_metrics['episode_reward'])
                    avg_task_completion = np.mean(step_metrics['task_completion'])
                    avg_preferences = np.mean(step_metrics['preferences_satisfied'])
                    avg_questions = np.mean(step_metrics['num_questions'])
                    avg_length = np.mean(step_metrics['episode_length'])
                    
                    metrics_history['avg_reward'].append(avg_reward)
                    metrics_history['task_completion'].append(avg_task_completion)
                    
                    # Log to wandb
                    if args.wandb_project:
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
                    
                    if ppo_step % args.log_frequency == 0:
                        print(f"Step {ppo_step}: Reward={avg_reward:.3f}, Task={avg_task_completion:.3f}, KL={stats.get('objective/kl', 0):.4f}")
                    
                    # Early stopping
                    if stats.get("objective/kl", 0) > config.target_kl * 10:
                        print(f"Early stopping: KL divergence too high ({stats.get('objective/kl', 0):.4f})")
                        break
                    
                    # Save best model
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        save_checkpoint(ppo_trainer, ppo_step, args.output_dir, best_avg_reward)
                        print(f"New best model saved with reward: {avg_reward:.3f}")
                        
            except Exception as e:
                print(f"Warning: PPO step {ppo_step} failed: {e}")
                continue
        
        # Regular checkpointing
        if ppo_step % args.checkpoint_frequency == 0 and ppo_step > 0:
            save_checkpoint(ppo_trainer, ppo_step, args.output_dir)
    
    # Save final model
    final_path = save_checkpoint(ppo_trainer, args.max_steps, args.output_dir, best_avg_reward)
    
    if args.wandb_project:
        wandb.finish()
    
    print("Training completed!")
    print(f"Best average reward achieved: {best_avg_reward:.3f}")
    print(f"Final model saved to: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="PPO Training for ADAPT")
    
    # Model and data
    parser.add_argument("--model_name", type=str, required=True, help="Base model path or HF model name")
    parser.add_argument("--config_path", type=str, default="cfg/split0_run_config_train.json", help="Task config file")
    parser.add_argument("--split", type=int, default=0, choices=[0,1,2,3], help="Cross-validation split")
    parser.add_argument("--output_dir", type=str, default="ppo_checkpoints", help="Output directory")
    
    # PPO hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1.41e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=1, help="Mini batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--ppo_epochs", type=int, default=3, help="PPO epochs per batch")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--vf_coef", type=float, default=0.25, help="Value function coefficient")
    parser.add_argument("--cliprange", type=float, default=0.15, help="PPO clip range")
    parser.add_argument("--cliprange_value", type=float, default=0.15, help="Value clip range")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.9, help="GAE lambda")
    parser.add_argument("--target_kl", type=float, default=0.005, help="Target KL divergence")
    
    # Training settings
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum PPO steps")
    parser.add_argument("--max_episode_length", type=int, default=20, help="Maximum episode length")
    parser.add_argument("--checkpoint_frequency", type=int, default=25, help="Checkpoint save frequency")
    parser.add_argument("--log_frequency", type=int, default=5, help="Logging frequency")
    
    # Agent settings
    parser.add_argument("--temperature_planner", type=float, default=0.7, help="Planner temperature")
    parser.add_argument("--reward_lambda", type=float, default=0.01, help="Reward penalty lambda")
    parser.add_argument("--no_ask_option", action="store_true", help="Disable asking questions")
    parser.add_argument("--user_info_with_summary", action="store_true", help="Use summary in user info")
    
    # Memory optimization
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    
    # Logging
    parser.add_argument("--wandb_project", type=str, default="adapt-ppo", help="Wandb project name")
    parser.add_argument("--wandb_offline", action="store_true", help="Run wandb offline")
    parser.add_argument("--run_name", type=str, help="Run name for wandb")
    
    args = parser.parse_args()
    
    # Update config path based on split
    if "split0" in args.config_path:
        args.config_path = args.config_path.replace("split0", f"split{args.split}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Start training
    train_ppo(args)


if __name__ == "__main__":
    main()