import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from tqdm import tqdm

from src.ppo_planner import LLMAgent_Planner_PPO
from src.Environment import Environment
from src.task_sampler import PPOTaskSampler
from src.ppo_reward import PPORewardFunction
from src.utils import _tokenizer as global_tokenizer

# --- Configuration ---
ppo_config_params = {
    "model_name": "path/to/your/sft_tuned_model",  # Your SFT model
    "learning_rate": 1.41e-5,
    "batch_size": 64,
    "mini_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "ppo_epochs": 4,
    "log_with": "wandb",
}
config = PPOConfig(**ppo_config_params)

# --- Model Loading ---
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True
)

ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True
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
    persona_id="PPO_Agent",  # This will be updated per episode
    policy_model=ppo_trainer.model,
    critic_model=ppo_trainer.critic,
    tokenizer_instance=tokenizer,
    temperature_planner=0.7,
    no_ask_option=False,
    user_info_with_summary=False
)

# Initialize task sampler (using same config format as evaluation)
task_sampler = PPOTaskSampler(
    config_path="cfg/split0_run_config_train.json",  # Same as used in run_eval.py
    split="train"
)

# Print task distribution for verification
print("Task distribution:", task_sampler.get_task_distribution())

# Initialize reward function
reward_function = PPORewardFunction(tokenizer, lambda_penalty=0.01)

# --- Training Loop ---
max_ppo_steps = 1000
max_episode_length = 50

for ppo_step in tqdm(range(max_ppo_steps)):
    queries = []
    responses = []
    rewards = []
    values = []
    
    for episode in range(config.batch_size):
        # Sample task and scene using existing evaluation pipeline approach
        current_task, persona_id, scene_data = task_sampler.sample_task_and_scene()
        
        # Initialize environment (same as run_eval.py -> run_eval_interaction)
        env = Environment(scene_data)
        rollout_past = []
        
        # Update agent persona for this episode
        ppo_planner_agent.persona_id = persona_id
        ppo_planner_agent.reset()
        
        episode_queries = []
        episode_responses = []
        episode_rewards = []
        episode_values = []
        
        for step in range(max_episode_length):
            # Collect one step (your existing method)
            query_tensor, response_tokens, value, interaction_data = ppo_planner_agent.collect_rollout_step(
                env, rollout_past, current_task
            )
            
            # Calculate step reward
            step_reward = reward_function.calculate_step_reward(interaction_data, env, rollout_past)
            
            episode_queries.append(query_tensor)
            episode_responses.append(response_tokens)
            episode_rewards.append(step_reward)
            episode_values.append(value)
            
            # Update rollout history
            rollout_past.append(interaction_data)
            
            # Check for episode termination
            if interaction_data["action_enum"] == "done" or step == max_episode_length - 1:
                break
        
        # Calculate trajectory reward and update final rewards
        rollout_data = {
            "task": current_task,
            "persona_id": persona_id,
            "rollout": rollout_past,
            "initial_scene": env.initial_scene,  # Assuming env has this
            "finished": rollout_past[-1]["action_enum"] == "done" if rollout_past else False
        }
        
        final_rewards = reward_function.calculate_episode_rewards(rollout_data, episode_rewards)
        
        # Convert to tensors and add to batch
        for q, r, reward, v in zip(episode_queries, episode_responses, final_rewards, episode_values):
            queries.append(q)
            responses.append(r)
            rewards.append(torch.tensor(reward, device=ppo_trainer.accelerator.device))
            values.append(v)
    
    # Perform PPO update
    stats = ppo_trainer.step(queries, responses, rewards)
    ppo_trainer.log_stats(stats, batch={"queries": queries, "responses": responses}, rewards=rewards)
    
    # Log additional metrics
    if ppo_step % 10 == 0:
        avg_reward = torch.stack(rewards).mean().item()
        print(f"Step {ppo_step}, Average Reward: {avg_reward:.3f}")

        # Get detailed reward breakdown for the last episode
        if 'rollout_data' in locals():
            reward_info = reward_function.get_reward_info(rollout_data)
            print(f"  Task Completion: {reward_info['task_completion']:.3f}")
            print(f"  Preferences Satisfied: {reward_info['preferences_satisfied']}")
            print(f"  Questions Asked: {reward_info['num_questions']}")
            print(f"  Question Penalty: {reward_info['question_penalty']:.3f}")
            print(f"  Response Penalty: {reward_info['response_penalty']:.3f}")
            print(f"  Total Trajectory Reward: {reward_info['total_trajectory_reward']:.3f}")

# Save the trained model
ppo_trainer.save_pretrained("path/to/your/ppo_trained_model")