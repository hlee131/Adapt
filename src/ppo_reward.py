import torch
import json
from src.convert_and_evaluate import process_trace
from src.Environment import Environment

class PPORewardFunction:
    def __init__(self, tokenizer, lambda_penalty=0.01, reward_shaping=True):
        self.tokenizer = tokenizer
        self.lambda_penalty = lambda_penalty
        self.reward_shaping = reward_shaping
        
    def calculate_step_reward(self, interaction_data, env, rollout_past):
        """Calculate immediate step reward with shaped rewards."""
        reward = 0.0
        
        # Success/failure shaping
        if interaction_data["success"]:
            reward += 0.1
        else:
            reward -= 0.05
            
        # Question efficiency penalty
        if interaction_data["action_enum"] == "ask":
            reward -= self.lambda_penalty
            
        # Progress reward (optional reward shaping)
        if self.reward_shaping:
            if interaction_data["action_enum"] in ["pickup", "place", "open", "close"]:
                reward += 0.05  # Small reward for meaningful actions
                
        return reward
    
    def calculate_episode_rewards(self, rollout_data, step_rewards):
        """Calculate final trajectory rewards using ADAPT evaluation."""
        try:
            # Ensure proper format for process_trace
            if "sim_steps" not in rollout_data:
                rollout_data["sim_steps"] = len(rollout_data["rollout"])
            if "episode_length" not in rollout_data:
                rollout_data["episode_length"] = len(rollout_data["rollout"])
            
            # Use existing ADAPT evaluation
            from src.convert_and_evaluate import process_trace
            results_dict, _ = process_trace(rollout_data)
            
            # Main reward components based on ADAPT metrics
            task_completion_reward = results_dict.get("task_completion_fraction", 0.0) * 10.0
            preference_reward = results_dict.get("reward_fraction", 0.0) * 5.0
            
            # Token-based penalties
            total_question_tokens = 0
            total_response_tokens = 0
            
            for step in rollout_data["rollout"]:
                if step.get("action_enum") == "ask":
                    question_text = step.get("action", "")
                    if question_text.startswith("Ask "):
                        question_text = question_text[4:]
                    question_tokens = len(self.tokenizer.encode(question_text, add_special_tokens=False))
                    total_question_tokens += question_tokens
                    
                    if step.get("user_feedback"):
                        response_tokens = len(self.tokenizer.encode(step["user_feedback"], add_special_tokens=False))
                        total_response_tokens += response_tokens
            
            question_penalty = -self.lambda_penalty * total_question_tokens / 100.0  # Normalize
            response_penalty = -self.lambda_penalty * total_response_tokens / 100.0  # Normalize
            
            trajectory_reward = task_completion_reward + preference_reward + question_penalty + response_penalty
            
        except Exception as e:
            print(f"Warning: process_trace failed: {e}")
            # Fallback simple reward
            if rollout_data.get("finished", False):
                trajectory_reward = 1.0
            else:
                trajectory_reward = -1.0
        
        # Distribute trajectory reward (only to final step to reduce variance)
        final_rewards = []
        for i, step_reward in enumerate(step_rewards):
            if i == len(step_rewards) - 1:  # Last step gets trajectory reward
                final_reward = step_reward + trajectory_reward
            else:
                final_reward = step_reward
            final_rewards.append(final_reward)
            
        return final_rewards
    
    def get_reward_info(self, rollout_data):
        """Get detailed reward breakdown for logging."""
        try:
            from src.convert_and_evaluate import process_trace
            results_dict, _ = process_trace(rollout_data)
            
            num_questions = sum(1 for step in rollout_data["rollout"] 
                              if step.get("action_enum") == "ask")
            
            return {
                "task_completion": results_dict.get("task_completion_fraction", 0.0),
                "preferences_satisfied": results_dict.get("reward_fraction", 0.0),
                "num_questions": num_questions,
                "total_reward": results_dict.get("task_completion_fraction", 0.0) * 10.0 + 
                              results_dict.get("reward_fraction", 0.0) * 5.0 - 
                              self.lambda_penalty * num_questions
            }
        except Exception as e:
            print(f"Warning: get_reward_info failed: {e}")
            return {
                "task_completion": 0.0,
                "preferences_satisfied": 0.0,
                "num_questions": 0,
                "total_reward": 0.0
            }
