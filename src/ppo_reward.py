import torch
import json
from src.convert_and_evaluate import process_trace
from src.Environment import Environment

class PPORewardFunction:
    def __init__(self, tokenizer, lambda_penalty=0.01):
        """
        Initialize PPO reward function.
        
        Args:
            tokenizer: Tokenizer for counting tokens in questions/responses
            lambda_penalty: Lambda parameter for token penalty (default: 0.01)
        """
        self.tokenizer = tokenizer
        self.lambda_penalty = lambda_penalty
        
    def calculate_step_reward(self, interaction_data, env, rollout_past):
        """
        Calculate immediate step reward (typically sparse, mainly for questions).
        
        Args:
            interaction_data: Current step interaction data
            env: Environment instance
            rollout_past: Previous rollout steps
            
        Returns:
            float: Step reward
        """
        step_reward = 0.0
        
        # Small negative reward for asking questions (immediate feedback)
        if interaction_data["action_enum"] == "ask":
            # Extract question text from action
            question_text = interaction_data.get("action", "")
            if question_text.startswith("Ask "):
                question_text = question_text[4:]  # Remove "Ask " prefix
            
            # Count question tokens and apply penalty
            question_tokens = len(self.tokenizer.encode(question_text, add_special_tokens=False))
            step_reward -= self.lambda_penalty * (question_tokens ** 2)
            
        # Small negative reward for failed actions
        if not interaction_data["success"]:
            step_reward -= 0.1
            
        return step_reward
    
    def calculate_episode_rewards(self, rollout_data, step_rewards):
        """
        Calculate final trajectory reward and redistribute across steps.
        
        Args:
            rollout_data: Complete episode data in format expected by process_trace
            step_rewards: List of step rewards from calculate_step_reward
            
        Returns:
            list: Final rewards for each step
        """
        # Ensure rollout_data has required fields for process_trace
        if "initial_scene" not in rollout_data:
            # Try to get from environment if available
            rollout_data["initial_scene"] = json.dumps({})  # Fallback
            
        # Add missing fields that process_trace expects
        if "num_questions" not in rollout_data:
            rollout_data["num_questions"] = sum(1 for step in rollout_data["rollout"] 
                                              if step.get("action_enum") == "ask")
        if "num_corrections" not in rollout_data:
            rollout_data["num_corrections"] = 0
        if "sim_steps" not in rollout_data:
            rollout_data["sim_steps"] = len(rollout_data["rollout"])
        if "episode_length" not in rollout_data:
            rollout_data["episode_length"] = len(rollout_data["rollout"])
            
        # Calculate terminal reward using existing evaluation infrastructure
        try:
            results_dict, _ = process_trace(rollout_data)
            
            # Task completion indicator I(a)
            task_completed = rollout_data.get("finished", False) and \
                           rollout_data["rollout"][-1]["action_enum"] == "done" if rollout_data["rollout"] else False
            I_a = 1.0 if task_completed else 0.0
            
            # Number of preferences satisfied
            num_prefs_satisfied = results_dict.get("max_penalty", 0) - results_dict.get("penalty", 0)
            
        except Exception as e:
            print(f"Warning: process_trace failed with error {e}, using fallback rewards")
            I_a = 1.0 if rollout_data.get("finished", False) else 0.0
            num_prefs_satisfied = 0
        
        # Calculate token penalties for questions and responses
        total_question_tokens = 0
        total_response_tokens = 0
        
        for step in rollout_data["rollout"]:
            if step.get("action_enum") == "ask":
                # Count question tokens
                question_text = step.get("action", "")
                if question_text.startswith("Ask "):
                    question_text = question_text[4:]
                question_tokens = len(self.tokenizer.encode(question_text, add_special_tokens=False))
                total_question_tokens += question_tokens
                
                # Count response tokens if user provided feedback
                if step.get("user_feedback"):
                    response_tokens = len(self.tokenizer.encode(step["user_feedback"], add_special_tokens=False))
                    total_response_tokens += response_tokens
        
        # Calculate total trajectory reward
        trajectory_reward = (I_a + 
                           0.3 * num_prefs_satisfied - 
                           self.lambda_penalty * (total_question_tokens ** 2) - 
                           self.lambda_penalty * (total_response_tokens ** 2))
        
        # Redistribute trajectory reward across steps
        # Common approach: give most reward at the end, but some to intermediate steps
        final_rewards = []
        num_steps = len(step_rewards)
        
        if num_steps == 0:
            return []
            
        for i, step_reward in enumerate(step_rewards):
            if i == num_steps - 1:  # Last step gets most of trajectory reward
                final_reward = step_reward + trajectory_reward
            else:  # Intermediate steps get smaller portion
                final_reward = step_reward + (trajectory_reward * 0.1 / max(1, num_steps - 1))
            
            final_rewards.append(final_reward)
        
        return final_rewards
    
    def get_reward_info(self, rollout_data):
        """
        Get detailed reward breakdown for logging/debugging.
        
        Args:
            rollout_data: Complete episode data
            
        Returns:
            dict: Detailed reward breakdown
        """
        try:
            results_dict, _ = process_trace(rollout_data)
            
            task_completed = rollout_data.get("finished", False) and \
                           rollout_data["rollout"][-1]["action_enum"] == "done" if rollout_data["rollout"] else False
            I_a = 1.0 if task_completed else 0.0
            num_prefs_satisfied = results_dict.get("max_penalty", 0) - results_dict.get("penalty", 0)
            
            # Calculate token counts
            total_question_tokens = 0
            total_response_tokens = 0
            num_questions = 0
            
            for step in rollout_data["rollout"]:
                if step.get("action_enum") == "ask":
                    num_questions += 1
                    question_text = step.get("action", "")
                    if question_text.startswith("Ask "):
                        question_text = question_text[4:]
                    question_tokens = len(self.tokenizer.encode(question_text, add_special_tokens=False))
                    total_question_tokens += question_tokens
                    
                    if step.get("user_feedback"):
                        response_tokens = len(self.tokenizer.encode(step["user_feedback"], add_special_tokens=False))
                        total_response_tokens += response_tokens
            
            trajectory_reward = (I_a + 
                               0.3 * num_prefs_satisfied - 
                               self.lambda_penalty * (total_question_tokens ** 2) - 
                               self.lambda_penalty * (total_response_tokens ** 2))
            
            return {
                "task_completion": I_a,
                "preferences_satisfied": num_prefs_satisfied,
                "preference_reward": 0.3 * num_prefs_satisfied,
                "num_questions": num_questions,
                "question_tokens": total_question_tokens,
                "response_tokens": total_response_tokens,
                "question_penalty": -self.lambda_penalty * (total_question_tokens ** 2),
                "response_penalty": -self.lambda_penalty * (total_response_tokens ** 2),
                "total_trajectory_reward": trajectory_reward,
                "preferences_violated": results_dict.get("penalty", 0),
                "task_completion_fraction": results_dict.get("task_completion_fraction", 0)
            }
            
        except Exception as e:
            print(f"Warning: Could not get detailed reward info: {e}")
            return {
                "task_completion": 1.0 if rollout_data.get("finished", False) else 0.0,
                "preferences_satisfied": 0,
                "preference_reward": 0,
                "num_questions": 0,
                "question_tokens": 0,
                "response_tokens": 0,
                "question_penalty": 0,
                "response_penalty": 0,
                "total_trajectory_reward": 1.0 if rollout_data.get("finished", False) else 0.0,
                "preferences_violated": 0,
                "task_completion_fraction": 0
            }