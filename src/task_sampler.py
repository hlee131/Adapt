import json
import random
import os
from env.scene import SceneGenerator

class PPOTaskSampler:
    def __init__(self, config_path="cfg/split0_run_config_train.json", split="train"):
        """
        Initialize task sampler using existing config files from evaluation pipeline.
        
        Args:
            config_path: Path to config file (same format as used in run_eval.py)
            split: "train" for training scenes
        """
        self.split = split
        
        # Load the same config format used by run_eval.py
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        # Validate config has required fields (same as run_eval.py expects)
        required_fields = ["personas", "goals"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Config missing required field: {field}")
        
        # Initialize scene generator (same as in run_eval.py)
        self.scene_generator = SceneGenerator(split=split)
        
        print(f"Loaded task sampler with {len(self.config['personas'])} personas and {len(self.config['goals'])} goals")
        
    def sample_task_and_scene(self):
        """
        Sample a random task, persona, and scene for PPO training.
        
        Returns:
            tuple: (task, persona_id, scene_data)
        """
        # Randomly sample persona and task (unlike evaluation which is deterministic)
        persona_id = random.choice(self.config["personas"])
        task = random.choice(self.config["goals"])
        
        # Generate a random scene (same as evaluation)
        scene_data = self.scene_generator()
        
        return task, persona_id, scene_data
    
    def get_task_distribution(self):
        """Get the task distribution for debugging/analysis"""
        return {
            "personas": self.config["personas"],
            "goals": self.config["goals"],
            "total_combinations": len(self.config["personas"]) * len(self.config["goals"])
        }