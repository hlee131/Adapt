import torch
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from src.LLMAgent import LLMAgent_Planner, prompt_from_rollout, create_grammar
from src.utils import planner_system_prompt

class LLMAgent_Planner_PPO(LLMAgent_Planner):
    def __init__(
        self,
        persona_id: str,
        policy_model,  # Pass the actual model instance
        critic_model,  # Pass the actual critic instance
        tokenizer_instance,  # Pass the tokenizer instance
        temperature_planner: float = 0.7,
        no_ask_option: bool = False,
        user_info_with_summary: bool = False,
        **kwargs
    ):
        # Manually set LLMAgent attributes to avoid loading a new model
        self.model = policy_model  # This is the PPO actor
        self.tokenizer = tokenizer_instance
        self.device = self.model.device
        self.model_in_path = getattr(policy_model.config, "_name_or_path", "ppo_model")
        self.name_or_path = self.model_in_path
        
        # Set default LLM parameters
        self.default_llm_params = {
            "max_new_tokens": 250,
            "temperature": 1.0,
            "sampling": False
        }
        
        # Set LLMAgent_Planner attributes
        self.agent_name = "Planner_PPO"
        self.persona_id = persona_id
        self.user_info = ""
        self.example_history = []
        self.max_actions = 4
        self.max_summaries = 4
        self.probability_thresh = 0
        self.temperature = temperature_planner
        self.no_ask_option = no_ask_option
        self.user_info_with_summary = user_info_with_summary
        
        # Store critic model
        self.critic = critic_model

    def reset(self):
        """Reset the planner state"""
        self.user_info = ""
        self.example_history = []

    def add_user_info(self, info):
        """Add user preference information"""
        if info is None: 
            return
        self.user_info = info

    def push_example(self, example_task, example_rollout):
        """Add example to history"""
        self.example_history.append((example_task, example_rollout))

    def get_action_and_value(self, env, rollout_past, task):
        """Generate action using PPO policy and get value from critic with error handling."""
        try:
            # 1. Construct prompt using existing evaluation pipeline logic
            prompt_msgs, _ = prompt_from_rollout(
                rollout_past,
                assistant="robot",
                skip=[],
                change_user_to=self.persona_id,
                skip_failed=True,
                action_only=True
            )
            
            # Add example history
            for i_ex, (example_task, example_rollout) in enumerate(self.example_history):
                prompt_msgs_ex, _ = prompt_from_rollout(
                    example_rollout,
                    assistant="robot",
                    skip=[],
                    change_user_to=self.persona_id,
                    skip_failed=True,
                    action_only=True,
                )
                prompt_msgs = (
                    [("user", f"Example {i_ex}, Task {example_task}:")]
                    + prompt_msgs_ex
                    + prompt_msgs
                )
            
            # Create system prompt
            system_prompt_msg = ("system", planner_system_prompt(
                self.persona_id, self.user_info, env, task, 
                no_ask_option=self.no_ask_option, action_only=True
            ))
            
            spoonfeeding_summary = f'What is the next step to complete the task: {task}?'
            if len(self.user_info) > 0 and self.user_info_with_summary:
                spoonfeeding_summary = f"Remember, {self.persona_id}'s preferences include: " + self.user_info + ". " + spoonfeeding_summary
            
            user_query_msg = ("user", spoonfeeding_summary)
            full_prompt_msgs = [system_prompt_msg] + prompt_msgs + [user_query_msg]
            
            # 2. Prepare tensors for PPO with proper tokenization
            query_text = self.tokenizer.apply_chat_template(
                [{"role": p[0], "content": p[1]} for p in full_prompt_msgs],
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Ensure consistent tokenization
            query_tensor = self.tokenizer.encode(
                query_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=2048
            ).to(self.device)
            
            # 3. Get value from critic
            with torch.no_grad():
                # For value, use query without generation prompt
                value_text = self.tokenizer.apply_chat_template(
                    [{"role": p[0], "content": p[1]} for p in full_prompt_msgs],
                    tokenize=False,
                    add_generation_prompt=False
                )
                value_tensor = self.tokenizer.encode(
                    value_text, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(self.critic.device)
                
                value = self.critic(value_tensor)[0].squeeze(-1).detach()
            
            # 4. Generate action with temperature sampling
            with torch.no_grad():
                generation_outputs = self.model.generate(
                    query_tensor,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Extract response tokens
            response_tokens = generation_outputs.sequences[0][query_tensor.shape[1]:]
            action_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            return query_tensor.squeeze(0), response_tokens, value, action_text
            
        except Exception as e:
            print(f"Error in get_action_and_value: {e}")
            # Return fallback values
            dummy_query = torch.zeros(10, dtype=torch.long, device=self.device)
            dummy_response = torch.zeros(5, dtype=torch.long, device=self.device)
            dummy_value = torch.tensor(0.0, device=self.device)
            return dummy_query, dummy_response, dummy_value, "Done"
        
    def collect_rollout_step(self, env, rollout_past, task):
        """Collect a single rollout step for PPO training."""
        try:
            query_tensor, response_tokens, value, action_text = self.get_action_and_value(env, rollout_past, task)
            
            # Create interaction data similar to standard planner
            interaction_data = {
                "action": action_text,
                "action_enum": self._parse_action_enum(action_text),
                "step": len(rollout_past),
                "query_tensor": query_tensor,
                "response_tokens": response_tokens,
                "value": value
            }
            
            return query_tensor, response_tokens, value, interaction_data
            
        except Exception as e:
            print(f"Error in collect_rollout_step: {e}")
            # Return dummy values for failed steps
            dummy_query = torch.zeros(10, dtype=torch.long, device=self.device)
            dummy_response = torch.zeros(5, dtype=torch.long, device=self.device)
            dummy_value = torch.tensor(0.0, device=self.device)
            dummy_interaction = {
                "action": "Done",
                "action_enum": "done",
                "step": len(rollout_past),
                "query_tensor": dummy_query,
                "response_tokens": dummy_response,
                "value": dummy_value
            }
            return dummy_query, dummy_response, dummy_value, dummy_interaction
        
    def _parse_action_enum(self, action_text):
        """Parse action text to determine action type."""
        action_lower = action_text.lower().strip()
        if action_lower in ["done", "finish", "complete"]:
            return "done"
        elif action_lower.startswith("ask") or "?" in action_text:
            return "ask"
        else:
            return "act"