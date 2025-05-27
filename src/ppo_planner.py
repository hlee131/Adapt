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
        """
        Generate action using PPO policy and get value from critic.
        Maintains consistency with evaluation pipeline by using existing methods.
        """
        # 1. Construct prompt using existing evaluation pipeline logic
        prompt_msgs, _ = prompt_from_rollout(
            rollout_past,
            assistant="robot",
            skip=[],
            change_user_to=self.persona_id,
            skip_failed=True,
            action_only=True  # Consistent with evaluation
        )
        
        # Add example history (same as in LLMAgent_Planner.__call__)
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
        
        # Construct spoonfeeding summary (same as evaluation)
        spoonfeeding_summary = ''
        if len(self.user_info) > 0 and self.user_info_with_summary:
            spoonfeeding_summary += f"Remember, {self.persona_id}'s preferences include: " + self.user_info + "."
        spoonfeeding_summary += f'What is the next step to complete the task: {task}?'
        
        # Create system prompt and full message list
        system_prompt_msg = ("system", planner_system_prompt(
            self.persona_id, self.user_info, env, task, 
            no_ask_option=self.no_ask_option, action_only=True
        ))
        user_query_msg = ("user", spoonfeeding_summary)
        
        full_prompt_msgs = [system_prompt_msg] + prompt_msgs + [user_query_msg]
        
        # 2. Prepare tensors for PPO
        # Query for generation (actor needs add_generation_prompt=True)
        query_text = self.tokenizer.apply_chat_template(
            [{"role": p[0], "content": p[1]} for p in full_prompt_msgs],
            tokenize=False,
            add_generation_prompt=True
        )
        query_tensor = self.tokenizer.encode(query_text, return_tensors="pt").to(self.device)
        
        # Query for value (critic typically doesn't need generation prompt)
        value_query_text = self.tokenizer.apply_chat_template(
            [{"role": p[0], "content": p[1]} for p in full_prompt_msgs],
            tokenize=False,
            add_generation_prompt=False
        )
        value_query_tensor = self.tokenizer.encode(value_query_text, return_tensors="pt").to(self.critic.device)
        
        # 3. Get value from critic
        with torch.no_grad():
            value = self.critic(value_query_tensor)[0].squeeze().detach()
        
        # 4. Generate action using existing grammar-constrained generation
        action_grammar_str = create_grammar(env, no_ask_option=self.no_ask_option)
        
        # Use the existing run_llm method for consistency with evaluation
        llm_outputs = self.run_llm(
            prompt_msgs=full_prompt_msgs,
            temperature=self.temperature,
            constrained_gen_grammar=action_grammar_str,
            add_generation_prompt=True
        )
        action_text = llm_outputs[0]["generation"].strip()
        
        # 5. Tokenize the generated action for PPO
        response_tokens = self.tokenizer.encode(
            action_text, return_tensors="pt", add_special_tokens=False
        ).to(self.device).squeeze(0)
        
        return query_tensor.squeeze(0), response_tokens, value, action_text

    def collect_rollout_step(self, env, rollout_past, task):
        """
        Collect one step of rollout data for PPO training.
        Returns all necessary components for PPO update.
        """
        # Get action and value
        query_tensor, response_tokens, value, action_text = self.get_action_and_value(
            env, rollout_past, task
        )
        
        # Step environment
        success, observation_msg, action_enum, action_args = env.step(action_text)
        
        # Create interaction data (consistent with evaluation format)
        interaction_data = {
            "thought": "",  # PPO typically doesn't use explicit thoughts
            "action": action_text,
            "success": success,
            "observation": observation_msg,
            "action_enum": action_enum,
            "action_args": action_args,
        }
        
        return query_tensor, response_tokens, value, interaction_data