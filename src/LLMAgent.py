import os
import sys
from copy import deepcopy
import numpy as np

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessorList

from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint


from src.convert_and_evaluate import process_trace
from src.utils import (
    DEVICE,
    _tokenizer,
    create_grammar,
    planner_system_prompt,
    prompt_from_rollout,
    privileged_task_descriptions,
    NO_CONSTRAINED_GENERATION,
)

from env.reward_models.persona_rewards.get_preference_list import get_preference_list
          
def summarize_standalone(run_llm, prompt, temperature):
    llm_output = run_llm(prompt_msgs=prompt, max_tokens=512, temperature=temperature, stops=["Action:", "Thought:", "Observation:", "User:"])[0]
    response, probs = llm_output["generation"], llm_output["mean_prob"]
    response = response.strip()
    return response


def summarize_standalone_n_samples(run_llm, prompt, temperature, return_n=1):
    for llm_output in run_llm(prompt_msgs=prompt, max_tokens=1024, return_n=return_n, temperature=temperature, stops=["Action:", "Thought:", "Observation:", "User:"]):
        response, probs = llm_output["generation"], llm_output["mean_prob"]
        response = response.strip()
        yield response


LOCAL_MODELS = {}

class LLMAgent():
    def __init__(self, llm_runname=None):
        path = llm_runname
        if path is None:
            path = "meta-llama/Llama-3.1-8B-Instruct"
        self.model_in_path = path
        if path in LOCAL_MODELS:
            self.model = LOCAL_MODELS[path]['model']
            self.tokenizer = LOCAL_MODELS[path]['tokenizer']
        else:    
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_in_path)
            self.model_cache_dir = (
                f"models/cache"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_in_path,
                device_map="auto",
                cache_dir=self.model_cache_dir,
                torch_dtype=torch.bfloat16,
            )
            LOCAL_MODELS[path] = {'model':self.model, 'tokenizer':self.tokenizer}
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token is None or self.tokenizer.pad_token == "":
            self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(0)

        self.default_llm_params = {
            "max_new_tokens": 250,
            "temperature": 1.0,
            "sampling": False
        }
        self.device = DEVICE
        self.name_or_path = self.model_in_path
        
    def get_grammar_processor(self, grammar_string='root ::= "temp"'):
        grammar = IncrementalGrammarConstraint(
                grammar_string,
                "root",
                self.tokenizer,
            )
        grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
        return grammar_processor

    
    def batch_generate(self, prompts, max_new_tokens, temperature, generation_args):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)

        force_output = False
        if generation_args is not None and 'force_action' in generation_args:
            force_output = True

            generation_args['force_action'] = ['the '+x for x in generation_args['force_action']]
            # Generation being tested for
            if all(isinstance(x, str) for x in generation_args['force_action']):
                self.tokenizer.padding_side = "right"
                generated_tokens_forced = self.tokenizer(generation_args['force_action'], return_tensors="pt", padding=True).input_ids
            else:
                generated_tokens_forced = torch.stack(generation_args['force_action'], dim=0)
            generated_tokens_forced = generated_tokens_forced[:,2:]

            # Inputs for the full text
            all_ids = torch.concatenate([inputs.input_ids, generated_tokens_forced], dim=-1)
            all_masks = torch.concatenate([inputs.attention_mask, torch.ones_like(generated_tokens_forced)], dim=-1)
            
            # Forward pass to get logits
            with torch.no_grad():
                try:
                    outputs_forced = self.model.forward(all_ids.to(self.device),attention_mask=all_masks.to(self.device))
                except:
                    outputs_forced = self.model(all_ids.to(self.device),attention_mask=all_masks.to(self.device))
            logits = outputs_forced.logits

            # Logits for the generated part (left shift by 1 because output is shifted by 1)
            generation_logits = logits[:, inputs.input_ids.shape[-1]-1:-1, :]
            output_scores = []
            for i in range(generation_logits.shape[1]):
                output_scores.append(generation_logits[:,i,:]) 

            generated_tokens_batch = generated_tokens_forced

        if not force_output:
            logits_processor_list = LogitsProcessorList()
            if generation_args is not None and "grammar_definition" in generation_args:
                logits_processor_list.append(self.get_grammar_processor(generation_args["grammar_definition"]))
            output = self.model.generate(
                inputs.input_ids.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
                logits_processor = logits_processor_list
            )
            assert len(inputs.input_ids.shape) == 2
            output_scores = output.scores
            n_prompt_tokens = inputs.input_ids.shape[1]
            generated_tokens_batch = output.sequences[:, n_prompt_tokens:]
        
        self.tokenizer.padding_side = "left"
        text = self.tokenizer.batch_decode(
            generated_tokens_batch,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        batch_out = []
        for batch_i in range(inputs.input_ids.shape[0]):
            prompt = prompts[batch_i]
            generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens_batch[batch_i])
            # There are only scores for predicted tokens
            assert len(generated_tokens_batch[batch_i]) == len(generated_tokens) == len(output_scores), "Length of generated tokens and output scores do not match: {} vs {} vs {}".format(len(generated_tokens_batch[batch_i]), len(output_scores), len(generated_tokens))
            generated_tokens = [x for x in generated_tokens if x != self.tokenizer.eos_token]

            mean_prob = []
            for i in range(len(generated_tokens)):
                if self.tokenizer.convert_tokens_to_string([generated_tokens[i]]) == "\n":
                    if i == 0:
                        continue
                    else:
                        break
                else:
                    prob = torch.softmax(output_scores[i][batch_i], 0)[generated_tokens_batch[batch_i][i]]
                    mean_prob.append(prob.cpu())

            mean_prob = float(np.mean(mean_prob)) 
            if np.isnan(mean_prob):
                mean_prob = -1.
            batch_out.append(
                {
                    "generation": text[batch_i].strip(),
                    "prompt": prompt,
                    "mean_prob": mean_prob,
                }
            )

        return batch_out
        
    
    def batch_generate_fn(self, prompts, stops=[], **llm_params):
        
        responses = self.batch_generate(
            prompts=prompts,
            max_new_tokens=llm_params["max_new_tokens"],
            temperature=llm_params["temperature"],
            generation_args=llm_params.get("generation_args", {}),
        )

        def cut_at_stop(text):
            for stop in stops+['<|']:
                if stop in text:
                    text = text.split(stop)[0]
            return text

        responses = [
            {
                "prompt": p,
                "generation": cut_at_stop(r["generation"]),
                "mean_prob": r["mean_prob"],
            }
            for p,r in zip(prompts,responses)
        ]

        return responses

    def get_probs(
        self,
        full_prompts,
        generations,
    ):
        outputs = self.batch_generate_fn(prompts=full_prompts, max_new_tokens=200, temperature=1.0, generation_args={"force_action":generations})
        return [o['mean_prob'] for o in outputs]
    
    def run_llm(
        self,
        prompt_msgs,
        stops=[],
        max_tokens=None,
        temperature=None,
        return_n=1,
        constrained_gen_grammar=None,
        add_generation_prompt=True,
    ):
        if NO_CONSTRAINED_GENERATION:
            constrained_gen_grammar = None
        llm_params = self.default_llm_params.copy()
        if max_tokens is not None:
            llm_params["max_new_tokens"] = max_tokens
        if temperature is not None:
            llm_params["temperature"] = temperature
            if temperature > 0.001:
                llm_params["sampling"] = True
        full_prompt = _tokenizer.apply_chat_template(
            [{"role": p[0], "content": p[1]} for p in prompt_msgs],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if not add_generation_prompt:
            if full_prompt.endswith('<|eot_id|>'):
                full_prompt = full_prompt[:-len('<|eot_id|>')]
            else:
                print(f"Full prompt without generation prompt does not end with <|eot_id|> {full_prompt}")
        responses = []
        full_prompts = [full_prompt for _ in range(return_n)]
        
        if constrained_gen_grammar is not None:
            grammar_str = constrained_gen_grammar
            llm_params["generation_args"] = {
                "grammar_definition": grammar_str
            }

        responses = self.batch_generate_fn(prompts=full_prompts, stops=stops, **llm_params)
       
        assert len(responses) > 0, "How come the LLM didn't output anything?!"
        return responses

class LLMAgent_Planner(LLMAgent):
    def __init__(self, persona_id, model_name_planner, temperature_planner, user_info_with_summary, no_ask_option=False, **kwargs):
        super().__init__(llm_runname = model_name_planner)
        self.agent_name = "Planner"
        self.persona_id = persona_id
        self.user_info = ""
        self.example_history = []
        self.max_actions = 4
        self.max_summaries = 4
        self.probability_thresh = 0
        self.temperature = temperature_planner
        self.no_ask_option = no_ask_option
        self.user_info_with_summary = user_info_with_summary

    def reset(self):
        self.user_info = ""
        self.example_history = []
        print("Resetting Planner...")
        
    def reflection(self, action_pred, thought_pred, action_expected, thought_expected, env, rollout_past, task, action_only=False):
        prompt_msgs, _ = prompt_from_rollout(
            rollout_past + [{"thought":thought_pred, "action":action_pred, "success":True}],
            assistant="none",
            skip=[],
            change_user_to=self.persona_id,
            skip_failed=True,
            action_only=action_only,
        )
        system_prompt_reflection = f"You are an expert at task planning, and can guide a robot on how to provide assistance in a manner that {self.persona_id} wants to {task}. \nReflect on this difference between the action the robot should have predicted, and actually predicted. Was there some knowledge about {self.persona_id}'s preferences, which if the robot knew about, it would have predicted the expected action?\nExample: The robot cannot predict that it should use almonds when making chia pudding, when it doesn't know that {self.persona_id} wants their chia pudding topped with almonds. If so, it could ask {self.persona_id} a question to clarify their preference first, such as 'Question: What toppings do you want on your sweet breakfasts, like chia pudding?' If no question needs to be asked to predict the expected action, you can say 'Question: None'. Answer with a single question and do not provide any additional information or explanation."
        
        all_prompts = [
                (
                    "system",
                    system_prompt_reflection,
                )
            ] \
            + prompt_msgs + \
            [
                (
                    "user", 
                    f"Instead of {action_pred}, the robot was expected to perform {action_expected}" +
                    (f", because if the robot knew {self.persona_id} better, it would have thought that '{thought_expected}'"  if thought_expected.strip()!="" else "")+
                    f".\nWhat question could the robot have asked {self.persona_id} so that it could predict {action_expected} as its next step?"
                ),
            ]

        llm_outputs_list = self.run_llm(
            all_prompts,
            temperature=self.temperature,
            constrained_gen_grammar='root ::= " Question: " ([ ()_.,-?!/a-zA-Z_0-9])+',
            stops = ["?","Action:"]
        )
        question = llm_outputs_list[0]
        question, _ = question["generation"], question["mean_prob"]
        question = question.replace("Question:","")+'?"'
        return question.strip()
    
    def reflection_action_as_student(self, env, action_priv, thought_priv, initial_action, initial_thought, rollout_past, ask_interaction, task, action_only=False, no_history=False, progress_summ=None, privileged_info_at_step=None):
        system_prompt_msg = ("system",planner_system_prompt(self.persona_id, self.user_info, env, task, no_ask_option=self.no_ask_option, action_only=action_only))
        
        prompt_msgs_wo_ask, _ = prompt_from_rollout(
            rollout_past,
            assistant="robot",
            skip=[],
            change_user_to=self.persona_id,
            skip_failed=True,
            action_only=action_only,
        )
        
        prompt_msgs_with_ask, _ = prompt_from_rollout(
            rollout_past+[ask_interaction],
            assistant="robot",
            skip=[],
            change_user_to=self.persona_id,
            skip_failed=True,
            action_only=action_only,
        )
                
        spoonfeeding_summary = ''
        if privileged_info_at_step is not None:
            spoonfeeding_summary += privileged_info_at_step + '\n'
        elif len(self.user_info) > 0 and self.user_info_with_summary:
            spoonfeeding_summary += f"Remember, {self.persona_id}'s preferences include: " + self.user_info +"."
        if progress_summ is not None:
            spoonfeeding_summary += progress_summ + '\n'
        spoonfeeding_summary += f'What is the next step to complete the task: {task}?'
        
        prompt_msgs_wo_ask = [system_prompt_msg] + prompt_msgs_wo_ask + [("user", spoonfeeding_summary)]
        prompt_msgs_with_ask = [system_prompt_msg] + prompt_msgs_with_ask + [("user", spoonfeeding_summary)]
        if no_history:
            prompt_msgs_wo_ask = [system_prompt_msg] + [("user", spoonfeeding_summary)]
            prompt_msgs_with_ask = [system_prompt_msg] + [("user", spoonfeeding_summary)]
        
        prompt_msgs_wo_ask = _tokenizer.apply_chat_template(
                [{"role": p[0], "content": p[1]} for p in prompt_msgs_wo_ask],
                tokenize=False,
                add_generation_prompt=True,
            )
        prompt_msgs_with_ask = _tokenizer.apply_chat_template(
                [{"role": p[0], "content": p[1]} for p in prompt_msgs_with_ask],
                tokenize=False,
                add_generation_prompt=True,
            )
        
        if action_only:
            prompt_msgs_wo_ask += "Action:"
            prompt_msgs_with_ask += "Action:"
            teacher_completion = f"{action_priv}"
            student_completion = f"{initial_action}"
            question_completion = f"{ask_interaction['action']}"
        else:
            prompt_msgs_wo_ask += "Thought:"
            prompt_msgs_with_ask += "Thought:"
            teacher_completion = f"{thought_priv}\nAction: {action_priv}"
            student_completion = f"{initial_thought}\nAction: {initial_action}"
            question_completion = f"{ask_interaction['thought']}\nAction: {ask_interaction['action']}"
        
        prompts = [prompt_msgs_wo_ask, prompt_msgs_with_ask, prompt_msgs_with_ask, prompt_msgs_wo_ask, prompt_msgs_wo_ask]
        completions = [teacher_completion, student_completion, teacher_completion, question_completion, student_completion]
        probs = self.get_probs(prompts, completions)
        
        probabilities = {'p_s_a_t':probs[0], 
                         'p_s_a_s_ques':probs[1], 
                         'p_s_a_t_ques':probs[2],
                         'p_s_ques': probs[3],
                         'p_s_a_s': probs[4]}
        
        return probabilities, prompts, completions
        
        
    def reflection_action_as_teacher(self, env, action_priv, thought_priv, initial_action, initial_thought, rollout_past, ask_interaction, task, progress_summ=None, privileged_info_at_step=None):
        system_prompt_msg = ("system",planner_system_prompt(self.persona_id, self.user_info, env, task, no_ask_option=self.no_ask_option, action_only=False))
        
        prompt_msgs_wo_ask, _ = prompt_from_rollout(
            rollout_past,
            assistant="robot",
            skip=[],
            change_user_to=self.persona_id,
            skip_failed=True,
        )
        prompt_msgs_wo_ask = [system_prompt_msg] + prompt_msgs_wo_ask
        
        prompt_msgs_with_ask, _ = prompt_from_rollout(
            rollout_past+[ask_interaction],
            assistant="robot",
            skip=[],
            change_user_to=self.persona_id,
            skip_failed=True,
        )
        prompt_msgs_with_ask = [system_prompt_msg] + prompt_msgs_with_ask
        
        spoonfeeding_summary = ''
        if privileged_info_at_step is not None:
            spoonfeeding_summary += privileged_info_at_step + '\n'
        elif len(self.user_info) > 0 and self.user_info_with_summary:
            spoonfeeding_summary += f"Remember, {self.persona_id}'s preferences include: " + self.user_info +"."
        if progress_summ is not None:
            spoonfeeding_summary += progress_summ + '\n'
        spoonfeeding_summary += f'What is the next step to complete the task: {task}?'
        
        teacher_completion = f"Thought: {thought_priv}\nAction: {action_priv}"
        student_completion = f"Thought: {initial_thought}\nAction: {initial_action}"
        
        prompts = [prompt_msgs_wo_ask, prompt_msgs_with_ask, prompt_msgs_with_ask]
        completions = [student_completion, student_completion, teacher_completion]
        probs = self.get_probs(prompts, completions)
        
        probabilities = {'p_t_a_s':probs[0],
                         'p_t_a_s_ques':probs[1],
                         'p_t_a_t_ques':probs[2]}
    
        return probabilities

    def __call__(self, env, rollout_past, task, progress_summ=None, privileged_info_at_step=None, summarize_todos=False, no_generate=False, skip_failed=False, action_only=False, no_history=False, unconstrained_planner=False, force_ask=False):
      
        prompt_msgs, _ = prompt_from_rollout(
            rollout_past,
            assistant="robot",
            skip=[],
            change_user_to=self.persona_id,
            skip_failed=skip_failed,
            action_only=action_only,
        )

        for i_ex, (example_task, example_rollout) in enumerate(self.example_history):
            prompt_msgs_ex, _ = prompt_from_rollout(
                example_rollout,
                assistant="robot",
                skip=[],
                change_user_to=self.persona_id,
                skip_failed=skip_failed,
                action_only=action_only,
            )
            prompt_msgs = (
                [("user", f"Example {i_ex}, Task {example_task}:")]
                + prompt_msgs_ex
                + prompt_msgs
            )
            
        if no_history:
            prompt_msgs = []
            
        spoonfeeding_summary = ''
        if privileged_info_at_step is not None:
            spoonfeeding_summary += privileged_info_at_step + '\n'
        elif len(self.user_info) > 0 and self.user_info_with_summary:
            spoonfeeding_summary += f"Remember, {self.persona_id}'s preferences include: " + self.user_info +"."
        if progress_summ is not None:
            spoonfeeding_summary += progress_summ + '\n'
        spoonfeeding_summary += f'What is the next step to complete the task: {task}?'
        prompt_msgs.append(("user", spoonfeeding_summary))

        system_prompt_msg = ("system",planner_system_prompt(self.persona_id, self.user_info, env, task, no_ask_option=self.no_ask_option, action_only=action_only))


        prompt = ''
        action_grammar_str = create_grammar(env, no_ask_option=self.no_ask_option)
        if unconstrained_planner:
            action_grammar_str = None
        
        if force_ask:
            action_grammar_str = open('src/planner_grammar_question_only.ebnf').read()
        
        if no_generate:
            return None , None, action_grammar_str, [system_prompt_msg] + prompt_msgs, None
        
        prob = 0
        if not action_only:
            ## Predict Thought
            llm_outputs_list = self.run_llm(
                [system_prompt_msg] + prompt_msgs + [("assistant", "Thought: ")],
                temperature=self.temperature,
                add_generation_prompt=False,
                stops=["Action:", "Observation:", "User:"],
                )
            thought = llm_outputs_list[0]["generation"]
            prompt = llm_outputs_list[0]["prompt"]
            prob += llm_outputs_list[0]['mean_prob']

            ## Predict Action
            llm_outputs_list = self.run_llm(
                [system_prompt_msg] + prompt_msgs + [("assistant", "Thought: "+thought+"\nAction: ")],
                temperature=self.temperature,
                constrained_gen_grammar=action_grammar_str,
                add_generation_prompt=False,
                stops=["Thought:", "Observation:", "User:", ".", "\n", "|", "/", "<"],
                )
            action = llm_outputs_list[0]["generation"]
            prob += llm_outputs_list[0]['mean_prob']
            prob /= 2
        else:
            thought = ''
            ## Predict Action
            llm_outputs_list = self.run_llm(
                [system_prompt_msg] + prompt_msgs + [("assistant", "Action: ")],
                temperature=self.temperature,
                constrained_gen_grammar=action_grammar_str,
                add_generation_prompt=False,
                stops=["Thought:", "Observation:", "User:", ".", "\n", "|", "/", "<"],
                )
            action = llm_outputs_list[0]["generation"]
            prompt = llm_outputs_list[0]["prompt"]
            prob = llm_outputs_list[0]['mean_prob']
        
        if action.lower().startswith("look for"):
            action = action.split('_')[0]

        return action.strip(), thought.strip(), action_grammar_str, prompt, prob

    def add_user_info(self, info):
        if info is None: return
        self.user_info = info

    def push_example(self, example_task, example_rollout):
        self.example_history.append((example_task, example_rollout))


class LLMAgent_Persona(LLMAgent):
    def __init__(self, persona_id, model_name_base, skip_persona_syntax_failures, temperature_persona, **kwargs):
        super().__init__(llm_runname = model_name_base)
        self.agent_name = persona_id
        self.system_prompt = (
            f"You are teaching a household assistive robot in performing various assistive tasks in a manner {self.agent_name} would like. The robot may not know {self.agent_name}'s preferences, so your job is to guide the robot to perform the given task for {self.agent_name}. Be sure to guide the robot to make only those dishes that the task calls for, e.g. if the task is to make a waffle do not ask the robot to make other things, such as coffee. Answer direct questions regarding your preferences, and not the avilability or location of objects. In the latter case, encourage the robot to search and explore different locations. Even if the robot makes an irreversible error, be sure to provide a correction so that the robot does not repeat it's mistakes the next time."
            + f"\n\nGiven the current state of the house and what you know about {self.agent_name} and the task at hand, you will respond to the robot's last question concisely, and in first person, as if you are {self.agent_name}."
        )
        self.task = None
        self.preferences_list = get_preference_list(persona_id=persona_id)
        self.privileged_preferences = '\n'.join(self.preferences_list)
        self.skip_persona_syntax_failures = skip_persona_syntax_failures
        self.temperature = temperature_persona

    def reset(self):
        self.task = None
        self.preferences_list = None
        print("Resetting Persona...")

    def get_privileged_summary(self):
        summary = ""
        summary += (
            f"You know the following about {self.agent_name}'s preferences.\n"
            + self.privileged_preferences
        )
        return summary

    def answer_question(self, env, rollout_steps, task):
        inst2 = f"Look at the following interaction and provide a short answer to the robot's last question based on {self.agent_name}'s preferences. If {self.agent_name} is flexible in their preference, make a choice arbirarily, but make sure to tell the robot that usually {self.agent_name} is flexible, and options which they would be okay with. Make sure to be consistent with your previous feedback."
        
        system_prompt = (
            self.system_prompt
            + "\n\n"
            + f"Environment State:\n{env.prompt_string()}"
            + f"\n\nTask: {self.task}"
            + f"\n {self.agent_name} has the following preferences.\n{self.preferences_list}"
            + f"\n\n{inst2}"
        )

        prompt_msgs = [("system", system_prompt)]
        prompt_msgs_add, _ = prompt_from_rollout(
            rollout_steps,
            assistant="user",
            skip=["memory"],
            skip_failed=self.skip_persona_syntax_failures,
        )
        prompt_msgs += prompt_msgs_add

        llm_output = self.run_llm(prompt_msgs=prompt_msgs, temperature = self.temperature, stops=["Action:", "Thought:", "Observation:", "User:", "\n"])
        response, probs = llm_output[0]["generation"], llm_output[0]["mean_prob"]
        response_subjective = response
        response_objective = 0
        if "OK" in response:
            maybe_ok = (
                response.replace("Correction:", "")
                .replace("Feedback:", "")
                .replace(".", "")
                .strip()
            )
            if maybe_ok.startswith("OK"):
                response_objective = 1
                response_subjective = "OK"
        else:
            response_subjective = response
        return response_subjective.strip(), response_objective

    def __call__(self, env, rollout_data, current_interaction, ask_or_correct, task):
        self.task = task
        rollout_steps = deepcopy(rollout_data["rollout"])
        rollout_steps += [current_interaction]
        if ask_or_correct == "ask":
            return self.answer_question(env, rollout_steps, task)
        elif ask_or_correct == "confirm_done":
            rollout_data_copy = deepcopy(rollout_data)
            rollout_data_copy["rollout"] = rollout_steps
            results_dict = process_trace(rollout_data_copy)
            return results_dict

        
