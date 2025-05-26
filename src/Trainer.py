import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)

from trl import DPOConfig, DPOTrainer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from src.Environment import Environment
from src.utils import (
    _tokenizer,
    now_timestr,
    planner_system_prompt,
    prompt_from_rollout,
    save_hf_checkpoint
)

WANDB_PROJECT = "PolicyPersonalization"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT

MAX_TOKENS = 4096


def extract_data_for_summary(rollout_data, tokenizer, summary_index = None, separate_summaries=False):
    summaries = rollout_data["summary"]
    summary = sorted(summaries, key=lambda x: len(x))[-1]
    rollout = [{"role": p[0], "content": p[1]} for p in rollout_data['prompt']]
    if len(rollout) > 50:
        rollout = rollout[:49] + [rollout[-1]]
    prompt = _tokenizer.apply_chat_template(
        rollout, tokenize=False, add_generation_prompt=True
    )
    token_num = len(tokenizer(prompt + summary)["input_ids"])
    while token_num > MAX_TOKENS and len(rollout) > 2:
        rollout = rollout[:-2] + [rollout[-1]]
        prompt = _tokenizer.apply_chat_template(
            rollout, tokenize=False, add_generation_prompt=True
        )
        token_num = len(tokenizer(prompt + summary)["input_ids"])
    if summary_index is not None:
        if separate_summaries:
            return prompt, summaries[summary_index]
        return prompt + summaries[summary_index]
    else:
        if separate_summaries:
            return prompt, summaries
        return [prompt + summary for summary in summaries]

def extract_data_for_planner(rollout_data, tokenizer, custom_prior_info=None):
    prompt_msgs, _ = prompt_from_rollout(
        rollout_data["rollout"],
        assistant="robot",
        skip=[],
        change_user_to=rollout_data["persona_id"],
        skip_failed=True,
    )
    if custom_prior_info is not None:
        rollout_data["prior_persona_knowledge"] = custom_prior_info
    system_prompt = planner_system_prompt(
        rollout_data["persona_id"],
        rollout_data["prior_persona_knowledge"],
        Environment(json.loads(rollout_data["initial_scene"])),
        rollout_data["task"],
        action_only=True,
    )
    prompt_msgs = [{"role": "system", "content": system_prompt}] + [
        {"role": p[0], "content": p[1]} for p in prompt_msgs
    ]
    prompt = _tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    while len(tokenizer(prompt)["input_ids"]) > MAX_TOKENS and len(prompt_msgs) > 1:
        prompt_msgs = prompt_msgs[:-1]
        prompt = _tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
    return prompt

def gen_dummy():
    for i in range(10):
        yield {
            "prompt": f"{i}_prompt",
            "chosen": f"{i}_chosen",
            "rejected": f"{i}_rejected",
        }


def gen_sft_dummy(**kwargs):
    dataset_file = f"logs/SFT_dummy.jsonl"
    if not os.path.exists(dataset_file):
        for i in range(10):
            open(dataset_file, "a").write(
                json.dumps(
                    {
                        "text": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nYesterday<|eot_id|>"
                    }
                )
                + "\n"
            )
            open(dataset_file, "a").write(
                json.dumps(
                    {
                        "text": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHey Jude<|eot_id|>"
                    }
                )
                + "\n"
            )
            open(dataset_file, "a").write(
                json.dumps(
                    {
                        "text": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHere Comes the Sun<|eot_id|>"
                    }
                )
                + "\n"
            )
            open(dataset_file, "a").write(
                json.dumps(
                    {
                        "text": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nNorwegian Wood<|eot_id|>"
                    }
                )
                + "\n"
            )

    return dataset_file


def gen_sft_and_dpo(
    use_summary_type=["none"],
    train_persona_task={},
    data_path="logs",
    use_existing=None,
    thresh_reward=None,
    remove_priv_prior=False
):
    prior_string = "Prior_" + "_".join(use_summary_type)
    dataset_summary_file = os.path.join(data_path, f"train_summary_SFT_{prior_string}.json")
    dataset_summary = {
        "personas" : set(),
        "goals": set(),
        "missing_first": [],
        "missing_summary": [],
        "missing_second": [],
    }

    dataset_file_planner = os.path.join(data_path, f"train_set_SFT_{prior_string}_planner.jsonl")
    dataset_file_summary = os.path.join(data_path, f"train_set_SFT_{prior_string}_summary.jsonl")
    dataset_file_summary_best = os.path.join(data_path, f"train_set_SFT_{prior_string}_summary_best.jsonl")
    dataset_file_summary_dpo = os.path.join(data_path, f"train_set_SFT_{prior_string}_summary_dpo.jsonl")
    dataset_file_planner_best = os.path.join(data_path, f"train_set_SFT_{prior_string}_planner_best.jsonl")
    dataset_file_planner_dpo = os.path.join(data_path, f"train_set_SFT_{prior_string}_planner_dpo.jsonl")
    if thresh_reward is not None:
        dataset_file_planner = dataset_file_planner.replace(".jsonl", f"_thresh{thresh_reward}.jsonl")
        dataset_file_summary = dataset_file_summary.replace(".jsonl", f"_thresh{thresh_reward}.jsonl")
        dataset_file_summary_best = dataset_file_summary_best.replace(".jsonl", f"_thresh{thresh_reward}.jsonl")
        dataset_file_summary_dpo = dataset_file_summary_dpo.replace(".jsonl", f"_thresh{thresh_reward}.jsonl")
        dataset_file_planner_best = dataset_file_planner_best.replace(".jsonl", f"_thresh{thresh_reward}.jsonl")
        dataset_file_planner_dpo = os.path.join(data_path, f"train_set_SFT_{prior_string}_planner_dpo.jsonl")

    dataset_filedict = {"planner": dataset_file_planner, 
                        "summary": dataset_file_summary,
                        "summary_best": dataset_file_summary_best,
                        "summary_dpo": dataset_file_summary_dpo,
                        "planner_best": dataset_file_planner_best,
                        "planner_dpo": dataset_file_planner_dpo}

    if all([os.path.exists(dataset_file) for dataset_file in dataset_filedict.values()]):
        use_existing = use_existing if use_existing is not None else (
            input(
                f"{dataset_file} already exists. Would you like to use it? (y/n)"
            ).lower()
            == "y"
        )
        if use_existing:
            return dataset_filedict

    for dataset_file in dataset_filedict.values():
        if os.path.exists(dataset_file):
            os.remove(dataset_file)

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct"
    )

    def use_file(fn):
        fn = os.path.basename(fn)
        valid_persona = any(
            [persona in fn for persona in train_persona_task["personas"]]
        )
        valid_task = any(
            [task.replace(" ", "_") in fn for task in train_persona_task["goals"]]
        )
        return valid_persona and valid_task

    def use_rollout(rollout_data):
        if thresh_reward is None:
            return rollout_data['rollout'][-1]['action_enum'] == "done"
        else:
            return rollout_data['rollout'][-1]['action_enum'] == "done" and rollout_data['total_reward'] >= thresh_reward


    print("Creating dataset file for training...")
    bad_interactions = []
    interactions_included = []

    ## First interaction
    max_rollouts = 0
    expected_rewards_planner = {}
    open(dataset_file_planner, "w").write("")
    for filename in os.listdir(os.path.join(data_path, "data")):
        if not use_file(filename):
            continue
        filepath = os.path.join(data_path, "data", filename)
        rollout_data = json.load(open(filepath))
        filename = filename.replace(".json", "")
        persona = filename[:filename.index("_")]
        task = filename[filename.index("_")+1:-1].replace("_"," ")
        rollout_num = int(filename[-1])
        if rollout_num > max_rollouts:
            max_rollouts = rollout_num
        interactions_included.append((persona, task, rollout_num))
        dataset_summary["personas"].add(persona)
        dataset_summary["goals"].add(task)
        if (persona, task) not in expected_rewards_planner:
            expected_rewards_planner[(persona, task)] = {}
        expected_rewards_planner[(persona, task)][filepath] = rollout_data['total_reward']
        if use_rollout(rollout_data):
            open(dataset_file_planner, "a").write(
                json.dumps({"text": extract_data_for_planner(rollout_data, tokenizer, custom_prior_info='' if remove_priv_prior else None)}) + "\n"
            )
        else:
            bad_interactions.append((persona, task, rollout_num))
    for persona in dataset_summary["personas"]:
        for task in dataset_summary["goals"]:
            for rollout_num in range(max_rollouts+1):
                if (persona, task, rollout_num) not in interactions_included:
                        dataset_summary["missing_first"].append((persona, task, rollout_num))

    num_first_interactions_from_file = len(open(dataset_file_planner).readlines())
    print(f"Finished {num_first_interactions_from_file} first interactions... Written to {dataset_file_planner}")
    expected_rewards_summary = {}
    expected_rewards_planner = {}

    ## Summary
    interactions_included = []
    open(dataset_file_summary, "w").write("")
    if not os.path.exists(os.path.join(data_path, "summaries")):
        os.makedirs(os.path.join(data_path, "summaries"))
    for filename in os.listdir(os.path.join(data_path, "summaries")):
        if not use_file(filename):
            continue
        rollout_data = json.load(open(os.path.join(data_path, "summaries", filename)))
        filename = filename.replace(".json", "")
        persona = filename[:filename.index("_")]
        task = filename[filename.index("_")+1:-1].replace("_"," ")
        rollout_num = int(filename[-1])
        interactions_included.append((persona, task, rollout_num))
        if (persona, task, rollout_num) in bad_interactions:
            continue
        expected_rewards_summary[(persona, task, rollout_num)] = [[] for _ in rollout_data["summary"]]
        for summary_text in extract_data_for_summary(rollout_data, tokenizer):
            open(dataset_file_summary, "a").write(
                json.dumps({"text": summary_text}) + "\n"
            )
    for persona in dataset_summary["personas"]:
        for task in dataset_summary["goals"]:
            for rollout_num in range(max_rollouts+1):
                if (persona, task, rollout_num) not in interactions_included:
                    if (persona, task, rollout_num) not in dataset_summary["missing_first"]:
                        dataset_summary["missing_summary"].append((persona, task, rollout_num))
                else:
                    if (persona, task, rollout_num) in dataset_summary["missing_first"]:
                        raise KeyError(f"Missing first interaction for {persona} and {task} for {rollout_num}, but summary found!!")

    print(f"Finished {len(open(dataset_file_summary).readlines())} summaries... Written to {dataset_file_summary}")

    ## Second interaction
    bad_interactions = []
    interactions_included = []
    if not os.path.exists(os.path.join(data_path, "summary_data")):
        os.makedirs(os.path.join(data_path, "summary_data"))
    second_interaction_directory = os.path.join(data_path, "summary_data")
    for filename in os.listdir(second_interaction_directory):
        if not use_file(filename):
            continue
        filepath = os.path.join(second_interaction_directory, filename)
        rollout_data = json.load(open(filepath))
        persona = filename[:filename.index("_")]
        filename_cut = filename[filename.index("_")+1:]
        task_prev_and_summ = filename_cut.split("_seen_")[0].split("_unseen_")[0]
        task_prev = " ".join(task_prev_and_summ.split("_")[:-2])
        rollout_num_prev = int(task_prev_and_summ.split("_")[-2])
        summary_num = int(task_prev_and_summ.split("_")[-1])
        filename_cut = filename_cut.split("_seen_")[-1].split("_unseen_")[-1].replace(".json","")
        task = " ".join(filename_cut.split("_")[:-1])
        rollout_num = " ".join(filename_cut.split("_")[-1])
        interactions_included.append((persona, task_prev, rollout_num_prev, task, rollout_num))
        if (persona,task_prev,rollout_num_prev) not in expected_rewards_summary: continue
        expected_rewards_summary[(persona, task_prev, rollout_num_prev)][summary_num].append(rollout_data['total_reward'])
        if (persona, task_prev,rollout_num_prev) not in expected_rewards_planner:
            expected_rewards_planner[(persona, task_prev, rollout_num_prev)] = {}
        expected_rewards_planner[(persona, task_prev, rollout_num_prev)][filepath] = rollout_data['total_reward']
        if use_rollout(rollout_data):
            open(dataset_file_planner, "a").write(
                json.dumps({"text": extract_data_for_planner(rollout_data, tokenizer)}) + "\n"
            )
    for persona in dataset_summary["personas"]:
        for task_prev in dataset_summary["goals"]:
            for rollout_prev in range(max_rollouts+1):
                for task in dataset_summary["goals"]:
                    for rollout_num in range(max_rollouts+1):
                        if (persona, task_prev, rollout_prev, task, rollout_num) not in interactions_included:
                            if (persona, task_prev, rollout_prev) not in dataset_summary["missing_first"] and (persona, task_prev, rollout_prev) not in dataset_summary["missing_summary"]:
                                dataset_summary["missing_second"].append((persona, task_prev, rollout_prev, task, rollout_num))
                        else:
                            if (persona, task_prev, rollout_prev) in dataset_summary["missing_first"] or (persona, task_prev, rollout_prev) in dataset_summary["missing_summary"]:
                                raise KeyError(f"Missing first interaction or summary for {persona} and {task} for rollout {rollout_prev}, but second interaction found!!")
    print(f"Finished {len(open(dataset_file_planner).readlines()) - num_first_interactions_from_file} second interactions... Written to {dataset_file_planner}")

    expected_rewards_summary = {k: [np.mean(vv) for vv in v] for k, v in expected_rewards_summary.items()}

    ## Summary Top-K and DPO
    open(dataset_file_summary_dpo,'w').write("")
    open(dataset_file_summary_best,'w').write("")
    for filename in os.listdir(os.path.join(data_path, "summaries")):
        if not use_file(filename):
            continue
        data = json.load(open(os.path.join(data_path, "summaries", filename)))
        rollout_data = data
        filename = filename.replace(".json", "")
        persona = filename[:filename.index("_")]
        task = filename[filename.index("_")+1:-1].replace("_"," ")
        rollout_num = int(filename[-1])
        if (persona, task, rollout_num) not in expected_rewards_summary: continue
        prompt, summaries = extract_data_for_summary(rollout_data, tokenizer, separate_summaries=True)
        for reward1, summary1 in zip(expected_rewards_summary[(persona, task, rollout_num)], summaries):
            for reward2, summary2 in zip(expected_rewards_summary[(persona, task, rollout_num)], summaries):
                if reward1 > reward2:
                    open(dataset_file_summary_dpo, "a").write(
                        json.dumps({"prompt": prompt, "chosen": summary1, "rejected": summary2}) + "\n"
                    )
                elif reward2 > reward1:
                    open(dataset_file_summary_dpo, "a").write(
                        json.dumps({"prompt": prompt, "chosen": summary2, "rejected": summary1}) + "\n"
                    )
        best_summary_index = np.argmax(expected_rewards_summary[(persona, task, rollout_num)])
        summary_text = summaries[best_summary_index]
        open(dataset_file_summary_best, "a").write(
            json.dumps({"text": summary_text}) + "\n"
        )
    print(f"Finished {len(open(dataset_file_summary_dpo).readlines())} DPO summary data... Written to {dataset_file_summary_dpo}")
    print(f"Finished {len(open(dataset_file_summary_best).readlines())} topK summary data... Written to {dataset_file_summary_best}")

    ## Planner Top-K and DPO
    open(dataset_file_planner_dpo,'w').write("")
    open(dataset_file_planner_best, "w").write("")
    for key, reward_by_rollout_filepath in expected_rewards_planner.items():
        best_file = max(reward_by_rollout_filepath, key = reward_by_rollout_filepath.get)
        rollout_data = json.load(open(best_file))
        open(dataset_file_planner_best, "a").write(
                json.dumps({"text": extract_data_for_planner(rollout_data, tokenizer)}) + "\n"
            )
        all_rollout_data = {k: extract_data_for_planner(json.load(open(k)), tokenizer) for k in reward_by_rollout_filepath.keys()}
        for filename1, reward1 in reward_by_rollout_filepath.items():
            for filename2, reward2 in reward_by_rollout_filepath.items():
                if reward1 > reward2:
                    open(dataset_file_planner_dpo, "a").write(
                        json.dumps({"prompt": all_rollout_data[filename1], "chosen": all_rollout_data[filename1],"rejected": all_rollout_data[filename2]}) + "\n"
                    )
                elif reward2 > reward1:
                    open(dataset_file_planner_dpo, "a").write(
                        json.dumps({"prompt": all_rollout_data[filename2], "chosen": all_rollout_data[filename2],"rejected": all_rollout_data[filename1]}) + "\n"
                    )

    print(f"Total {num_first_interactions_from_file} first interactions... Written to {dataset_file_planner}")
    print(f"Total {len(open(dataset_file_summary).readlines())} summaries... Written to {dataset_file_summary}")
    print(f"Total {len(open(dataset_file_planner).readlines()) - num_first_interactions_from_file} second interactions... Written to {dataset_file_planner}")

    print(f"Total {len(open(dataset_file_summary_dpo).readlines())} DPO summary data... Written to {dataset_file_summary_dpo}")
    print(f"Total {len(open(dataset_file_summary_best).readlines())} topK summary data... Written to {dataset_file_summary_best}")

    print(f"Finished {len(open(dataset_file_planner_dpo).readlines())} DPO planner data... Written to {dataset_file_planner_dpo}")
    print(f"Finished {len(open(dataset_file_planner_best).readlines())} topK planner data... Written to {dataset_file_planner_best}")

    dataset_summary["personas"] = list(dataset_summary["personas"])
    dataset_summary["goals"] = list(dataset_summary["goals"])
    dataset_summary["rewards"] = {k[0]+'-'+k[1]:v for k,v in expected_rewards_summary.items()}
    open(dataset_summary_file, "w").write(json.dumps(dataset_summary, indent=4))

    return dataset_filedict


class LLMAgentTrainable():
    def __init__(self, out_filename=None, **parameters):
        base_model = parameters["base_model"]
        self.model_in_path = base_model
        if 'overwrite_base_model' in parameters and parameters['overwrite_base_model']:
            self.model_out_path = self.model_in_path
        elif 'exact_outpath' in parameters:
            self.model_out_path = parameters['exact_outpath']
        else:
            self.model_out_path = (
                f"{base_model.replace('/Meta-Llama-','/trained/Meta-Llama-')}_"
                + out_filename
                + "_"
                + now_timestr()
            )
        base_model_shorthand = base_model.split("/")[-1].replace("Meta-Llama-", "")
        print(f"Model in path: {self.model_in_path}")
        print(f"Model out path: {self.model_out_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_in_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_cache_dir = (
            "models/cache"
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_in_path,
            device_map="auto",
            cache_dir=self.model_cache_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model_peft = self._get_peft(model, parameters)
        print(
            f"PEFT model loaded on devices: ",
            set([p.device for p in self.model_peft.parameters()]),
        )
        
        self.training_args = {
            'output_dir': self.model_out_path,
            'auto_find_batch_size': True,
            'logging_steps': 5,
            'evaluation_strategy': "steps",
            'eval_steps': 10,
            'save_steps': 10,
            'load_best_model_at_end': True,
            'max_seq_length': MAX_TOKENS,
            'weight_decay':parameters['weight_decay'],
            'learning_rate':parameters['learning_rate'],
            'run_name': f"{base_model_shorthand}_{out_filename}_wd{parameters['weight_decay']}_lr{parameters['learning_rate']}_lora{parameters['lora_rank']}_{parameters['lora_dropout']}_{parameters['lora_alpha']}",
        }
        if 'epochs' in parameters and parameters['epochs'] is not None:
            self.training_args['num_train_epochs'] = parameters['epochs']
        elif 'max_training_steps' in parameters and parameters['max_training_steps'] is not None:
            self.training_args['max_steps'] = parameters['max_training_steps']
        else:
            raise KeyError("Need to specify either max_training_steps or epochs")
        
        custom_args = {
            'lora_config': self.lora_config,
            'training_args': self.training_args,
            }
        custom_args['parameters'] = parameters
        os.makedirs(self.model_out_path, exist_ok=True)
        json.dump(custom_args, open(os.path.join(self.model_out_path, "custom_training_params.json"), "w"))
        torch.cuda.empty_cache()
        self.version = 0
        if self.model_in_path == self.model_out_path:
            self.version = max([int(f.split("_")[-1].split('.')[0]) for f in os.listdir(self.model_out_path) if f.startswith("version")]+[0])
        open(os.path.join(self.model_out_path, f"version_{self.version}.txt"), "w").write("")

    def _get_peft(self, model, parameters):
        target_modules = ["q_proj", "v_proj"]
        self.lora_config = {
            'r': parameters['lora_rank'],
            'target_modules': target_modules,
            'lora_alpha': parameters['lora_alpha'],
            'lora_dropout': parameters['lora_dropout'],
            'bias': "none",
            'task_type': "CAUSAL_LM",
        }
        model_peft = get_peft_model(model, LoraConfig(**self.lora_config))
        return model_peft
        
    def train_sft(self, datasets, memorization=False):
        print(f"Training using SFT.....")
        print(f"Using dataset of length: {len(datasets['train'])}")
        print(
            "------------------------------------------------------------------------------------------"
        )
        print("Dataset Example:")
        for d in datasets["train"]:
            print(d)
            break
        print(
            "------------------------------------------------------------------------------------------"
        )
        self.training_args['run_name'] += f"_SFT"

        collator = None
        collator = DataCollatorForCompletionOnlyLM(
                response_template="<|start_header_id|>assistant<|end_header_id|>",
                tokenizer=self.tokenizer,
            )

        for d in datasets["train"]:
            num_things_to_learn = (
                (collator([self.tokenizer(d["text"])])["labels"]) != -100
            ).sum()
            if num_things_to_learn < 1:
                print(f"WARNING!!! Skipping datapoint because it has no unmasked token to learn...")
        callbacks = [] if memorization else [EarlyStoppingCallback(early_stopping_patience=5)]
        if memorization:
            self.training_args['load_best_model_at_end'] = False
            print(f"Using memorization mode...\n{self.training_args}")
        trainer = SFTTrainer(
            self.model_peft,
            args=SFTConfig(**self.training_args),
            train_dataset=datasets["train"],
            eval_dataset=datasets["test"],
            dataset_text_field="text",
            tokenizer=self.tokenizer,
            data_collator=collator,
            callbacks=callbacks,
        )
        try:
            trainer.train()
        except Exception as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"Training failed with error: {e}. If this is keyboard interrupt, ignore.")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        finally:
            self.save_peft()


    def train_dpo(self, datasets):
        if 'dpo' not in self.model_out_path.lower():
            self.model_out_path = self.model_out_path + '_dpo'
        print(f"Training using DPO.....")
        print(f"Using dataset of length: {len(datasets['train'])}")
        print("------------------------------------------------------------------------------------------")
        print("Dataset Example:")
        for d in datasets['train']:
            print(d)
            break
        print("------------------------------------------------------------------------------------------")
        self.training_args['run_name'] += f"_DPO"

        self.training_args['remove_unused_columns'] = True
        self.model_ref = AutoModelForCausalLM.from_pretrained(
            self.model_in_path,
            device_map="auto",
            cache_dir=self.model_cache_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        del self.training_args['max_seq_length']
        
        dpo_trainer = DPOTrainer(
            self.model_peft,
            ref_model=self.model_ref,
            args=DPOConfig(**self.training_args),
            train_dataset=datasets['train'],
            eval_dataset=datasets['test'],
            tokenizer=self.tokenizer,
        )
        try:
            dpo_trainer.train()
        except Exception as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"DPO Training failed with error: {e}. If this is keyboard interrupt, ignore.")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        finally:
            dpo_trainer.push_to_hub()
            self.save_peft()
        

    def save_peft(self):
        model = self.model_peft.merge_and_unload()
        
        if os.path.exists(self.model_out_path):
            print(f"Moving previous model to {self.model_out_path}_{now_timestr()}")
            shutil.move(self.model_out_path, self.model_out_path+'_'+now_timestr())
        self.tokenizer.save_pretrained(self.model_out_path, safe_serialization=False)
        model.save_pretrained(self.model_out_path, safe_serialization=False)

        print("Converting checkpoint...")
        save_hf_checkpoint(
            checkpoint_dir=Path(self.model_out_path),
            model_name=self.model_in_path,
        )
        
        ## Remove previous version files
        self.version += 1
        for file in os.listdir(self.model_out_path):
            if file.startswith("version_") and file.endswith(".txt"):
                os.remove(os.path.join(self.model_out_path, file))
        open(os.path.join(self.model_out_path, f"version_{self.version}.txt"), "w").write("")
        open(os.path.join(self.model_out_path, f"configs_manual.json"), "w").write(json.dumps(self.training_args, indent=4))
