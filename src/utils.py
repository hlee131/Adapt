import json
import os
import random
import time
import re

import torch
import numpy as np
from transformers import AutoTokenizer

from env.scene import SceneGenerator, key_from_scene, scene_from_key

SEED = int(os.environ.get('SEED', 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')

def get_from_env(env_varname, default):
    python_var = os.environ.get(env_varname, None)
    if python_var is None:
        print(f"\nNo {env_varname} found in the environment variables.........\nSetting to {default} as default.\n")
        return default
    else:
        return python_var == '1'

NO_ASK_OPTION = get_from_env('NO_ASK_OPTION', False)
NO_UNDO_OPTION = get_from_env('NO_UNDO_OPTION', True)
NO_CONSTRAINED_GENERATION = get_from_env('NO_CONSTRAINED_GENERATION', False)

VERBOSE = True
VERBOSE_RUNLEVEL = True
NO_REPEAT_HORIZON = 4
NO_REPEAT_MAX_REPITITIONS = 3
MAX_STUCK_STEPS = 10
MAX_STEPS_FOR_SUMMARIZATION_PROMPT = 50

DEVICE = "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"
if torch.cuda.is_available():
    DEVICE = "cuda"

action_options = {
    "ask": "Ask <X>",
    "open": "Open <X>",
    "close": "Close <X>",
    "heat": "Heat <X>",
    "turn_on": "Turn on <X>",
    "turn_off": "Turn off <X>",
    "search_to_find": "Search <X> to find <X>",
    "search": "Search <X>",
    "look_for": "Look for <X>",
    "find": "Find <X>",
    "move_from": "Move <X> from <X> to <X>",
    "move": "Move <X> to <X>",
    "serve": "Serve the object <X> at <X>",
    "place": "Place the object <X> at <X>",
    "mix": "Mix all items in <X> to get <X>",
    "cook": "Cook items in <X> to get <X>",
    "chop_obj": "Chop the object <X> to get <X>",
    "chop": "Chop <X> to get <X>",
    "pour": "Pour <X> from <X> to <X>",
    "pour_into": "Pour <X> from <X> into <X>",
    "done": "Declare Done",
    "complete": "Task Complete",
    "undo": "Undo Last Action",
    "freeform_contents": "<X> items in <X> to get <X>",
    "freeform": "<X> the object <X> to get <X>",
}

state_change_actions = [
    "dice","slice","mince","peel","grate","julienne","cube","blend","whisk","beat","fold","knead","marinate","sear","simmer","boil","steam","poach","fry","sauté","grill","roast","bake","broil","braise","stew","blanch","reduce","deglaze","caramelize","glaze","season","crack","shuck","zest","purée","mash","strain","drain","stuff","score","tenderize","dust","sprinkle","sift","separate","whip","garnish","scramble","flip","mix","stir","flip","fry","melt","spread","drizzle","stack","ladle","roll","beat","grease","fold","serve","assemble","layer","toast","crush","brew","grind","grate","shave","shred","cut","trim","clean","wash","rinse","dry","soak","toss","coat"
]
state_change_actions += [elem.capitalize() for elem in state_change_actions]

privileged_task_descriptions = {
    "Prepare eggs for breakfast": "To prepare eggs, you should whisk eggs with salt and pepper, heat butter or oil in a pan, then cook on medium heat, stirring occasionally for scrambled eggs or letting them set for fried eggs, prepare any sides, and serve." ,
    "Prepare omelette for breakfast": "To make an omelette, you should whisk eggs with salt and pepper, pour into a heated, greased pan, cook until set, add fillings, prepare sides, and serve.",
    "Prepare cereal for breakfast": "To prepare cereal, you should pour cereal into a bowl, add milk, any toppings and serve.",
    "Make toast and coffee for breakfast": "To make toast, you should toast bread by placing a slice of bread in the toaster, spread butter or jam, and serve. To make coffee, you should brew coffee by transferring coffee grounds and water into the coffee machine, add milk or sugar if desired, and serve.",
    "Make yoghurt parfait for breakfast": "To make a yoghurt parfait, you should layer yoghurt, granola, and fruit in a glass, bowl, or cup, and serve.",
    "Make tea and eggs for breakfast": "To make tea, you should steep the preferred tea bag in hot water, add milk or sugar if desired, and serve. To prepare eggs, you should whisk eggs with salt and pepper, heat butter or oil in a pan, then cook on medium heat, stirring occasionally for scrambled eggs or letting them set for fried eggs, prepare any sides, and serve.",
    "Make cereal and coffee for breakfast": "To prepare cereal, you should pour cereal into a bowl, add milk, any toppings and serve. To make coffee, you should brew coffee by transferring coffee grounds and water into the coffee machine, add milk or sugar if desired, and serve.",
    "Make toast and eggs for breakfast": "To make toast, you should toast bread by placing a slice of bread in the toaster, spread butter or jam, and serve. To prepare eggs, you should whisk eggs with salt and pepper, heat butter or oil in a pan, then cook on medium heat, stirring occasionally for scrambled eggs or letting them set for fried eggs, prepare any sides, and serve.",
}

parsed_actions = {
    'syntax_ask': [
        "ask"
        ],
    'syntax_states': [
        "open",
        "close",
        "heat",
        "turn_on",
        "turn_off"
        ],
    'syntax_find': [
        "look_for",
        "find"
        ],
    'syntax_search': [
        "search"
        ],
    'syntax_new_contents': [
        "mix",
        "cook",
        "freeform_contents"
        ],
    'syntax_new_object': [
        "chop",
        "freeform"
        ],
    'syntax_move': [
        "move"
        ],
    'syntax_move_from': [
        "move_from"
        ],
    'syntax_pour': [
        "pour",
        "pour_into"
        ],
    'syntax_single': [
        "done",
        "complete"
        ],
}


action_description = [
    "- Open <X>: open an instance of an articulated furniture or object. e.g. Open cabinet",
    "- Close <X>: close an instance of an articulated furniture or object. e.g. Close cabinet",
    "- Heat <X>: heat a container, which is located on a heating appliance. e.g. Heat pan_0",
    "- Turn on <X>: turn on an appliance. e.g. Turn on stove_0",
    "- Turn off <X>: turn off an appliance. e.g. Turn off stove_0",
    "- Search <X>: search a container in the house. e.g. Search counter_0",
    "- Look for <X>: Look for an object in the whole house. Use this with a single word referring to the most generic category name of an object. Be sure to look for objects one-at-a-time. e.g. Look for milk",
    "- Move <X> to <Y>: move an object X from wherever it currently is to the furniture or location Y. e.g. Move plate_0 to table_2",
    "- Mix all items in <X> to get <Y>: mix items that exist in a container, typically to create a new entity. e.g. Mix all items in bowl_0 to get cake_batter",
    "- Cook items in <X> to get <Y>: cook something in a stove, oven or other such appliance to create a cooked version of that entity. e.g. Cook items in pan_0 to get scrambled_eggs",
    "- Chop <X> to get <Y>: chop items on a cutting board to create a chopped version of that entity. e.g. Chop apple_0 to get chopped_apple",
    "- Pour <X> from <Y> to <Z>: pour an entity X from one container Y to another container Z. e.g. Pour milk from milk_carton_0 to mug_0",
    "- Move <X> from <Y> to <Z>: move content X of container Y to another container Z. e.g. Move apple from apple_bag_0 to bowl_1",
    "- <X> items in <Y> to get <Z>: freeform action to change object state, such as whisk, heat, blend, etc. e.g. whisk items in pan_0 to get custard, brew coffee_grounds to get brewed_coffee etc.",
    "- <X> the object <Y> to get <Z>: freeform action to change object state, such as chop, peel, crack, wash, wipe, etc. e.g. chop the object tomato_1 to get finely_chopped_tomato, chop the object onion_0 to get sliced_onion, crack the object egg_4 to get cracked_egg, etc.",
    '- Ask <X>: ask a freeform question to the user to decide between multiple options and make sure you adhere to the user\'s preferences. e.g. Ask "Would you like the eggs sunny-side-up or scrambled?"',
    "- Undo Last Action: undo the last action that was performed. Use this when the effect of the last action was not what you expected. This action cannot be repeated.",
    "- Declare Done: indicate that the task is complete. Make sure to use this exactly once at the end of the task.",
]

if NO_UNDO_OPTION:
    del action_options["undo"]
    action_description = [a for a in action_description if not a.startswith("- Undo Last Action")]

if NO_ASK_OPTION:
    del action_options["ask"]
action_description_no_ask = [a for a in action_description if "- Ask" not in a]
action_description_no_ask = "\n".join(action_description_no_ask)
action_description = "\n".join(action_description)

def now_timestr():
    t = time.localtime()
    return "{:02d}{:02d}_{:02d}{:02d}".format(t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)

def planner_system_prompt(persona_id, user_info, env, task, no_ask_option=False, action_only=True):
    action_description_based_on_ask = action_description_no_ask if no_ask_option else action_description
    system_prompt = (
        f"You are an expert at task planning, and know how to provide assistance in a manner that {persona_id} wants for preparing and serving breakfasts such as making cereal, pancakes, toast, waffles, french toast, coffee, tea, etc. You have the ability to take the following actions:\n"
        + action_description_based_on_ask
        + "\n\nFor a given task, you will provide the next action required to achieve a given task. "
    )
    if action_only:
        system_prompt += 'e.g. \nAction: Move apple_0 from counter_0 to table_0.'
    else:
        system_prompt += 'Before each step you will provide your reason or intention behind your action. \ne.g. \nThought: I need to find eggs to prepare an omelet \nAction: Find eggs. '
    
    system_prompt += "Do NOT repeat your last action.\nNote that you must do the task in a way that the user prefers. Think of different variations, modifications, sides, etc. applicable to the given task, and do the task in a way that you think the user would prefer."

    if not NO_UNDO_OPTION:
        system_prompt += " You can use undo last action if the outcome is not what you desired. Use it to fix your plan, rather than trying to find workarounds."
    if not no_ask_option:
        system_prompt += " You have interacted with this person before, so use prior information you have about the person to complete the task as much as possible. Only ask questions if you don't have any related information. Be sure to only ask about their preferences. Think about whether multiple options are available for a particular ingredient, and think creatively about toppings and sides that the person might prefer. Do not ask for help in finding things; the user may not know what objects exist in the environment and where they might be located. Ask general questions to learn about the user's preferences which can prove useful in preparing future breakfasts in addition to the one at hand."
    system_prompt += "\n\n"
    if len(user_info) > 0:
        system_prompt += f"You know the following about {persona_id}:\n{user_info}\n\n"
    else:
        system_prompt += f"You have no prior information about {persona_id}.\n\n"
    system_prompt += (
        "You will be performing tasks in a house with the following layout:\n"
        + env.prompt_string(furniture_only=True)
        + "\n\n"
    )
    system_prompt += f"What is the next action required to achieve the task: {task}?"
    return system_prompt

def planner_system_prompt_todo(persona_id, user_info, env, task, no_ask_option=False):
    add_prompt = ''
    if not no_ask_option:
        add_prompt = ', and all the information that you need to ask the user about their preferences in order to make sure you satisfy them'
    system_prompt = (
        f"You are an expert at task planning, and your job is to instruct a robot to do the given task in a manner that {persona_id} wants for preparing and serving breakfasts such as making cereal, pancakes, toast, waffles, french toast, coffee, tea, etc."
        + f" You will summarize what needs to be done to achieve the task goal{add_prompt}. Be sure to include notes of caution for the robot so that it does not make mistakes in the process of performing the task, including the components, timing and location of different actions."
    )
    system_prompt += "\n\n"
    if len(user_info) > 0:
        system_prompt += f"You know the following about {persona_id}:\n{user_info}\n\n"
    else:
        system_prompt += f"You have no prior information about {persona_id}.\n\n"
    system_prompt += (
        "You will be performing tasks in a house with the following layout:\n"
        + env.prompt_string(furniture_only=True)
        + "\n\n"
    )
    system_prompt += f"What do you need to do{'' if no_ask_option else ' and ask'} the user to achieve the task: {task}?"
    return system_prompt


def prompt_from_rollout(
    rollout, assistant, skip=[], change_user_to=None, skip_failed=False, action_only=False,
):
    prompt_msgs = []
    robot_failures = []

    def check_and_append(role, value):
        if role in skip:
            return
        if assistant == role:
            role = "assistant"
        prompt_msgs.append((role, value))

    for snapshot in rollout:
        if not snapshot["success"] and skip_failed:
            continue
        if assistant in ["user","none"]:
            check_and_append("robot", f"Action: {snapshot['action']}")
        elif assistant in ["robot", "memory"]:
            if 'thought' not in snapshot and snapshot['action_enum'] == 'done':
                snapshot['thought'] = "I have completed the task. I will declare it done."
            if action_only:
                check_and_append(
                    "robot", f"Action: {snapshot['action']}"
                )
            else:
                check_and_append(
                    "robot", f"Thought: {snapshot['thought']}\nAction: {snapshot['action']}"
                )
        else:
            raise ValueError(
                f"Did you forget to account hidden interaction for role: {assistant}?"
            )
        if 'observation' in snapshot and snapshot["observation"] is not None:
            check_and_append("environment", f"Observation: {snapshot['observation']}")
        if 'user_feedback' in snapshot and snapshot["user_feedback"] is not None:
            check_and_append("user", f"{snapshot['user_feedback']}")

    return prompt_msgs, robot_failures


def summarize_prompt(rollout_past, persona_id, user_info, get_text=False):
    prompt_msgs = [
        (
            "system",
            f"You are an expert at task planning to provide personalized assistance to a user. You will reflect upon and understand user feedback in addition to any prior knowledge you already have about the person, so that you can use it in future tasks. You will respond with a one paragraph summary of takeaways from information obtained through user feedback in the interaction, which could be useful in future assistive tasks. The summary must include information about the user regarding their preferences regarding the task, dietary choices and preferences regarding the prcoess of preparing meals and communication, etc. You should think about what information could be useful in assisting with similar tasks in the future, such as choices that the user is indifferent about, so that you don't have to ask them about it in the future, and strategies regarding what kind of questions you need to ask, and when, to get the necessary information in time and avoid repeating your mistakes.\n\n",
        )
    ]
    if len(user_info) > 0:
        prompt_msgs.append(
            ("user", f"You already knew the following about {persona_id}:\n{user_info}\n\n")
        )
    else:
        prompt_msgs.append(("user", f"You had no information about {persona_id} prior to this interaction.\n\n"))
    ## if it is a list of many episode rollouts
    if isinstance(rollout_past[0], list):
        for i_rollout, single_rollout in enumerate(rollout_past):
            prompt_msgs += [("user", f"Interaction {i_rollout+1}:\n")]
            prompt_msgs_add, _ = prompt_from_rollout(
                single_rollout[:MAX_STEPS_FOR_SUMMARIZATION_PROMPT],
                assistant="memory",
                skip=[],
                change_user_to=persona_id,
                skip_failed=False,
            )
            prompt_msgs += prompt_msgs_add
    else:
        prompt_msgs_add, _ = prompt_from_rollout(
            rollout_past,
            assistant="memory",
            skip=[],
            change_user_to=persona_id,
            skip_failed=False,
        )
        prompt_msgs += prompt_msgs_add
    prompt_msgs.append(
        (
            "user",
            f"Summarize what you know about {persona_id}'s preferences based on what you already knew and the above interaction.\n\n",
        )
    )

    if get_text:
        full_prompt = _tokenizer.apply_chat_template(
            [{"role": p[0], "content": p[1]} for p in prompt_msgs],
            tokenize=False,
            add_generation_prompt=True,
        )
        return full_prompt
    else:
        return prompt_msgs


def get_logfilepath(
    persona_id,
    task,
    rollout_num=0,
    logs_dir="logs",
    custom_prefix="",
    **kwargs,
):
    os.makedirs(f"{logs_dir}/dump", exist_ok=True)
    os.makedirs(f"{logs_dir}/data", exist_ok=True)
    os.makedirs(f"{logs_dir}/summaries", exist_ok=True)
    timestr = now_timestr()
    runstr = f"{persona_id}_{task.replace(' ','_')}"
    kwd = custom_prefix
    for arg in kwargs:
        if kwargs[arg]:
            kwd += f"{arg.strip().replace('_','')}_"
    log_filepath = f"{logs_dir}/dump/{kwd}{runstr}.txt"
    data_filepath = log_filepath.replace("dump/", "data/").replace(".txt", f"_{rollout_num}.json")
    if os.path.exists(data_filepath):
        print(f"Data file already exists: {data_filepath}")
        return None, None

    return log_filepath, data_filepath

def get_all_permutations(
    item_list, num_items_per=None, allow_repitition=False, ordered=True
):
    if num_items_per is None:
        num_items_per = len(item_list)
    combinations = [[]]
    for _ in range(num_items_per):
        gc_new_list = []
        for gc_prev in combinations:
            for next_goal in item_list:
                if allow_repitition or next_goal not in gc_prev:
                    gc_new_list.append(gc_prev + [next_goal])
        combinations = gc_new_list

    if not ordered:
        combinations = set([";".join(sorted(item)) for item in combinations])
        combinations = [item.split(";") for item in combinations]

    return combinations


def create_grammar(env, no_ask_option=False):

    rules = []

    rules.append(
        "existing_entity ::= " + " | ".join([f'"{k}"' for k in env.get_all_entities()])
    )
    rules.append(
        "existing_entity_or_content ::= " + " | ".join([f'"{k}"' for k in env.get_all_entities(including_contents=True)])
    )
    rules.append(
        "existing_containers ::= "
        + " | ".join([f'"{k}"' for k in env.get_containers()])
    )
    any_containers_filled = len(env.get_containers(filled_only=True)) > 0
    if any_containers_filled:
        rules.append(
            "existing_filled_containers ::= "
            + " | ".join([f'"{k}"' for k in env.get_containers(filled_only=True)])
        )
    rules.append(
        "existing_foods ::= " + " | ".join([f'"{k}"' for k in env.get_foods()])
    )
    rules.append(
        "existing_food_containers ::= "
        + " | ".join([f'"{k}"' for k in env.get_food_packets()])
    )
    rules.append(
        "existing_content_containers ::= "
        + " | ".join([f'"{k}"' for k in env.get_content_from_container()])
    )
    rules.append(
        "action_name ::= "
        + " | ".join([f'"{k.lower()}"' for k in state_change_actions]+[f'"{k.title()}"' for k in state_change_actions])
    )

    rules_from_file = open("src/planner_grammar.ebnf").read()

    if not any_containers_filled:
        rules_from_file_list = rules_from_file.split("\n\n")
        rules_from_file_list = [r for r in rules_from_file_list if not r.startswith('syntax_new_contents')]
        rules_from_file = "\n\n".join(rules_from_file_list)
        rules_from_file = rules_from_file.replace("syntax_new_contents |", "")
        
    if NO_UNDO_OPTION:
        rules.append('syntax_single ::= "Declare Done" | "Task Complete"')
    else:
        rules.append('syntax_single ::= "Declare Done" | "Task Complete" | "Undo Last Action"')
    
    if no_ask_option:
        rules_from_file = rules_from_file.replace("syntax_ask | ", "")

    grammar_str = rules_from_file + "\n\n".join(rules)

    return grammar_str



def map_to_instruct(data, response_template = "<|start_header_id|>assistant<|end_header_id|>"):
    full_text = data["text"]
    if response_template not in full_text:
        raise RuntimeWarning(f"Response template {response_template} not found in data {full_text[:1000]}...{full_text[-1000:]}.")
    prompt = full_text[:full_text.index(response_template)+len(response_template)]
    completion = full_text[full_text.index(response_template)+len(response_template):]
    return {"prompt": prompt, "completion": completion}

def map_from_instruct(data, response_template = "<|start_header_id|>assistant<|end_header_id|>"):
    empty_header = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>'
    prompts = []
    for p in data["prompt"]:
        if p.endswith('Action:'):
            p = p[:-len('Action:')]
        if p.endswith('Thought:'):
            p = p[:-len('Thought:')]
    return {"text":[(p+c) for p,c in zip(data["prompt"],data["completion"])]}

def map_from_instruct_only_action(data, response_template = "<|start_header_id|>assistant<|end_header_id|>"):
    empty_header = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>'
    prompts = []
    for p in data["prompt"]:
        if p.endswith('Action:'):
            p = p[:-len('Action:')]
        if p.endswith('Thought:'):
            p = p[:-len('Thought:')]
    return {"text":[(p+'Action:'+c.split('Action:')[-1]) for p,c in zip(data["prompt"],data["completion"])]}

def _cut_prompt(prompt):
    empty_header = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>'
    prompt = prompt.replace(empty_header,'')
    system_prompt = prompt.split('<|eot_id|>')[0] + '<|eot_id|>'
    remaining_prompt = '<|eot_id|>'.join(prompt.split('<|eot_id|>')[1:]).replace('\n\nAction:Thought: \nAction:','\n\nAction:')
    splits = remaining_prompt.split('Thought:')
    if len(splits) > 1:
        remaining_prompt = 'Action:'.join([splits[0]]+[p.split('Action:')[-1] for p in splits[1:]])
    return system_prompt + remaining_prompt
            
def remove_thought_in_dpo(data):
    processed = [{"prompt": _cut_prompt(p), "chosen": c.split('Action:')[-1].strip(), "rejected": r.split('Action:')[-1].strip()} for p,c,r in zip(data["prompt"],data["chosen"],data["rejected"])]
    return {"prompt":[elem['prompt'] for elem in processed], "chosen":[elem['chosen'] for elem in processed], "rejected":[elem['rejected'] for elem in processed]}

def remove_thought_in_sft(data):
    empty_header = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>'
    return {"text":[_cut_prompt(p).replace(empty_header,'').replace('Action:Action:','Action:').replace('Thought:Thought:','Thought:') for p in data["text"]]}

def map_from_instruct_only_action(data, response_template = "<|start_header_id|>assistant<|end_header_id|>"):
    return remove_thought_in_sft(map_from_instruct(data, response_template))


def save_hf_checkpoint(checkpoint_dir,model_name):
    if model_name is None:
        model_name = checkpoint_dir.name

    config = dict(
            rope_base=500000,
            block_size=8192,
            n_layer=80,
            n_head=64,
            n_local_heads=8,
            dim=8192,
            intermediate_size=28672,
            vocab_size=128256,
        )

    if "8B" in model_name:
        config = dict(
            rope_base=500000,
            block_size=8192,
            n_layer=32,
            n_head=32,
            n_local_heads=8,
            dim=4096,
            intermediate_size=14336,
            vocab_size=128256,
        )

    config["head_dim"] = config["dim"] // config["n_head"]
    print(f"Using the following model config. This could be wrong if using a different model than Llama 3.1 70B or 8B:\n{config}")

    # Load the json file containing weight mapping
    model_map_json = checkpoint_dir / "pytorch_model.bin.index.json"

    assert model_map_json.is_file()

    with open(model_map_json) as json_map:
        bin_index = json.load(json_map)

    weight_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }
    bin_files = {checkpoint_dir / binv for binv in bin_index["weight_map"].values()}

    def permute(w, n_head):
        dim = config['dim']
        return (
            w.view(n_head, 2, config['head_dim'] // 2, dim)
            .transpose(1, 2)
            .reshape(config['head_dim'] * n_head, dim)
        )

    merged_result = {}
    for file in sorted(bin_files):
        state_dict = torch.load(
            str(file), map_location="cpu", mmap=True, weights_only=True
        )
        merged_result.update(state_dict)
    final_result = {}
    for key, value in merged_result.items():
        if "layers" in key:
            abstract_key = re.sub(r"(\d+)", "{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]

        final_result[new_key] = value

    for key in tuple(final_result.keys()):
        if "wq" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            q = permute(q, config['n_head'])
            k = permute(k, config['n_local_heads'])
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]
    print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
    torch.save(final_result, checkpoint_dir / "model.pth")
