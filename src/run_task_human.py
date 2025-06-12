import os
import json
from copy import deepcopy

from src.LLMAgent import LLMAgent_Planner, LLMAgent_Persona
from src.Environment import Environment
from src.utils import VERBOSE, VERBOSE_RUNLEVEL, NO_ASK_OPTION, NO_UNDO_OPTION
from src.convert_and_evaluate import get_privileged_preferences_at_step

def run_task(
    scene,
    task,
    persona_id,
    prior_user_info,  ## only use for custom prior info; usually you shouldn't need it
    data_filepath,
    parameters,
    additional_data={},
):
    env = Environment(scene)
    
    max_steps_executed = parameters['max_steps_executed']
    max_steps_predicted = parameters['max_steps_predicted']
    action_only_planner = (not parameters['interleave_thought']) if 'interleave_thought' in parameters else False
    no_history_planner = parameters['no_history_planner'] if 'no_history_planner' in parameters else False
    unconstrained_planner = parameters['unconstrained_planner'] if 'unconstrained_planner' in parameters else False
    if 'no_ask_option' not in parameters:
        parameters['no_ask_option'] = False
    parameters['no_ask_option'] = parameters['no_ask_option'] or parameters['use_privileged_prior']
    planner = LLMAgent_Planner(persona_id, **parameters)
    persona = LLMAgent_Persona(persona_id, **parameters)
    force_question_every_n = parameters['force_question_every_n'] if 'force_question_every_n' in parameters else None
    planner.add_user_info(prior_user_info)
    reward_ask = -1
    max_completion_reward = 50
    steps_remaining = max_steps_executed
    max_steps_w_failure = max_steps_predicted
    if force_question_every_n is not None:
        max_steps_w_failure = int(max_steps_w_failure * (1/force_question_every_n + 1))
        max_steps_executed = int(max_steps_executed * (1/force_question_every_n + 1))

    rollout_data = {
        "task": task,
        "persona_id": persona.agent_name,
        "finished": False,
        "summary_history": None,
        "privileged_goal": None,
        "prior_persona_knowledge": None,
        "total_reward": 0,
        "num_questions": 0,
        "num_corrections": 0,
        "goal_completion_reward": 0,
        "goal_completion_fraction": 0,
        "sim_steps": 0,
        "episode_length": 0,
        "rollout": [],
        "summary": None,
        "initial_scene": json.dumps(env.full_scene),
        "actions_performed": None,
        "entities_created": None,
        "objects_used": None,
        "mixtures": None,
        "transformations": None,
        "final_object_locations": None,
    }
    rollout_data.update(additional_data)
    persona.task = task
    if parameters['use_privileged_prior']:
        assert prior_user_info is None, "Prior user info and privileged prior cannot both be used"
        planner.add_user_info(persona.privileged_preferences)
    rollout_data["prior_persona_knowledge"] = planner.user_info
    if os.path.exists(data_filepath):
        rollout_data = json.load(open(data_filepath))
        env = Environment(json.loads(rollout_data["initial_scene"]))
    failure_streak = 0
    privileged_info_at_step = None
    for step in range(max_steps_w_failure):
        if len(rollout_data["rollout"]) > step:
            if rollout_data["rollout"][step]["success"]:
                action = rollout_data["rollout"][step]["action"]
                env.step(action)
                continue
        if VERBOSE_RUNLEVEL:
            print(f"******** Step {step} ********")

        privileged_info_at_step = get_privileged_preferences_at_step(rollout_data)
        progress_summ = env.summarize_progress()
        force_ask = False
        if force_question_every_n is not None:
            actions = [rolloutstep['action_enum'] for rolloutstep in rollout_data['rollout'][-force_question_every_n:]]
            if 'ask' not in actions and len(actions) == force_question_every_n:
                force_ask = True
        action, thought, grammar, prompt, prob_response = planner(env, 
                                                rollout_data["rollout"], 
                                                task, 
                                                progress_summ=progress_summ, 
                                                privileged_info_at_step=privileged_info_at_step if parameters['use_privileged_prior'] else None, 
                                                no_generate=False,
                                                action_only=action_only_planner,
                                                no_history=no_history_planner,
                                                unconstrained_planner=unconstrained_planner,
                                                force_ask=force_ask,
                                                )
        rollout_data["episode_length"] += 1
        success, msg, action_enum, action_args = env.step(action)
        if success:
            steps_remaining -= 1
        if VERBOSE_RUNLEVEL:
            print(f"--------------------------------")
            print(f"Thought: {thought}\nAction: {action}")
            print(f"Observation: {msg}\n", end="")
        current_interaction = {
            "step": env.step_num,
            "progress_summary": progress_summ,
            "privileged_info": privileged_info_at_step,
            "grammar": grammar,
            "thought": thought,
            "action": action,
            "action_enum": action_enum,
            "action_args": action_args,
            "success": success,
            "observation": msg,
            "user_feedback": None,
            "reward": None,
            "planner_prompt": prompt,
            "prob_response": prob_response,
        }
        reward = None
        rollout_data["sim_steps"] = env.step_num
        if msg == "DONE":
            failure_streak = 0
            task_done = max_completion_reward
            try:
                results_dict = persona(env, rollout_data, current_interaction, ask_or_correct='confirm_done', task=task)
                rollout_data["num_pref_violated"] = results_dict["penalty"]
                rollout_data["num_pref_satisfied"] = results_dict["max_penalty"] - results_dict["penalty"]
                user_response_subjective = results_dict["messages"]
                response_objective = results_dict["reward_fraction"]
            except Exception as e:
                user_response_subjective, response_objective = "NOT evaluated", 0
            current_interaction["user_feedback"] = user_response_subjective
            if VERBOSE_RUNLEVEL:
                print(f"User: ({response_objective}) {user_response_subjective}\n")
            reward = response_objective * task_done
            rollout_data["goal_completion_fraction"] = response_objective
            rollout_data["goal_completion_reward"] = reward
            rollout_data["finished"] = True
        elif not success:
            pass
        elif action_enum in ["ask"]:
            assert (
                not NO_ASK_OPTION
            ), f"The model wasn't supposed to be asking questions!!"
            failure_streak = 0
            user_response_subjective, response_objective = persona(
                env, rollout_data, current_interaction, ask_or_correct="ask", task=task
            )
            current_interaction["user_feedback"] = user_response_subjective
            if VERBOSE_RUNLEVEL:
                print(f"User: ({response_objective}) {user_response_subjective}\n")
            reward = reward_ask
            rollout_data["num_questions"] += 1
        elif action_enum in [
            "move",
            "move_from",
            "mix",
            "cook",
            "pour",
            "pour_into",
            "peel",
            "place",
            "serve",
            "chop",
            "chop_obj",
            "freeform",
            "freeform_contents",
            "heat",
            "turn_on",
            "turn_off",
        ]:
            failure_streak = 0
        elif action_enum in ['undo']:
            assert not NO_UNDO_OPTION, f"The model wasn't supposed to be undoing actions!!"
        else:
            failure_streak += 1
            reward = 0
            if action_enum not in [
                "find",
                "search",
                "done",
                "open",
                "close",
                "look_for",
                "search_to_find",
            ]: 
                print(f"WARNING: You forgot to handle action {action_enum} in invoking the persona agent!!!")

        if step >= max_steps_w_failure - 1 or steps_remaining < 0:
            rollout_data["goal_completion_reward"] = 0
            rollout_data["finished"] = True

        current_interaction["reward"] = reward

        rollout_data["rollout"].append(current_interaction)
        if reward is not None:
            rollout_data["total_reward"] += reward

        if rollout_data["privileged_goal"] is None:
            rollout_data["privileged_goal"] = persona.preferences_list
        rollout_data.update(env.get_full_state())
        json.dump(rollout_data, open(data_filepath, "w"), indent=4)

        if rollout_data["finished"]:
            if VERBOSE:
                print()
                print(f"   - total_reward : {rollout_data['total_reward']}")
                print(f"   - num_questions : {rollout_data['num_questions']}")
                print(f"   - num_corrections : {rollout_data['num_corrections']}")
                print(f"   - goal_completion_reward : {rollout_data['goal_completion_reward']}")
                print(f"   - sim_steps : {rollout_data['sim_steps']}")
                print(f"   - episode_length : {rollout_data['episode_length']}")
                print()
            break

    return deepcopy(planner), deepcopy(persona)