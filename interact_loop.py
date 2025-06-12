from src.task_sampler import PPOTaskSampler
from src.run_task_human import run_task

def main():
    # Create a task sampler
    sampler = PPOTaskSampler("cfg/split0_run_config_train.json", "train")

    # Sample a random task, persona, and scene
    task, persona_id, scene_data = sampler.sample_task_and_scene()

    # Run the environment loop with run_task_human
    data_filepath = "data/human_run_data.json"
    parameters = {}
    run_task(
        scene_data,
        task,
        persona_id,
        prior_user_info=None,
        data_filepath=data_filepath,
        parameters=parameters
    )

if __name__ == "__main__":
    main()