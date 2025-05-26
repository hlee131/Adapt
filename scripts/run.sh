# Evaluation command from README 

llama_model=meta-llama/Llama-3.1-8B-Instruct

case "$1" in 
  eval)
    planner_model=${2:-$llama_model}

    for split in 0 1 2 3
      do
        for gen in seen_persona unseen_persona
          do
            python run_eval.py \
                --generalization_category $gen \
                --crossvalidation_split $split \
                --logs_dir logs \
                --model_name_planner $planner_model \
                --model_name_base $llama_model
        done
    done
    ;;

  train)  
    data=${2:-"data/"}
    trained_model=${3:-"models/reflection_dpo.pt"}

    python train.py \
      --base_model $llama_model \
      --trained_model $trained_model \
      --data_path $data \
    ;;

  *)
    echo "Invalid task: $1"
    ;;
esac
