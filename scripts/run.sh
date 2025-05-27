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
    data=${2:-"data/combined/"}
    trained_model=${3:-"models/reflection_dpo.pt"}

    python train.py \
      --base_model $llama_model \
      --trained_model $trained_model \
      --data_path $data \
    ;;

  *)
    echo "Invalid task: $1"
    ;;
  
  ppo)
    model=${2:-$llama_model}
    split=${3:-0}
    shift 3 2>/dev/null || shift 2 2>/dev/null || shift 1 2>/dev/null || true
    echo "Starting PPO training..."
    bash scripts/train_ppo.sh $model $split "$@"
    ;;

  *)
    echo "Invalid task: $1"
    echo "Available tasks: eval, train, ppo"
    echo ""
    echo "Examples:"
    echo "  bash scripts/run.sh eval"
    echo "  bash scripts/run.sh train"
    echo "  bash scripts/run.sh ppo meta-llama/Llama-3.1-8B-Instruct 0"
    ;;
esac
