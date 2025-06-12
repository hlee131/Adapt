# Evaluation command from README 

export WANDB_API_KEY=e0553b375da9957b03a9fe9face3c18af2f49714
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
                --planner_adapter_name $planner_model \
                --model_name_base $llama_model
        done
    done
    ;;

  train)  
    data=${2:-"data/combined/"}
    trained_model=${3:-"models/reflection_wpo"}

    python train.py \
      --base_model $llama_model \
      --trained_model $trained_model \
      --data_path $data \
    ;;
 
  gen)
    out_dir=${2:-"data"}
    for split in 0 1 2 3
    do
      python run_dataset_gen.py \
        --run_config cfg/split$split"_run_config_train.json" \
        --out_dir $out_dir \
        --model_name_base meta-llama/Llama-3.1-8B-Instruct
    done
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
