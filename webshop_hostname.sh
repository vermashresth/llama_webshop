#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=80G               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -t 0-8:00             # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_compute,gpu,seas_gpu,gpu_requeue        # Partition to submit to
#SBATCH -o slurm_logs/myoutput_%j.out
#SBATCH -e slurm_logs/myerrors_%j.err
start_time=$SECONDS

source ~/.bashrc
module load Mambaforge/22.11.1-fasrc01
module load cuda/12.4.1-fasrc01 cudnn/9.5.1.17_cuda12-fasrc01
nvidia-smi


conda activate webshop
cd web/WebShop/
python -m web_agent_site.app --log > flask.log 2>&1 &


cd ../..
# conda activate LLM_RL
conda activate /n/netscratch/tambe_lab/Lab/sverma/LLM_RL_pt2
# python3.9 llm_rl_scripts/webshop/mc/train_mc_bilevel_weightlearner.py HF gpt2 /n/home02/sverma/LMRL-Gym/web/WebShop/web_agent_site/envs/clean_0_1200_seed233_t0.5_1run.json  /n/home02/sverma/LMRL-Gym/web/WebShop/web_agent_site/envs/clean_0_1200_seed233_t0.5_1run.json --embeddings-path /n/home02/sverma/LMRL-Gym/JaxSEQ/JaxSeq/../web/WebShop/web_agent_site/envs/clean_gpt2_embedding_npz.npz --outputs-path /n/netscratch/tambe_lab/Lab/sverma/llm-rl/results --policy_n_rollouts 5 --train_bsize 8 --eval_loss_bsize 8 --grad_accum_steps 2 --wandb_project webshop --sample_frac 1 --val_split 0.3 --tau 1 --train_task_frac 0.5 --exp_name mc_a0_blvl_webshop  --log_every 250 --num_inner_iter 50 --num_outer_iter 100 --inner_opt_steps 20 --phi_update_factor 20 --cql_weight 10 --val_eval_same_distr --poisoning_frac 0.5 --init_alpha 0

# python3.9 llm_rl_scripts/webshop/mc/train_mc_bilevel_weightlearner_modelphi3B.py HF meta-llama/Llama-2-7b-hf \
# /n/home02/sverma/LMRL-Gym/web/WebShop/web_agent_site/envs/clean_0_1200_seed233_t0.5_1run.json  \
# /n/home02/sverma/LMRL-Gym/web/WebShop/web_agent_site/envs/clean_0_1200_seed233_t0.5_1run.json \
# --model_family llama \
# --embeddings-path /n/home02/sverma/LMRL-Gym/JaxSEQ/JaxSeq/../web/WebShop/web_agent_site/envs/clean_gpt2_embedding_npz.npz \
# --outputs-path /n/netscratch/tambe_lab/Lab/sverma/llm-rl/results --policy_n_rollouts 5 --train_bsize 8 \
# --eval_loss_bsize 8 --grad_accum_steps 2 --wandb_project webshop --sample_frac 1 --val_split 0.3 --tau 1 \
# --train_task_frac 0.5 --exp_name mc_a0_blvl_webshop  --log_every 250 --num_inner_iter 50 --num_outer_iter 100 \
# --inner_opt_steps 20 --phi_update_factor 20 --cql_weight 10 --val_eval_same_distr --poisoning_frac 0.5 --init_alpha 0

python llm_rl_scripts/webshop/mc/train_mc_bilevel_pytorch_llama3.py \
  meta-llama/Meta-Llama-3-8B \
  /n/home02/sverma/LMRL-Gym/web/WebShop/web_agent_site/envs/clean_0_1200_seed233_t0.5_1run.json \
  /n/home02/sverma/LMRL-Gym/web/WebShop/web_agent_site/envs/clean_0_1200_seed233_t0.5_1run.json \
  --embeddings-path /n/home02/sverma/LMRL-Gym/JaxSEQ/JaxSeq/../web/WebShop/web_agent_site/envs/clean_gpt2_embedding_npz.npz \
  --lora-r 16 --lora-alpha 32 \
  --train-bsize 1 --max-length 512 --bf16 --gradient_checkpointing \
  --num-outer-iter 10 --num-inner-iter 100 \
  --outputs-path  /n/netscratch/tambe_lab/Lab/sverma/llm-rl/results --hf_token $HF_TOKEN



python llm_rl_scripts/webshop/mc/train_mc_bilevel_pytorch_llama3.py \
  microsoft/phi-1_5 \
  /n/home02/sverma/LMRL-Gym/web/WebShop/web_agent_site/envs/clean_0_1200_seed233_t0.5_1run.json \
  /n/home02/sverma/LMRL-Gym/web/WebShop/web_agent_site/envs/clean_0_1200_seed233_t0.5_1run.json \
  --embeddings-path /n/home02/sverma/LMRL-Gym/JaxSEQ/JaxSeq/../web/WebShop/web_agent_site/envs/clean_gpt2_embedding_npz.npz \
  --lora-r 16 --lora-alpha 32 \
  --train-bsize 1 --max-length 512 --bf16 --gradient_checkpointing \
  --num-outer-iter 10 --num-inner-iter 100 \
  --outputs-path  /n/netscratch/tambe_lab/Lab/sverma/llm-rl/results --hf_token $HF_TOKEN



python llm_rl_scripts/webshop/mc/train_mc_bilevel_pytorch_llama3_qlora.py \
  /n/home02/sverma/LMRL-Gym/web/WebShop/web_agent_site/envs/clean_0_1200_seed233_t0.5_1run.json \
  /n/home02/sverma/LMRL-Gym/web/WebShop/web_agent_site/envs/clean_0_1200_seed233_t0.5_1run.json \
  --embeddings-path /n/home02/sverma/LMRL-Gym/JaxSEQ/JaxSeq/../web/WebShop/web_agent_site/envs/clean_gpt2_embedding_npz.npz \
  --train-bsize 1 --max-length 512 --bf16 --gradient_checkpointing \
  --num-outer-iter 10 --num-inner-iter 100 \
  --outputs-path  /n/netscratch/tambe_lab/Lab/sverma/llm-rl/results --hf_token $HF_TOKEN

end_time=$SECONDS
echo "Total execution time: $((end_time - start_time)) seconds"
