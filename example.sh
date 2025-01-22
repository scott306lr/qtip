#!/usr/bin/env bash

###############################################################################
# Bash Script to Quantize Llama 2 (7B) to 2 bits, Convert to HF Format,
# and Evaluate Perplexity
###############################################################################

# Exit immediately if a command exits with a non-zero status,
# treat unset variables as errors, and fail if any command within a pipeline fails
set -euo pipefail

# Trap any error and provide a friendly message before exiting
trap "echo 'Script encountered an error. Exiting.'; exit 1" ERR

# Ensure python is available
command -v python >/dev/null 2>&1 || {
  echo "Error: Python is not installed or not found in PATH."
  exit 1
}

###############################################################################
# Environment Variables and Paths
###############################################################################

# Specify GPUs (edit as needed)
CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES

# Paths (edit as needed)
HF_BASE=meta-llama/Llama-2-7b-chat-hf
CKPT=~/qtip_run/checkpoints/llama/2_7b_2bit
CUSTOM_HF=~/qtip_run/hf/meta-llama/Llama-2-7b-chat-hf
HESS=~/qtip_run/hess/llama-2-7b-chat
LOG=~/qtip_run/logs/llama/2_7b_2bit

# Create necessary directories
mkdir -p "$CKPT" "$LOG" "$CUSTOM_HF"

###############################################################################
# Main Quantization
###############################################################################
echo "Starting main quantization script..."

python -m quantize_llama.quantize_finetune_llama \
       --save_path "$CKPT" \
       --codebook bitshift \
       --base_model "$HF_BASE" \
       --in_hess_path "$HESS" \
       --scale_override 0.9 \
       --ft_epochs 0 \
       --td_x 16 \
       --td_y 16 \
       --L 16 \
       --K 2 \
       --V 2 \
       --decode_mode quantlut_sym \
       --tlut_bits 9

###############################################################################
# Convert to HF Format
###############################################################################
echo "Converting quantized model to HF format..."

python -m quantize_llama.hfize_llama \
       --quantized_path "$CKPT" \
       --hf_output_path "$CUSTOM_HF"

###############################################################################
# Optional: End-to-End Finetuning
###############################################################################
# Uncomment if you want to do end-to-end finetuning.
#
# echo "Starting end-to-end finetuning..."
# python -m quantize_llama.finetune_e2e_llama \
#        --base_model "$HF_BASE" \
#        --hf_path "$CUSTOM_HF" \
#        --devset_size 640 \
#        --ft_valid_size 128 \
#        --ft_epochs 4 \
#        --ft_update_freq 4 \
#        --ft_bs 2 \
#        --ctx_size 4096 \
#        --ft_train_lut \
#        --hf_output_path "$CUSTOM_HF"

###############################################################################
# Evaluation
###############################################################################
echo "Evaluating perplexity..."

python -m eval.eval_ppl --hf_path "$CUSTOM_HF"

# Uncomment to run zero-shot evaluation on common tasks:
# echo "Evaluating zero-shot performance..."
# python -m eval.eval_zeroshot \
#   --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
#   --batch_size 16 \
#   --hf_path "$CUSTOM_HF"

echo "Script completed successfully."