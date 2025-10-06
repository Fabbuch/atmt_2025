#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=4:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=translate_sk-en.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit


# TRANSLATE
python translate.py \
    --cuda \
    --input data_sk-en/DGT.en-sk_trim.sk \
    --src-tokenizer cz-en/tokenizers/cs-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --output sk-en_output.txt \
    --max-len 300
