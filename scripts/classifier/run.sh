export OMP_NUM_THREADS=$(nproc)

# Train the classifiers
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/classifier/train.py -c classifier/bert.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/classifier/train.py -c classifier/llama.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/classifier/train.py -c classifier/llama_1b_balanced.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/classifier/train.py -c classifier/llama_3b_balanced_augmented.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/classifier/train.py -c classifier/llama_3b_balanced.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/classifier/train.py -c classifier/llama_3b.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/classifier/train.py -c classifier/smol_balanced.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/classifier/train.py -c classifier/smol.yaml

CUDA_VISIBLE_DEVICES=3 python scripts/classifier/train.py -c classifier/smol.yaml

# Evaluate the classifiers
CUDA_VISIBLE_DEVICES=3 python scripts/classifier/eval.py -m hugosousa/classifier_smoll_135m_b -d timeset
CUDA_VISIBLE_DEVICES=3 python scripts/classifier/eval.py -m hugosousa/classifier_llama_3b_balanced -d q_timelines


CUDA_VISIBLE_DEVICES=1 python scripts/classifier/eval.py -m hugosousa/classifier_smoll_135m -d q_timelines
CUDA_VISIBLE_DEVICES=1 python scripts/classifier/eval.py -m hugosousa/classifier_smoll_135m -d timeset

