CUDA_VISIBLE_DEVICES=0 python scripts/classifier/train.py -c classifier/llama_3b_balanced_augmented.yaml
CUDA_VISIBLE_DEVICES=1 python scripts/classifier/train.py -c classifier/llama_3b_balanced_augmented.yaml
CUDA_VISIBLE_DEVICES=2 python scripts/classifier/train.py -c classifier/llama_3b_balanced_augmented.yaml
CUDA_VISIBLE_DEVICES=3 python scripts/classifier/train.py -c classifier/llama_3b_balanced_augmented.yaml

CUDA_VISIBLE_DEVICES=3 python scripts/classifier/train.py -c classifier/smol_balanced_augmented.yaml