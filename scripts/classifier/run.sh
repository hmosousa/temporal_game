# Train the classifiers
accelerate launch scripts/classifier/train.py -c classifier/bert.yaml
accelerate launch scripts/classifier/train.py -c classifier/llama.yaml
accelerate launch scripts/classifier/train.py -c classifier/llama_1b_balanced.yaml
accelerate launch scripts/classifier/train.py -c classifier/llama_3b_balanced.yaml
