# Train the classifiers
accelerate launh scripts/classifier/train.py -c classifier/bert.yaml
accelerate launh scripts/classifier/train.py -c classifier/llama.yaml
accelerate launh scripts/classifier/train.py -c classifier/llama_balanced.yaml
