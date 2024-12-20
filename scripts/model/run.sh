export OMP_NUM_THREADS=$(nproc)

# Train the classifiers
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py -c classifier/bert.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py -c classifier/llama.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py -c classifier/llama_1b_balanced.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py -c classifier/llama_3b_balanced_augmented.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py -c classifier/llama_3b_balanced.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py -c classifier/llama_3b.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py -c classifier/smol_balanced.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py -c classifier/smol.yaml

# Evaluate the classifiers
python scripts/model/eval.py -m hugosousa/SmolLM-135M-TemporalQuestions -d temporal_questions
