python scripts/agents/test.py --agent_name before 
python scripts/agents/test.py --agent_name random 
python scripts/agents/test.py --agent_name mcts --num_simulations 500
python scripts/agents/test.py --agent_name lm --model_name "meta-llama/Llama-3.1-8B-Instruct"
python scripts/agents/test.py --agent_name trained --model_name "hugosousa/classifier_llama_1b"
python scripts/agents/test.py --agent_name trained --model_name "hugosousa/classifier_llama_1b_balanced"

python scripts/agents/print_results.py 