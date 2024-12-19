mkdir -p project-root/{data/{input/{ideological,ambiguous,counterfactuals,neutral},runtime/{llm_output,labeled_data},processed},src,config}
touch project-root/requirements.txt
touch project-root/main.py
touch project-root/src/{generate_llm_output.py,labeling_gui.py,evaluation_metrics.py,visualize_results.py}
touch project-root/config/experiment_config.json 