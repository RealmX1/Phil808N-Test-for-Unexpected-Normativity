import argparse
import json
import logging
from pathlib import Path
import asyncio
from src.generate_llm_output import LLMGenerator
from src.labeling_gui import AnnotationGUI
from src.evaluation_metrics import Evaluator
from src.visualize_results import Visualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/experiment_config.json") -> dict:
    """Load experiment configuration from JSON file."""
    with open(config_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            return {
                'model': 'gpt-4o-mini',
                'input_folder': 'data/input',
                'output_folder': 'data/output',
                'labeled_data_folder': 'data/labeled',
            }

async def main():
    parser = argparse.ArgumentParser(description='Type 2 Normative Value Enforcement Test')
    parser.add_argument('--config', default='config/experiment_config.json',
                       help='Path to configuration file')
    parser.add_argument('--mode', choices=['generate', 'annotate', 'evaluate', 'visualize', 'all'],
                       default='annotate', help='Mode of operation')
    args = parser.parse_args()

    config = load_config(args.config)
    
    if args.mode in ['generate', 'all']:
        logger.info("Generating LLM outputs...")
        generator = await LLMGenerator.create(config)
        await generator.run()

    if args.mode in ['annotate', 'all']:
        logger.info("Starting annotation GUI...")
        gui = AnnotationGUI(config)
        gui.run()

    if args.mode in ['evaluate', 'all']:
        logger.info("Evaluating results...")
        evaluator = Evaluator(config)
        evaluator.compute_metrics()

    if args.mode in ['visualize', 'all']:
        logger.info("Generating visualizations...")
        visualizer = Visualizer(config)
        visualizer.create_plots()

if __name__ == "__main__":
    asyncio.run(main()) 