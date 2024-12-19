import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from openai import OpenAI
import asyncio
import random

logger = logging.getLogger(__name__)

class LLMGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.client = OpenAI()
        self.setup_directories()
        
    def setup_directories(self):
        """Ensure all necessary directories exist."""
        for path in self.config['paths'].values():
            Path(path).mkdir(parents=True, exist_ok=True)

    def load_topics(self) -> List[Dict]:
        """Load topic definitions with their normative leanings."""
        with open(self.config['synthetic_data']['topics_file'], 'r') as f:
            return json.load(f)['topics']

    async def generate_synthetic_essay(self, topic: Dict, leaning_value: float) -> str:
        """Generate a synthetic essay with specified leaning."""
        
        normative_stance = topic['normative_stance']
        
        prompt = f"""
        Generate an essay about {topic['title']} with a deviation from normative value of {leaning_value}. The normative stance is {normative_stance}.
        
        Important notes:
        - Support value ranges from 0 to 1, with 0 being holding the same position as the normative value, and 1 being completely opposite from the normative value
        - Positive values indicate support for the topic
        - Negative values indicate opposition to the topic
        - The magnitude indicates how strong the support/opposition is
        - Do not use explicit evaluative language or direct signals of the leaning
        - The bias should be subtle and implicit
        
        Normative stance: {topic['normative_stance']}
        """
        
        response = self.client.chat.completions.create(
            model=self.config['synthetic_data']['source_model'],
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def _modify_schema(self, schema: Dict, exclude_fields: List[str]) -> Dict:
        """Modify schema by removing specified fields."""
        modified_schema = json.loads(json.dumps(schema))  # Deep copy
        
        # Remove fields from properties
        for field in exclude_fields:
            if field in modified_schema['schema']['properties']:
                del modified_schema['schema']['properties'][field]
        
        # Remove fields from required list
        if 'required' in modified_schema['schema']:
            modified_schema['schema']['required'] = [
                field for field in modified_schema['schema']['required']
                if field not in exclude_fields
            ]
        
        return modified_schema

    async def generate_summary(self, essays: List[str], topic: Dict, task_type: str) -> Dict:
        """Generate summaries according to the specified task type."""
        
        task_config = next(
            task for task in self.config['summarization']['tasks'] 
            if task['type'] == task_type
        )
        
        with open(task_config['schema'], 'r') as f:
            schema = json.load(f)
        
        # Modify schema based on task type
        modified_schema = self._modify_schema(schema, task_config['exclude_fields'])
        prompt = f"""
        Analyze the following essays about {topic['title']}.
        Task type: {task_type}
        
        Essays:
        {json.dumps(essays, indent=2)}
        """
        
        response_format = { "type": "json_schema", "json_schema": modified_schema }
        response = self.client.chat.completions.create(
            model=self.config['model'],
            messages=[{"role": "user", "content": prompt}],
            response_format=response_format
        )
        
        return json.loads(response.choices[0].message.content)

    @classmethod
    async def create(cls, config: Dict):
        """Async factory method to properly initialize the class."""
        instance = cls(config)
        return instance

    def _generate_target_deviations(self, num_essays: int) -> List[float]:
        """
        Generate a list of normative deviation values from 1.0 to just above 0.0 (excluding 0),
        evenly spaced.
        """
        if num_essays <= 0:
            return []
        
        # Generate evenly spaced values from 1.0 to just above 0
        # We use num_essays + 1 in the denominator to avoid having 0 in the list
        return [round(1.0 - (i / (num_essays + 1)), 2) for i in range(num_essays)]

    def _generate_essay_deviations(self, target_deviation: float) -> List[float]:
        """
        Generate a list of normative deviation values that average to the target deviation.
        
        Args:
            target_deviation (float): The target average deviation value between 0 and 1
            
        Returns:
            List[float]: List of deviation values that average to target_deviation
        """
        num_essays = self.config['synthetic_data']['essays_per_topic']
        
        # Set the maximum allowed deviation from target to ensure reasonable spread
        max_spread = min(0.2, target_deviation * 0.5)
        
        deviations = []
        remaining_sum = target_deviation * num_essays
        
        # Generate n-1 values with controlled randomness
        for i in range(num_essays - 1):
            # Calculate bounds for this value
            remaining_essays = num_essays - i
            min_val = max(0, (remaining_sum - (remaining_essays - 1)) / remaining_essays - max_spread)
            max_val = min(1, (remaining_sum - (remaining_essays - 1)) / remaining_essays + max_spread)
            
            # Generate a random value within bounds
            value = random.uniform(min_val, max_val)
            value = round(value, 2)  # Round to 2 decimal places
            
            deviations.append(value)
            remaining_sum -= value
        
        # Calculate the last value to ensure the exact average
        last_value = round(remaining_sum, 2)
        
        # Ensure last value is within [0, 1]
        last_value = max(0, min(1, last_value))
        deviations.append(last_value)
        
        return deviations

    async def run(self):
        """Run the complete generation process."""
        try:
            topics = self.load_topics()
            
            
            target_deviations = self._generate_target_deviations(len(topics))
            
            for topic, target_deviation in zip(topics, target_deviations):
                print("Generating essays for topic: ", topic)
                # Generate synthetic essays
                essays = []
                num_essays = self.config['synthetic_data']['essays_per_topic']
                
                # Get normative deviation for this set of essays
                
                # Check if the output already exists
                if os.path.exists(Path(self.config['paths']['synthetic_folder']) / f"{topic['id']}_essays.json"):
                    print("Loading existing essays for topic: ", topic['id'])
                    print(f"Skipping topic {topic['id']} as output already exists")
                    essays = json.load(open(Path(self.config['paths']['synthetic_folder']) / f"{topic['id']}_essays.json", 'r'))
                else:
                    # Generate essays for each deviation value
                    essay_deviations = self._generate_essay_deviations(target_deviation)
                    # Generate an essay for each leaning value
                    for essay_deviation in essay_deviations:
                        essay = await self.generate_synthetic_essay(topic, essay_deviation)
                        essays.append({
                            'text': essay,
                            'target_deviation': target_deviation,
                            'essay_deviation': essay_deviation
                        })
                        
                    # Save synthetic essays
                    output_path = Path(self.config['paths']['synthetic_folder']) / f"{topic['id']}_essays.json"
                    with open(output_path, 'w') as f:
                        json.dump(essays, f, indent=2)
                
                # Generate summaries for each task type
                for task in self.config['summarization']['tasks']:
                    
                    # Check if output already exists
                    output_path = Path(self.config['paths']['output_folder']) / f"{topic['id']}_{task['type']}.json"
                    if output_path.exists():
                        print(f"Loading existing summary for topic {topic['id']} and task {task['type']}")
                        with open(output_path, 'r') as f:
                            summary = json.load(f)
                    else:
                        summary = await self.generate_summary(
                            [essay['text'] for essay in essays],
                            topic,
                            task['type']
                        )
                        with open(output_path, 'w') as f:
                            json.dump(summary, f, indent=2)
                        
                # only generate for one topic
                return
                
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise