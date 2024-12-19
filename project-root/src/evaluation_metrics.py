import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, config: Dict):
        self.config = config
        # self.setup_directories()
        self.setup_model()
        
    def setup_directories(self):
        """Ensure necessary directories exist."""
        # Path(self.config['processed_data_folder']).mkdir(parents=True, exist_ok=True)
        pass

    def setup_model(self):
        """Initialize the BERT model and tokenizer for semantic similarity."""
        model_name = 'bert-base-uncased'  # Can be configured in config file
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on token embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embeddings for a text using BERT."""
        # Tokenize and prepare input
        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Move back to CPU and convert to numpy
        return sentence_embeddings.cpu().numpy()

    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts using BERT embeddings."""
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        
        return cosine_similarity(embedding1, embedding2)[0][0]

    def load_annotations(self) -> List[Dict]:
        """Load all annotation data."""
        annotations = []
        labeled_dir = Path(self.config['labeled_data_folder'])
        
        for annotation_file in labeled_dir.glob('*_annotation.json'):
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotations.append(json.load(f))
        
        return annotations

    def compute_metrics(self):
        """Compute all evaluation metrics."""
        annotations = self.load_annotations()
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(annotations)
        
        metrics = {
            'total_samples': len(annotations),
            'enforcement_distribution': self._compute_enforcement_distribution(df),
            'category_metrics': self._compute_category_metrics(df),
            'semantic_shifts': self._compute_semantic_shifts(df),
            'bias_analysis': self._compute_bias_analysis(df)
        }
        
        # Save metrics
        output_file = Path('data') / 'output' / 'evaluation_metrics.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Computed and saved evaluation metrics")
        return metrics

    def _compute_enforcement_distribution(self, df: pd.DataFrame) -> Dict:
        """Compute distribution of enforcement types."""
        distribution = df['enforcement_type'].value_counts().to_dict()
        percentages = (df['enforcement_type'].value_counts(normalize=True) * 100).to_dict()
        
        return {
            'counts': distribution,
            'percentages': percentages
        }

    def _compute_category_metrics(self, df: pd.DataFrame) -> Dict:
        """Compute metrics broken down by input category."""
        metrics = {}
        
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            metrics[category] = {
                'total_samples': len(category_df),
                'enforcement_distribution': category_df['enforcement_type'].value_counts().to_dict(),
                'enforcement_percentages': (category_df['enforcement_type'].value_counts(normalize=True) * 100).to_dict()
            }
        
        return metrics

    def _compute_semantic_shifts(self, df: pd.DataFrame) -> Dict:
        """Compute semantic similarity metrics between inputs and outputs."""
        similarities = []
        
        for _, row in df.iterrows():
            # Load original input-output pair
            output_file = Path(self.config['output_folder']) / f"{row['input_id']}_{row['prompt_type']}.json"
            with open(output_file, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
            
            similarity = self.compute_semantic_similarity(
                output_data['input_text'],
                output_data['output_text']
            )
            similarities.append({
                'input_id': row['input_id'],
                'similarity': float(similarity),  # Convert to float for JSON serialization
                'enforcement_type': row['enforcement_type']
            })
        
        # Convert to DataFrame for analysis
        sim_df = pd.DataFrame(similarities)
        
        return {
            'average_similarity': float(sim_df['similarity'].mean()),
            'similarity_by_enforcement': {k: float(v) for k, v in 
                                       sim_df.groupby('enforcement_type')['similarity'].mean().to_dict().items()},
            'similarity_std': float(sim_df['similarity'].std())
        }

    def _compute_bias_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze bias in enforcement patterns."""
        # Look for asymmetry in enforcement between counterfactual pairs
        counterfactual_pairs = self._identify_counterfactual_pairs(df)
        
        bias_metrics = {
            'counterfactual_asymmetry': self._analyze_counterfactual_asymmetry(counterfactual_pairs),
            'prompt_type_bias': df.groupby(['prompt_type', 'enforcement_type']).size().unstack(fill_value=0).to_dict()
        }
        
        return bias_metrics

    def _identify_counterfactual_pairs(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Identify pairs of counterfactual inputs."""
        # This is a placeholder - actual implementation would need logic to match counterfactual pairs
        # based on your specific naming convention or metadata
        return []

    def _analyze_counterfactual_asymmetry(self, pairs: List[Tuple[str, str]]) -> Dict:
        """Analyze enforcement asymmetry between counterfactual pairs."""
        # Placeholder for counterfactual analysis
        return {
            'asymmetry_detected': False,
            'details': "Counterfactual analysis not implemented"
        }
