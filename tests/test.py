# Libraries
import unittest
import torch
import json
from transformers import AutoTokenizer
import os
import re
from pathlib import Path
from safetensors.torch import safe_open
from typing import List, Union, Dict
from ..nerfit.trainer import Trainer, TrainerConfig


# Testing trainer
class TestTrainer(Trainer):
    def _parse_annotation(
        annotations:List[
            Union[
                Dict[str,str],
                Dict[str,List[List[str]]],
                Dict[str,List[Dict[str,Union[int,str]]]],
                str
            ]
        ]
    ) -> List[Dict[str,Union[str,int]]]:
        output = []
        for annotation in annotations:
            pattern = re.compile(r'\[(.*?): (.*?)\]')
            matches = pattern.finditer(annotation)
            text = annotation
            entities = []
            offset = 0
            for m in matches:
                entity = m.group(2).strip()
                label = m.group(1).strip()
                start_idx = m.start() - offset
                end_idx = start_idx + len(entity)
                entities.append([start_idx, end_idx, label])
                # Replace the annotated part with the entity name in the text
                annotated_text = m.group(0)
                text = text[:m.start()-offset] + entity + text[m.end()  -offset:]
                # Update the offset to account for the removed annotation
                offset += len(annotated_text) - len(entity)
            output.append({"text": text,"entities": entities})
        return output
    
    def _prepare_model(self) -> torch.nn.Module:
        """
        Prepares a simple model with an embedding layer and a dense layer.
        """
        class SimpleNERModel(torch.nn.Module):
            def __init__(self, embedding_dim: int, num_labels: int):
                super(SimpleNERModel, self).__init__()
                self.embeddings = torch.nn.EmbeddingBag(embedding_dim, embedding_dim)
                self.projection_layer = torch.nn.Linear(embedding_dim, num_labels)
            
            def forward(self, input_ids, attention_mask=None):
                x = self.embeddings(input_ids)
                x = self.projection_layer(x)
                return x
        
        embedding_dim = next(iter(self.ent2emb.values())).shape[0]  # Get embedding dimension
        num_labels = len(self.ent2emb)
        model = SimpleNERModel(embedding_dim, num_labels)
        
        return model
    
    def _prepare_tokenizer(self) -> AutoTokenizer:
        """
        Use the RobertaTokenizer from a typical 'roberta-base' checkpoint.
        """
        return AutoTokenizer.from_pretrained("./tokenizer")


# Unit test
class TestTrainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load sample data
        with open('./sample_data.txt', 'r', encoding='utf-8') as f:
            annotations = f.read().split('\n')
        # Load lookup table
        ent2emb = safe_open('./artifacts/ent2emb.safetensors')

        # Initialize tokenizer and config
        config = TrainerConfig(
            model_name='roberta-base',
            train_annotations=annotations,
            val_annotations=annotations,
            ent2emb=ent2emb,
            num_steps=10,  # Reduced steps for testing
            callback_steps=5,
            save_steps=5,
            batch_size=2,
            output_dir='./model_test'
        )

        # Initialize TextTrainer
        cls.trainer = TestTrainer(config)
    
    def test_fit(self):
        # Run the fit method
        self.trainer.fit()
        
        # Check if model checkpoints are saved
        output_dir = Path(self.trainer.config.output_dir)
        self.assertTrue(output_dir.exists(), "Output directory does not exist")
        checkpoints = list(output_dir.glob('checkpoint-*'))
        self.assertGreater(len(checkpoints), 0, "No checkpoints saved during training")

    @classmethod
    def tearDownClass(cls):
        # Clean up the output directory
        if os.path.exists(cls.trainer.config.output_dir):
            os.system(f'rm -r {cls.trainer.config.output_dir}')

if __name__ == '__main__':
    unittest.main()
