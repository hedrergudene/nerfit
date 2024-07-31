import torch
from torch.utils.data import Dataset
from fuzzywuzzy import fuzz
from typing import List, Dict
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import re

# Helper methods
def preprocess_text(text:str):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def find_entity_positions(entity:str, text:str):
    matches = []
    pattern = re.compile(r'\b' + re.escape(entity) + r'\b', re.IGNORECASE)
    for match in pattern.finditer(text):
        matches.append((match.start(), match.end()))
    return matches

def fuzzy_find_entity(entity:str, text:str, threshold=90):
    tokens = text.split()
    for i in range(len(tokens)):
        window = ' '.join(tokens[i:i+len(entity.split())])
        if fuzz.ratio(entity.lower(), window.lower()) >= threshold:
            start_idx = text.lower().find(window.lower())
            end_idx = start_idx + len(window)
            return (start_idx, end_idx)
    return None

# Dataset
class nerfitDataset(Dataset):
    def __init__(
            self,
            annotations:Dict[str,Dict[str,str]],
            st_model:SentenceTransformer,
            encoder_tokenizer:AutoTokenizer,
            entity_descriptions:Dict[str,str]
    ):
        self.annotations = annotations
        self.st_model = st_model
        self.tokenizer = encoder_tokenizer
        self.entity_descriptions = entity_descriptions
        self.entity_embeddings = self._compute_entity_embeddings()

    def _compute_entity_embeddings(self):
        descriptions = list(self.entity_descriptions.values())
        with torch.no_grad():
            embeddings = self.st_model.encode(descriptions, convert_to_tensor=True, normalize_embeddings=True)
        return embeddings

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annot = self.annotations[idx]
        return self._collate_HuggingFace(annot)

    def _collate_HuggingFace(self, annotation):
        tokens = self.tokenizer.encode_plus(
            annotation['input'],
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        offset_mapping = tokens['offset_mapping'].squeeze().tolist()
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()

        labels = torch.zeros(len(self.entity_descriptions), len(input_ids), dtype=torch.float32)
        for ent in annotation['output']:
            entity, label = ent.split('<>')[0].strip(), ent.split('<>')[1].strip()
            if entity == '' or label == '':
                continue
            positions = find_entity_positions(entity, annotation['input'])
            if not positions:
                position = fuzzy_find_entity(entity, annotation['input'])
                if position:
                    positions = [position]
            if positions:
                for start, end in positions:
                    start_token_idx = end_token_idx = None
                    for idx, (token_start, token_end) in enumerate(offset_mapping):
                        if token_start <= start < token_end:
                            start_token_idx = idx
                        if token_start < end <= token_end:
                            end_token_idx = idx + 1
                            break
                    if start_token_idx is not None and end_token_idx is not None:
                        label_idx = list(self.entity_descriptions.keys()).index(label)
                        labels[label_idx, start_token_idx:end_token_idx] = 1.0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }