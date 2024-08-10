# Libraries
import torch
from torch.utils.data import Dataset
from typing import List, Dict
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import re

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

    def _parse_annotation(annotation: str):
        pattern = re.compile(r'\[(.*?)\]\((.*?): (.*?)\)')
        matches = pattern.finditer(annotation)

        text = annotation
        entities = []
        offset = 0

        for m in matches:
            entity = m.group(1)
            label = m.group(2)
            description = m.group(3)
            start_idx = m.start() - offset
            end_idx = start_idx + len(entity)

            entities.append({
                "start": start_idx,
                "end": end_idx,
                "entity": entity,
                "label": label,
                "description": description
            })

            # Replace the annotated part with the entity name in the text
            annotated_text = m.group(0)
            text = text[:m.start()-offset] + entity + text[m.end()-offset:]

            # Update the offset to account for the removed annotation
            offset += len(annotated_text) - len(entity)

        return {
            "text": text,
            "entities": entities
        }            

    def _collate_pretraining(self, annotation):
        # Parse text
        annot = self._parse_annotation(annotation)
        tokens = self.tokenizer.encode_plus(
            annot['text'],
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        offset_mapping = tokens['offset_mapping'].squeeze().tolist()
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()

        labels = torch.zeros(len(self.entity_descriptions), len(input_ids), dtype=torch.float32)
        for ent in annotation['output']:
            # 
            start_token_idx = end_token_idx = None
            for idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start <= ent['start'] < token_end:
                    start_token_idx = idx
                if token_start < ent['end'] <= token_end:
                    end_token_idx = idx + 1
                    break
            label_idx = list(self.entity_descriptions.keys()).index(label)
            labels[label_idx, start_token_idx:end_token_idx] = 1.0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def _collate_NER(self, annotation):
        # Tokenize text
        annot = self._parse_annotation(annotation)
        tokens = self.tokenizer.encode_plus(
            annot['text'],
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt',
        )

        # Create array to store class labels for tokens
        targets = []
        ents = sorted(annot['entities'], key=lambda x: x[0])  # sort entities by start position
        ent_idx = 0
        num_ents = len(ents)

        # Process each token
        for c, d in torch.squeeze(tokens['offset_mapping']):
            # Special token
            if c == 0 and d == 0:
                targets.append(-100)  # Append label for special tokens
                continue

            # Manage entity indices
            while ent_idx < num_ents and ents[ent_idx][1] < c:
                ent_idx += 1  # Move past entities that end before this token starts

            # Check if current token is within any entity
            hit = False
            if ent_idx < num_ents and ents[ent_idx][0] <= c < ents[ent_idx][1]:
                label = ents[ent_idx][2]
                # Check if it's the start of an entity
                if c == ents[ent_idx][0]:
                    targets.append(self.tag2idx['B-' + label])
                else:
                    targets.append(self.tag2idx['I-' + label])
                hit = True
            # If no entity matches, mark as O (Outside any entity)
            if not hit:
                targets.append(self.tag2idx['O'])

        return {
            'input_ids': torch.squeeze(tokens['input_ids']),
            'attention_mask': torch.squeeze(tokens['attention_mask']),
            'token_type_ids': torch.squeeze(tokens.get('token_type_ids', torch.zeros_like(tokens['input_ids']))),  # safe handling in case token_type_ids are not returned
            'labels': torch.LongTensor(targets)
        }

    def display_tokens_and_labels(self, idx):
        # Get item data
        item = self.__getitem__(idx)

        # Decode tokens to text
        tokens = self.tokenizer.convert_ids_to_tokens(item['input_ids'])

        # Retrieve labels
        labels = item['labels'].tolist()

        # Reverse map from indices to labels (for displaying purposes)
        idx2tag = {v: k for k, v in self.tag2idx.items()}

        # Prepare display format
        token_label_pairs = [(token, idx2tag.get(label, 'O')) for token, label in zip(tokens, labels)]

        # Print or return the formatted token-label pairs
        for token, label in token_label_pairs:
            print(f"{token} [{label}]")