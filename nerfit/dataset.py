# Libraries
import torch
from torch.utils.data import Dataset
from typing import List, Dict
from transformers import AutoTokenizer
import re

# Dataset
class nerfitDataset(Dataset):
    def __init__(
            self,
            annotations:Dict[str,Dict[str,str]],
            ent2emb:Dict[str,torch.Tensor],
            encoder_tokenizer:AutoTokenizer
    ):
        self.annotations = annotations
        self.tokenizer = encoder_tokenizer
        self.ent2emb = ent2emb
        self.label2id = {label: idx for idx, label in enumerate(self.ent2emb.keys())}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annot = self.annotations[idx]
        return self._collate_pretraining(annot)

    def _parse_annotation(self, annotation: str):
        pattern = re.compile(r'\[(.*?)\]\((.*?): (.*?)\)')
        matches = pattern.finditer(annotation)

        text = annotation
        entities = []
        offset = 0

        for m in matches:
            entity = m.group(1).strip()
            start_idx = m.start() - offset
            end_idx = start_idx + len(entity)

            entities.append([start_idx, end_idx, entity])

            # Replace the annotated part with the entity name in the text
            annotated_text = m.group(0)
            text = text[:m.start()-offset] + entity + text[m.end()-offset:]

            # Update the offset to account for the removed annotation
            offset += len(annotated_text) - len(entity)

        return {
            "text": text,
            "entities": entities
        }

    def _collate_pretraining(self, annot):
        # Tokenize
        tokens = self.tokenizer.encode_plus(
            annot['text'],
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        offset_mapping = tokens['offset_mapping'].squeeze().tolist()
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()

        embeddings = []
        labels = []
        for ent in annot['entities']:
            #
            start_token_idx = end_token_idx = None
            for idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start <= ent[0] < token_end:
                    start_token_idx = idx
                if token_start < ent[1] <= token_end:
                    end_token_idx = idx + 1
                    break
            embeddings.append(self.ent2emb[ent[2]].unsqueeze(0))
            lab = torch.tensor([1 if idx in list(range(start_token_idx, end_token_idx)) else 0 for idx in range(len(input_ids))], dtype=torch.int32)
            labels.append(lab.unsqueeze(0))  # Add an extra dimension for concatenation

        return {
            'input_ids': input_ids,                                                         # Shape (num_tokens)
            'attention_mask': attention_mask,                                               # Shape (num_tokens)
            'embeddings': torch.cat(embeddings, dim=0) if embeddings else torch.tensor([]), # Shape (num_entities, embed_dim)
            'labels': torch.cat(labels, dim=0) if labels else torch.tensor([])              # Shape (num_entities, num_tokens)
        }