# Libraries
import re
import itertools
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

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
        self.label2id = {
            **{label: idx+1 for idx, label in enumerate([x[1]+x[0] for x in itertools.product(self.ent2emb.keys(),['B-','I-'])])},
            **{'O':0}
        }
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Fetch sample
        annot = self.annotations[idx]

        #
        # Tokenize
        #
        tokens = self.tokenizer.encode_plus(
            annot['text'],
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        offset_mapping = tokens['offset_mapping'].squeeze().tolist()
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()

        #
        # Pretraining labels
        #
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

        #
        # NER labels
        #
        # Create array to store class labels for tokens
        targets = []
        ents = sorted(annot['entities'], key=lambda x: x[0])  # sort entities by start position
        ent_idx = 0
        num_ents = len(ents)

        # Process each token
        for c, d in offset_mapping:
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
                    targets.append(self.label2id['B-' + label])
                else:
                    targets.append(self.label2id['I-' + label])
                hit = True
            # If no entity matches, mark as O (Outside any entity)
            if not hit:
                targets.append(self.label2id['O'])

        return {
            'input_ids': input_ids,                                                         # Shape (num_tokens)
            'attention_mask': attention_mask,                                               # Shape (num_tokens)
            'embeddings': torch.cat(embeddings, dim=0) if embeddings else torch.tensor([]), # Shape (num_entities, embed_dim)
            'labels_pretraining': torch.cat(labels, dim=0) if labels else torch.tensor([]), # Shape (num_entities, num_tokens)
            'labels_ner': torch.LongTensor(targets)                                         # Shape (num_tokens)
        }

    def display_tokens_and_labels(self, idx):
        # Get item data
        item = self.__getitem__(idx)
        
        # Decode tokens to text
        tokens = self.tokenizer.convert_ids_to_tokens(item['input_ids'])
        
        # Retrieve labels
        labels = item['labels_ner'].tolist()
        
        # Prepare display format
        token_label_pairs = [(token, self.id2label.get(label, 'O')) for token, label in zip(tokens, labels)]
        
        # Print or return the formatted token-label pairs
        for token, label in token_label_pairs:
            print(f"{token} [{label}]")