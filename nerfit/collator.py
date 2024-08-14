# Libraries
import torch
from typing import Dict, Union

# Main class
class nerfitDataCollator:
    def __init__(self, pad_token_id:int, max_length:int, projection_dim:int):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.projection_dim = projection_dim

    def __call__(self, batch:Dict[str,Union[int,torch.Tensor]]):
        # Extract the individual components from the batch
        input_ids_batch = [item['input_ids'] for item in batch]
        attention_mask_batch = [item['attention_mask'] for item in batch]
        labels_batch = [item['label'] for item in batch]
        embeddings_batch = [item['embedding'] for item in batch]

        # Pad input_ids and attention_mask
        input_ids_padded = self._pad_sequence(input_ids_batch, self.pad_token_id)
        attention_mask_padded = self._pad_sequence(attention_mask_batch, 0)

        # Determine the maximum number of entities and sequence length in the batch
        max_num_entities = max(emb.size(0) if emb.numel() > 0 else 0 for emb in embeddings_batch)
        max_labels_len = max(item.size(1) if item.numel() > 0 else 0 for item in labels_batch)

        # Initialize the labels tensor with -100 for padding
        labels_padded = torch.full((len(batch), max_num_entities, max_labels_len), -100, dtype=torch.float32)
        embeddings_padded = torch.zeros((len(batch), max_num_entities, self.projection_dim), dtype=torch.float32)

        for i, (labels, embeddings) in enumerate(zip(labels_batch, embeddings_batch)):
            if labels.numel() > 0:
                num_entities = labels.size(0)
                seq_length = labels.size(1)
                labels_padded[i, :num_entities, :seq_length] = labels

            if embeddings.numel() > 0:
                num_entities = embeddings.size(0)
                embeddings_padded[i, :num_entities, :] = embeddings

        return {
            'input_ids': input_ids_padded,               # Shape (batch_size, max_num_tokens)
            'attention_mask': attention_mask_padded,     # Shape (batch_size, max_num_tokens)
            'label': labels_padded,                     # Shape (batch_size, max_num_entities, max_num_tokens)
            'embedding': embeddings_padded              # Shape (batch_size, max_num_entities, projection_dim)
        }

    def _pad_sequence(self, sequences, pad_value):
        max_length = min(max([len(seq) for seq in sequences]), self.max_length)
        padded_sequences = torch.full((len(sequences), max_length), pad_value, dtype=torch.long)

        for i, seq in enumerate(sequences):
            length = min(len(seq), max_length)
            padded_sequences[i, :length] = seq[:length].clone().detach()

        return padded_sequences