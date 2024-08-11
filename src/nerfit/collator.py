# Libraries
import torch

# Main class
class nerfitDataCollator:
    def __init__(self, pad_token_id, max_length=512):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch):
        # Extract the individual components from the batch
        input_ids_batch = [item['input_ids'] for item in batch]
        attention_mask_batch = [item['attention_mask'] for item in batch]
        labels_batch = [item['labels'] for item in batch]
        embeddings_batch = [item['embeddings'] for item in batch]

        # Pad input_ids and attention_mask
        input_ids_padded = self._pad_sequence(input_ids_batch, self.pad_token_id)
        attention_mask_padded = self._pad_sequence(attention_mask_batch, 0)

        # Determine the padding for labels, considering cases with zero entities
        if labels_batch[0].numel() > 0:
            num_entities = labels_batch[0].size(0)
        else:
            num_entities = 0

        max_labels_len = max((item['labels'].size(1) if item['labels'].numel() > 0 else 0) for item in batch)
        labels_padded = torch.full((len(batch), num_entities, max_labels_len), -100, dtype=torch.long)

        for i, labels in enumerate(labels_batch):
            if labels.numel() > 0:
                labels_padded[i, :, :labels.size(1)] = labels

        # Determine the padding for embeddings, considering cases with zero entities
        if embeddings_batch[0].numel() > 0:
            embed_dim = embeddings_batch[0].size(1)
            max_num_entities = max(emb.size(0) for emb in embeddings_batch)
        else:
            embed_dim = 0
            max_num_entities = 0

        embeddings_padded = torch.zeros((len(batch), max_num_entities, embed_dim), dtype=torch.float32)

        for i, embeddings in enumerate(embeddings_batch):
            if embeddings.numel() > 0:
                embeddings_padded[i, :embeddings.size(0), :] = embeddings

        return {
            'input_ids': input_ids_padded,               # Shape (batch_size, max_num_tokens)
            'attention_mask': attention_mask_padded,     # Shape (batch_size, max_num_tokens)
            'labels': labels_padded,                     # Shape (batch_size, num_entities, max_num_tokens)
            'embeddings': embeddings_padded              # Shape (batch_size, max_num_entities, embed_dim)
        }

    def _pad_sequence(self, sequences, pad_value):
        max_length = min(max([len(seq) for seq in sequences]), self.max_length)
        padded_sequences = torch.full((len(sequences), max_length), pad_value, dtype=torch.long)

        for i, seq in enumerate(sequences):
            length = min(len(seq), max_length)
            padded_sequences[i, :length] = seq[:length].clone().detach()

        return padded_sequences