# Libraries
import torch

# Data collator
class nerfitDataCollator:
    def __init__(self, pad_token_id, max_length=512):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch):
        input_ids_batch = [item['input_ids'] for item in batch]
        attention_mask_batch = [item['attention_mask'] for item in batch]
        labels_batch = [item['labels'] for item in batch]

        input_ids_padded = self._pad_sequence(input_ids_batch, self.pad_token_id)
        attention_mask_padded = self._pad_sequence(attention_mask_batch, 0)

        max_labels_len = max(len(item['labels']) for item in batch)
        labels_padded = torch.zeros((len(batch), max_labels_len, self.max_length), dtype=torch.float32)

        for i, labels in enumerate(labels_batch):
            labels_padded[i, :, :labels.size(1)] = labels

        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask_padded,
            'labels': labels_padded
        }

    def _pad_sequence(self, sequences, pad_value):
        max_length = min(max([len(seq) for seq in sequences]), self.max_length)
        padded_sequences = torch.full((len(sequences), max_length), pad_value, dtype=torch.long)

        for i, seq in enumerate(sequences):
            length = min(len(seq), max_length)
            padded_sequences[i, :length] = torch.tensor(seq[:length], dtype=torch.long).clone().detach()

        return padded_sequences
