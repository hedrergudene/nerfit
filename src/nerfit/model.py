# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# Model
class nerfitModel(nn.Module):
    def __init__(self, model_name, projection_dim, entity_embeddings):
        super(nerfitModel, self).__init__()
        self.bert_model = AutoModel.from_pretrained(model_name)
        hidden_dim = self.bert_model.config.hidden_size
        self.projection_layer = nn.Linear(hidden_dim, projection_dim)
        self.entity_embeddings = F.normalize(nn.Parameter(entity_embeddings, requires_grad=False), p=2, dim=-1) # Shape (num_entities, projection_dim)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state # Shape (batch_size, num_tokens, hidden_dim)

        token_embeddings_projected = self.projection_layer(last_hidden_states) # Shape (batch_size, num_tokens, projection_dim)
        token_embeddings_normalized = F.normalize(token_embeddings_projected, p=2, dim=-1)

        if labels is not None:
            return self._compute_contrastive_loss(token_embeddings_normalized, labels)
        else:
            return token_embeddings_normalized

    def _compute_contrastive_loss(self, token_embeddings_normalized, labels):
        pos_mask = labels == 1
        neg_mask = labels == 0

        #TODO: Replace self.entity_embeddings 
        cosine_sim = torch.einsum('btd,cd->btc', token_embeddings_normalized, self.entity_embeddings)
        
        positive_scores = cosine_sim[pos_mask]
        negative_scores = cosine_sim[neg_mask]
        
        if positive_scores.size(0) == 0 or negative_scores.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True).to(cosine_sim.device)

        positive_scores = positive_scores.view(-1, 1)
        negative_scores = negative_scores.view(-1, 1)

        min_size = min(positive_scores.size(0), negative_scores.size(0))
        positive_scores = positive_scores[:min_size]
        negative_scores = negative_scores[:min_size]

        logits = torch.cat([positive_scores, negative_scores], dim=1)

        targets = torch.zeros(min_size, dtype=torch.long).to(cosine_sim.device)
        loss = F.cross_entropy(logits, targets)

        return loss
