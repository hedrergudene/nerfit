# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# Model
class nerfitModel(nn.Module):
    def __init__(
            self,
            model_name:str,
            projection_dim:int,
            #temperature:float=1.,
            #eps:float=1e-5,
            lora_r:int=16,
            lora_alpha:int=32,
            lora_dropout:float=0.1,
            inference_mode:bool=False
    ):
        super(nerfitModel, self).__init__()
        # Load model
        self.base_model = get_peft_model(
            model=AutoModel.from_pretrained(model_name),
            peft_config=LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                inference_mode=inference_mode,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                use_dora=True
            )
        )
        # Last layer
        self.projection_layer = nn.Linear(self.base_model.base_model.config.hidden_dim, projection_dim)

    def forward(self, input_ids, attention_mask, embeddings=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state # Shape (batch_size, num_tokens, hidden_dim)

        token_embeddings_projected = self.projection_layer(last_hidden_states) # Shape (batch_size, num_tokens, projection_dim)
        token_embeddings_normalized = F.normalize(token_embeddings_projected, p=2, dim=-1)

        if ((labels is not None) & (embeddings is not None)):
            logits = torch.bmm(
                embeddings,
                token_embeddings_normalized.transpose(1,2)
            )
            return {
                'loss': F.binary_cross_entropy_with_logits(logits, labels),
                'logits': token_embeddings_normalized
            }
        else:
            return {
                'loss': None,
                'logits': token_embeddings_normalized
            }