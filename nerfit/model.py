# Libraries
from typing import Optional, List, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import PeftModel, LoraConfig, TaskType

# nerfit
class nerfitModel(nn.Module):
    def __init__(
            self,
            model_name:str,
            projection_dim:int,
            #temperature:float=1.,
            #eps:float=1e-5,
            peft_lora:bool=False,
            peft_config:Optional[Dict[str,Union[int,float,bool]]]=None
    ):
        super(nerfitModel, self).__init__()
        # Load model
        if ((peft_lora) & (peft_config is None)):
            raise ValueError(f"You must specify LoRA parameters.")
        elif peft_config is not None:
            self.base_model = PeftModel(
                model=AutoModel.from_pretrained(model_name),
                peft_config=LoraConfig(
                    inference_mode=peft_config['inference_mode'],
                    r=peft_config['lora_r'],
                    lora_alpha=peft_config['lora_alpha'],
                    lora_dropout=peft_config['lora_dropout'],
                    use_dora=peft_config['use_dora']
                )
            )
        else:
            self.base_model = AutoModel.from_pretrained(model_name)
        # Last layer
        self.projection_layer = nn.Linear(self.base_model.base_model.config.hidden_size, projection_dim)

    def forward(self, input_ids, attention_mask, labels=None, embeddings=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state # Shape (batch_size, num_tokens, hidden_dim)

        token_embeddings_projected = self.projection_layer(last_hidden_states) # Shape (batch_size, num_tokens, projection_dim)
        token_embeddings_normalized = F.normalize(token_embeddings_projected, p=2, dim=-1)

        if ((labels is not None) & (embeddings is not None)):
            mask = (labels != -100)

            if embeddings.size(1) > 0:
                logits = torch.bmm(embeddings, token_embeddings_normalized.transpose(1, 2))
                logits_mask = logits * mask
                labels_mask = labels * mask
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_mask, labels_mask, reduction='sum')
                loss = loss / mask.sum()
            else:
                loss = torch.tensor(0.0, device=labels.device)
            
            return {'loss': loss, 'logits': token_embeddings_normalized}
        else:
            return {'loss': None, 'logits': token_embeddings_normalized}