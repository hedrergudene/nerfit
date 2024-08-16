# Libraries
import torch
import evaluate
from typing import Optional, List, Callable, Dict, Any, Union


# Configuration
class nerfitArguments:
    def __init__(
        self,
        model_name: str,
        train_annotations: List[
            Union[
                Dict[str,str],
                Dict[str,List[List[str]]],
                Dict[str,List[Dict[str,Union[int,str]]]],
                str
            ]
        ],
        val_annotations: List[
            Union[
                Dict[str,str],
                Dict[str,List[List[str]]],
                Dict[str,List[Dict[str,Union[int,str]]]],
                str
            ]
        ],
        ent2emb: Dict[str, torch.Tensor],
        peft_lora:bool=False,
        peft_config:Optional[Dict[str,Union[int,float,bool]]]=None, # {'lora_r':8,'lora_alpha':32,'lora_dropout':0.1, 'use_dora': True, 'inference_mode': False}
    ):
        """
        Args:
            model_name (str): Name of the pre-trained model to use.
            train_annotations (torch.utils.data.Dataset): Training dataset.
            val_annotations (torch.utils.data.Dataset): Validation dataset.
            ent2emb (Dict[str, torch.Tensor]): Entity to embedding lookup dictionary.
            peft_lora (bool, optional): Whether to use LoRA rank parameter. Defaults to False.
            peft_config (int, optional): LoRA configuration. Defaults to None.
        """
        self.model_name = model_name
        self.train_annotations = train_annotations
        self.val_annotations = val_annotations
        self.ent2emb = ent2emb
        self.peft_lora = peft_lora
        self.peft_config = peft_config
        self.metric = evaluate.load("seqeval")