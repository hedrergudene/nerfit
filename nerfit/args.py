from typing import (
    List, Dict, Union, Optional, Any
)
import torch
import json
import inspect
from copy import copy
from dataclasses import dataclass, field, fields

@dataclass
class TrainingArguments:
    model_name: str
    train_annotations: List[
        Union[
            Dict[str, str],
            Dict[str, List[List[str]]],
            Dict[str, List[Dict[str, Union[int, str]]]],
            str
        ]
    ]
    val_annotations: List[
        Union[
            Dict[str, str],
            Dict[str, List[List[str]]],
            Dict[str, List[Dict[str, Union[int, str]]]],
            str
        ]
    ]
    ent2emb: Dict[str, torch.Tensor]
    peft_lora: bool = False
    peft_config: Optional[Dict[str, Union[int, float, bool]]] = None
    output_dir: str = "./output_dir"
    dataloader_num_workers: tuple = (4, 4)
    per_device_train_batch_size: tuple = (8, 8)
    per_device_eval_batch_size: tuple = (16, 16)
    learning_rate: tuple = (1e-4, 1e-4)
    weight_decay: tuple = (1e-2, 1e-2)
    lr_scheduler_type: tuple = ('cosine', 'cosine')
    warmup_steps: tuple = (500, 500)
    fp16: tuple = (True, True)
    gradient_accumulation_steps: tuple = (1, 1)
    max_grad_norm: tuple = (1.0, 1.0)
    seed: tuple = (123, 123)
    max_steps: tuple = (1500, 1500)
    eval_strategy: tuple = ("steps", "steps")
    eval_steps: tuple = (250, 250)
    logging_strategy: tuple = ("steps", "steps")
    logging_steps: tuple = (250, 250)
    save_strategy: tuple = ("steps", "steps")
    save_steps: tuple = (250, 250)

    def __post_init__(self):
        # Validate the structure of train_annotations and val_annotations
        for annotations in [self.train_annotations, self.val_annotations]:
            if not all(isinstance(ann, (dict, str)) or (
                isinstance(ann, dict) and all(
                    isinstance(k, str) and (
                        isinstance(v, str) or
                        (isinstance(v, list) and all(isinstance(i, list) and all(isinstance(j, str) for j in i) for i in v)) or
                        (isinstance(v, list) and all(isinstance(i, dict) and all(isinstance(i_k, str) and isinstance(i_v, (int, str)) for i_k, i_v in i.items()) for i in v))
                    ) for k, v in ann.items()
                )
            ) for ann in annotations):
                raise ValueError("Each annotation in train_annotations and val_annotations must match the specified structure.")

        # Ensure tuple parameters are valid tuples
        def ensure_tuple(param_name: str, param_value: Any) -> Any:
            if isinstance(param_value, tuple):
                if len(param_value) != 2:
                    raise ValueError(f"{param_name} must be a tuple of length 2.")
                return param_value
            else:
                return (param_value, param_value)

        # Convert single values to tuples where needed
        self.dataloader_num_workers = ensure_tuple('dataloader_num_workers', self.dataloader_num_workers)
        self.per_device_train_batch_size = ensure_tuple('per_device_train_batch_size', self.per_device_train_batch_size)
        self.per_device_eval_batch_size = ensure_tuple('per_device_eval_batch_size', self.per_device_eval_batch_size)
        self.learning_rate = ensure_tuple('learning_rate', self.learning_rate)
        self.weight_decay = ensure_tuple('weight_decay', self.weight_decay)
        self.lr_scheduler_type = ensure_tuple('lr_scheduler_type', self.lr_scheduler_type)
        self.warmup_steps = ensure_tuple('warmup_steps', self.warmup_steps)
        self.fp16 = ensure_tuple('fp16', self.fp16)
        self.gradient_accumulation_steps = ensure_tuple('gradient_accumulation_steps', self.gradient_accumulation_steps)
        self.max_grad_norm = ensure_tuple('max_grad_norm', self.max_grad_norm)
        self.seed = ensure_tuple('seed', self.seed)
        self.max_steps = ensure_tuple('max_steps', self.max_steps)
        self.eval_strategy = ensure_tuple('eval_strategy', self.eval_strategy)
        self.eval_steps = ensure_tuple('eval_steps', self.eval_steps)
        self.logging_strategy = ensure_tuple('logging_strategy', self.logging_strategy)
        self.logging_steps = ensure_tuple('logging_steps', self.logging_steps)
        self.save_strategy = ensure_tuple('save_strategy', self.save_strategy)
        self.save_steps = ensure_tuple('save_steps', self.save_steps)

        # Specific validations for parameters that should be tuples
        if not isinstance(self.fp16, (tuple, bool)):
            raise TypeError("fp16 must be a boolean or a tuple of two booleans.")
        if any(not isinstance(param, (int, float, str, bool)) for param in [
            self.learning_rate[0], self.weight_decay[0], self.lr_scheduler_type[0],
            self.warmup_steps[0], self.gradient_accumulation_steps[0], self.max_grad_norm[0],
            self.seed[0], self.max_steps[0], self.eval_steps[0], self.logging_steps[0], self.save_steps[0]
        ]):
            raise TypeError("All parameters should be of type int, float, str, or bool.")
        
        # Additional specific validations
        if self.warmup_steps[0] < 0:
            raise ValueError("warmup_steps must be a non-negative integer.")
        if self.max_steps[0] <= 0:
            raise ValueError("max_steps must be a positive integer.")
        if self.eval_steps[0] <= 0:
            raise ValueError("eval_steps must be a positive integer.")
        if self.logging_steps[0] <= 0:
            raise ValueError("logging_steps must be a positive integer.")
        if self.save_steps[0] <= 0:
            raise ValueError("save_steps must be a positive integer.")
        
        if self.lr_scheduler_type[0] not in ['linear', 'cosine', 'constant']:
            raise ValueError("lr_scheduler_type must be one of 'linear', 'cosine', 'constant'.")
        if self.logging_strategy[0] not in ['no', 'epoch', 'steps']:
            raise ValueError("logging_strategy must be one of 'no', 'epoch', 'steps'.")
        if self.eval_strategy[0] not in ['no', 'epoch', 'steps']:
            raise ValueError("eval_strategy must be one of 'no', 'epoch', 'steps'.")
        if self.save_strategy[0] not in ['no', 'epoch', 'steps']:
            raise ValueError("save_strategy must be one of 'no', 'epoch', 'steps'.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert this instance to a dictionary."""
        return {field.name: getattr(self, field.name) for field in fields(self) if field.init}

    @classmethod
    def from_dict(cls, arguments: Dict[str, Any], ignore_extra: bool = False) -> 'TrainingArguments':
        """Initialize a TrainingArguments instance from a dictionary."""
        if ignore_extra:
            return cls(**{key: value for key, value in arguments.items() if key in inspect.signature(cls).parameters})
        return cls(**arguments)

    def copy(self) -> 'TrainingArguments':
        """Create a shallow copy of this TrainingArguments instance."""
        return copy(self)

    def update(self, arguments: Dict[str, Any], ignore_extra: bool = False) -> 'TrainingArguments':
        """Update and return a new TrainingArguments instance with additional arguments."""
        return TrainingArguments.from_dict({**self.to_dict(), **arguments}, ignore_extra=ignore_extra)

    def to_json_string(self) -> str:
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """Sanitized serialization for use with external systems."""
        d = self.to_dict()
        # Define valid types for sanitized dict
        valid_types = [bool, int, float, str]
        if torch is not None:
            valid_types.append(torch.Tensor)

        return {k: v if isinstance(v, tuple(valid_types)) else str(v) for k, v in d.items()}