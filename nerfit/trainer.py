# Libraries
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator
from itertools import cycle
import json
import os
from typing import Optional, List, Callable, Dict, Any, Union
from .dataset import nerfitDataset
from .collator import nerfitDataCollator
from .model import nerfitModel
from .utils import build_lookup_table, build_lookup_table_from_string


# Configuration
class TrainerConfig:
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
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        inference_mode: bool = False,
        num_steps: int = 1000,
        callback_steps: int = 100,
        save_steps: int = 100,
        batch_size: int = 32,
        backbone_lr: float = 2e-5,
        projection_lr: float = 1e-4,
        output_dir: str = './model',
        metrics_output_path: str = './metrics.json',
        patience: Optional[int] = None
    ):
        """
        Args:
            model_name (str): Name of the pre-trained model to use.
            train_annotations (torch.utils.data.Dataset): Training dataset.
            val_annotations (torch.utils.data.Dataset): Validation dataset.
            ent2emb (Dict[str, torch.Tensor]): Entity to embedding lookup dictionary.
            projection_dim (int): Dimension of the projection layer output.
            lora_r (int, optional): LoRA rank parameter. Defaults to 16.
            lora_alpha (int, optional): LoRA alpha parameter. Defaults to 32.
            lora_dropout (float, optional): Dropout rate for LoRA. Defaults to 0.1.
            inference_mode (bool, optional): If True, sets model to inference mode. Defaults to False.
            num_steps (int, optional): Number of training steps. Defaults to 1000.
            callback_steps (int, optional): Number of steps between each callback. Defaults to 100.
            save_steps (int, optional): Number of steps between each model checkpoint save. Defaults to 100.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            backbone_lr (float, optional): Learning rate for the backbone model. Defaults to 2e-5.
            projection_lr (float, optional): Learning rate for the projection layer. Defaults to 1e-4.
            output_dir (str, optional): Directory to save model checkpoints. Defaults to './model'.
            metrics_output_path (str, optional): File path to save training metrics. Defaults to './metrics.json'.
            patience (Optional[int], optional): Number of steps to wait for improvement before stopping. Defaults to None.
        """
        self.model_name = model_name
        self.train_annotations = train_annotations
        self.val_annotations = val_annotations
        self.ent2emb = ent2emb
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.inference_mode = inference_mode
        self.num_steps = num_steps
        self.callback_steps = callback_steps
        self.save_steps = save_steps
        self.batch_size = batch_size
        self.backbone_lr = backbone_lr
        self.projection_lr = projection_lr
        self.output_dir = output_dir
        self.metrics_output_path = metrics_output_path
        self.patience = patience



# Main class
class Trainer:
    def __init__(self, config: TrainerConfig):
        """
        Args:
            config (TrainerConfig): Configuration object containing all necessary training parameters.
        """
        self.config = config
        self.tokenizer = self._prepare_tokenizer()
        self.train_annotations = config.train_annotations
        self.val_annotations = config.val_annotations
        self.ent2emb = self._prepare_embeddings(config.ent2emb)
        self.model = self._prepare_model()
        self.accelerator = Accelerator()
        self.optimizer = self._configure_optimizer()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.num_steps)
        self.train_dataloader, self.val_dataloader = self._prepare_dataloader()
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0

    @staticmethod
    def _parse_annotation(
        annotations:List[
            Union[
                Dict[str,str],
                Dict[str,List[List[str]]],
                Dict[str,List[Dict[str,Union[int,str]]]],
                str
            ]
        ]
    ) -> List[Dict[str,Union[str]]]:
        raise NotImplementedError(f"This is a base class; you must build your own parsing strategy for your dataset.")

    def _prepare_embeddings(self, ent2emb:Optional[Dict[str, Union[torch.Tensor, str]]]) -> torch.Tensor:
        if ent2emb is None:
            return build_lookup_table(self._parse_annotation(self.train_annotations))
        elif all([isinstance(v,str) for _,v in ent2emb.items()]):
            return build_lookup_table_from_string(ent2emb)
        elif all([isinstance(v,torch.Tensor) for _,v in ent2emb.items()]):
            return ent2emb
        else:
            raise ValueError(f"`ent2emb` must either be None, a dictionary with label-description pairs, or label-tensor pairs.")

    def _prepare_dataset(self) -> tuple[Dataset, Dataset]:
        """
        Prepares the Dataset objects for training and validation splits.

        Returns:
            tuple: A tuple containing the training and validation Dataset objects.
        """
        train_dataset = nerfitDataset(
            self._parse_annotation(self.train_annotations),
            self.ent2emb,
            self.tokenizer
        )
        val_dataset = nerfitDataset(
            self._parse_annotation(self.val_annotations),
            self.ent2emb,
            self.tokenizer
        )
        return train_dataset, val_dataset

    def _data_collator(self) -> Callable:
        """
        Returns the data collator function for padding and batching the data.

        Returns:
            Callable: The data collator function.
        """
        return nerfitDataCollator(
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=self.model.base_model.config.max_position_embeddings,
            projection_dim=self.model.projection_layer.out_features
        )

    def _prepare_dataloader(self) -> tuple[DataLoader, DataLoader]:
        """
        Prepares the DataLoader objects for training and validation datasets.

        Returns:
            tuple: A tuple containing the training and validation DataLoader objects.
        """
        train_dataset, val_dataset = self._prepare_dataset()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._data_collator()
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            collate_fn=self._data_collator()
        )
        train_dataloader, val_dataloader = self.accelerator.prepare(train_dataloader, val_dataloader)
        return train_dataloader, val_dataloader

    def _prepare_tokenizer(self) -> AutoTokenizer:
        """
        Prepares the tokenizer for the training process.    

        This method loads the tokenizer based on the model name provided in the configuration.
        The tokenizer is essential for encoding text data into input IDs and attention masks
        that the model can process.
        """
        # Load the tokenizer using the model name from the configuration
        return AutoTokenizer.from_pretrained(self.config.model_name)

    def _prepare_model(self) -> torch.nn.Module:
        """
        Prepares the nerfitModel based on the provided configuration.
    
        Returns:
            torch.nn.Module: The prepared model.
        """
        model_name = self.config.model_name
        projection_dim = next(iter(self.ent2emb.values())).shape[-1]
        lora_r = self.config.lora_r
        lora_alpha = self.config.lora_alpha
        lora_dropout = self.config.lora_dropout
        inference_mode = self.config.inference_mode
    
        model = nerfitModel(
            model_name=model_name,
            projection_dim=projection_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            inference_mode=inference_mode
        )
    
        return model

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer with different learning rates for different model parts.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        return torch.optim.Adam([
            {'params': self.model.base_model.parameters(), 'lr': self.config.backbone_lr},
            {'params': self.model.projection_layer.parameters(), 'lr': self.config.projection_lr}
        ])

    def fit(self):
        """
        The main training loop that iterates through training steps, logs metrics, evaluates the model, and saves checkpoints.
        """
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.model.train()
        train_iter = cycle(self.train_dataloader)
        loss_values = []

        for step in range(self.config.num_steps):
            batch = next(train_iter)
            loss = self._training_step(batch)
            self.optimizer.step()
            self.scheduler.step()
            loss_values.append(loss.item())

            if (step + 1) % self.config.callback_steps == 0:
                self._log_training_metrics(loss_values, step)
                self._evaluate(step)

            if (step + 1) % self.config.save_steps == 0:
                self.save_model(step)

            if self.config.patience is not None:
                if self._early_stopping():
                    print(f"Early stopping at step {step + 1} due to no improvement in validation loss.")
                    break

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Performs a single training step: forward pass, loss calculation, and backpropagation.

        Args:
            batch (Dict[str, torch.Tensor]): A batch of data.

        Returns:
            torch.Tensor: The computed loss.
        """
        self.optimizer.zero_grad()
        output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        mask = (batch['labels'] != -100)
        batch['labels'][~mask] = 0

        if batch['embeddings'].size(1) > 0:
            logits = torch.bmm(batch['embeddings'], output.transpose(1, 2))
            logits = logits * mask
            batch['labels'] = batch['labels'] * mask
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch['labels'], reduction='sum')
            loss = loss / mask.sum()
        else:
            loss = torch.tensor(0.0, device=batch['labels'].device)

        self.accelerator.backward(loss)
        return loss

    def _evaluate(self, step: int):
        """
        Evaluates the model on the validation set and logs the validation loss.

        Args:
            step (int): The current training step.
        """
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                mask = (batch['labels'] != -100)
                batch['labels'][~mask] = 0

                if batch['embeddings'].size(1) > 0:
                    logits = torch.bmm(batch['embeddings'], output.transpose(1, 2))
                    logits = logits * mask
                    batch['labels'] = batch['labels'] * mask
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch['labels'], reduction='sum')
                    loss = loss / mask.sum()
                else:
                    loss = torch.tensor(0.0, device=batch['labels'].device)

                val_loss += loss.item()

        val_loss /= len(self.val_dataloader)
        print(f"Validation Loss at step {step + 1}: {val_loss:.4f}")
        self.model.train()
        self._early_stopping_update(val_loss)

    def save_model(self, step: int):
        """
        Saves the model checkpoint.

        Args:
            step (int): The current training step.
        """
        output_dir = os.path.join(self.config.output_dir, f"checkpoint-{step + 1}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved at {output_dir}")

    def _log_training_metrics(self, loss_values: List[float], step: int):
        """
        Logs training metrics such as average training loss and learning rates.

        Args:
            loss_values (List[float]): List of loss values for the current callback interval.
            step (int): The current training step.
        """
        avg_train_loss = sum(loss_values[-self.config.callback_steps:]) / self.config.callback_steps
        lr_body = self.optimizer.param_groups[0]['lr']
        lr_head = self.optimizer.param_groups[1]['lr']
        with open(self.config.metrics_output_path, 'a') as f:
            json.dump({
                'step': step + 1,
                'train_loss': avg_train_loss,
                'lr_body': lr_body,
                'lr_head': lr_head
            }, f)
            f.write('\n')

        print(f"Step {step + 1}/{self.config.num_steps} - Train Loss: {avg_train_loss:.4f} - LR Body: {lr_body:.6f} - LR Head: {lr_head:.6f}")

    def _early_stopping_update(self, val_loss: float):
        """
        Updates the early stopping counter based on the validation loss.

        Args:
            val_loss (float): The current validation loss.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

    def _early_stopping(self) -> bool:
        """
        Checks if early stopping should be triggered.

        Returns:
            bool: True if early stopping should occur, False otherwise.
        """
        if self.config.patience is None:
            return False
        return self.early_stopping_counter >= self.config.patience