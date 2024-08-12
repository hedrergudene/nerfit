# Libraries
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from accelerate import Accelerator
from itertools import cycle
import json
import os
from typing import Optional, List, Callable, Dict, Any
from .collator import nerfitDataCollator


# Configuration
class TrainerConfig:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        ent2emb: Dict[str, torch.Tensor],
        num_steps: int,
        callback_steps: int,
        save_steps: int,
        batch_size: int,
        backbone_lr: float,
        projection_lr: float,
        output_dir: str,
        metrics_output_path: str,
        patience: Optional[int] = None
    ):
        """
        Args:
            model (torch.nn.Module): The model to be trained.
            tokenizer (Any): Tokenizer used for text encoding.
            train_dataset (torch.utils.data.Dataset): Training dataset.
            val_dataset (torch.utils.data.Dataset): Validation dataset.
            ent2emb (Dict[str, torch.Tensor]): Entity to embedding lookup dictionary.
            num_steps (int): Number of training steps.
            callback_steps (int): Number of steps between each callback.
            save_steps (int): Number of steps between each model checkpoint save.
            batch_size (int): Batch size for training.
            backbone_lr (float): Learning rate for the backbone model.
            projection_lr (float): Learning rate for the projection layer.
            output_dir (str): Directory to save model checkpoints.
            metrics_output_path (str): File path to save training metrics.
            patience (Optional[int]): Number of steps to wait for improvement before stopping.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.ent2emb = ent2emb
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
class nerfitTrainer:
    def __init__(self, config: TrainerConfig):
        """
        Args:
            config (TrainerConfig): Configuration object containing all necessary training parameters.
        """
        self.config = config
        self.model = config.model
        self.tokenizer = config.tokenizer
        self.train_dataset = config.train_dataset
        self.val_dataset = config.val_dataset
        self.ent2emb = config.ent2emb
        self.accelerator = Accelerator()
        self.optimizer = self._configure_optimizer()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.num_steps)
        self.train_dataloader, self.val_dataloader = self._prepare_dataloader()
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0

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

    def _prepare_dataloader(self) -> tuple[DataLoader, DataLoader]:
        """
        Prepares the DataLoader objects for training and validation datasets.

        Returns:
            tuple: A tuple containing the training and validation DataLoader objects.
        """
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._data_collator()
        )
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            collate_fn=self._data_collator()
        )
        train_dataloader, val_dataloader = self.accelerator.prepare(train_dataloader, val_dataloader)
        return train_dataloader, val_dataloader

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