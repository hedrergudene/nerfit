# Libraries
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel, TaskType
from accelerate import Accelerator
from itertools import cycle
from rich.table import Table
from rich.console import Console
from tqdm.auto import tqdm
import json
import os
from typing import Optional, List, Callable, Dict, Any, Union
from nerfit.dataset import nerfitDataset
from nerfit.collator import nerfitDataCollator
from nerfit.model import nerfitModel
from nerfit.utils import build_lookup_table, build_lookup_table_from_string


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
        peft_lora:bool=False,
        peft_config:Optional[Dict[str,Union[int,float,bool]]]=None, # {'lora_r':8,'lora_alpha':32,'lora_dropout':0.1, 'use_dora': True}
        inference_mode: bool = False,
        dataloader_num_workers: int = 4,
        num_steps: int = 1000,
        eval_steps: int = 100,
        logging_steps: int = 100,
        batch_size: int = 32,
        backbone_lr: float = 2e-5,
        projection_lr: float = 1e-4,
        weight_decay: float = 1e-2,
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
            peft_lora (bool, optional): Whether to use LoRA rank parameter. Defaults to False.
            peft_config (int, optional): LoRA configuration. Defaults to None.
            inference_mode (bool, optional): If True, sets model to inference mode. Defaults to False.
            dataloader_num_workers (int, optional): CPU parallelisation when loading data. Defaults to 4.
            num_steps (int, optional): Number of training steps. Defaults to 1000.
            eval_steps (int, optional): Number of steps between each evaluation-callback. Defaults to 100.
            logging_steps (int, optional): Number of steps between logging metrics. Defaults to 100.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            backbone_lr (float, optional): Learning rate for the backbone model. Defaults to 2e-5.
            projection_lr (float, optional): Learning rate for the projection layer. Defaults to 1e-4.
            weight_decay (float, optional): Weight decay for both optimisers. Defaults to 1e-2.
            output_dir (str, optional): Directory to save model checkpoints. Defaults to './model'.
            metrics_output_path (str, optional): File path to save training metrics. Defaults to './metrics.json'.
            patience (Optional[int], optional): Number of steps to wait for improvement before stopping. Defaults to None.
        """
        self.model_name = model_name
        self.train_annotations = train_annotations
        self.val_annotations = val_annotations
        self.ent2emb = ent2emb
        self.peft_lora = peft_lora
        self.peft_config = peft_config
        self.inference_mode = inference_mode
        self.dataloader_num_workers = dataloader_num_workers
        self.num_steps = num_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.batch_size = batch_size
        self.backbone_lr = backbone_lr
        self.projection_lr = projection_lr
        self.weight_decay = weight_decay
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
        self.model = self._prepare_model(self.config.peft_lora, self.config.peft_config)
        self.accelerator = Accelerator()
        self.optimizer = self._configure_optimizer()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.num_steps)
        self.train_dataloader, self.val_dataloader = self._prepare_dataloader()

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
            num_workers=self.config.dataloader_num_workers,
            shuffle=True,
            collate_fn=self._data_collator()
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size * 2,
            num_workers=self.config.dataloader_num_workers,
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

    def _prepare_model(self, peft_lora:bool, peft_config:Optional[Dict[str,Union[int,float,bool]]]=None) -> torch.nn.Module:
        """
        Prepares the nerfitModel based on the provided configuration.

        Returns:
            torch.nn.Module: The prepared model.
        """
        model_name = self.config.model_name
        projection_dim = next(iter(self.ent2emb.values())).shape[-1]
        peft_lora = self.config.peft_lora
        peft_config = self.config.peft_config
        inference_mode = self.config.inference_mode

        model = nerfitModel(
            model_name=model_name,
            projection_dim=projection_dim,
            peft_lora=peft_lora,
            peft_config=peft_config,
            inference_mode=inference_mode
        )

        return model

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer with different learning rates for different model parts.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        return torch.optim.AdamW([
            {'params': self.model.base_model.parameters(), 'lr': self.config.backbone_lr, 'weight_decay': self.config.weight_decay},
            {'params': self.model.projection_layer.parameters(), 'lr': self.config.projection_lr, 'weight_decay': self.config.weight_decay}
        ])

    def train(self):
        # Pretraining stage
        self._fit_pretraining()
        # Prepare NER model
        self._setup_ner_checkpoint()

    def _fit_pretraining(self) -> None:
        """
        The main training loop that iterates through training steps, logs metrics, evaluates the model, and saves checkpoints.
        """
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.model.train()
        train_iter = cycle(self.train_dataloader)
        self._reset_params() # Initialize training params

        # Initialize the progress bar
        progress_bar = tqdm(range(self.config.num_steps), desc="Training", unit="step")
        

        for step in progress_bar:
            batch = next(train_iter)
            loss = self._training_step_pretraining(batch)
            self.optimizer.step()
            self.scheduler.step()
            self.train_loss = loss.item()

            # Update the progress bar with the current training loss
            progress_bar.set_postfix(
                {
                    "train_loss": self.train_loss,
                    "val_loss": self.val_loss if self.val_loss is not None else "N/A",
                    'body_lr': self.optimizer.param_groups[0]['lr'],
                    'head_lr': self.optimizer.param_groups[1]['lr']
                }
            )

            # Evaluation
            if (step + 1) % self.config.eval_steps == 0:
                self._evaluate_pretraining(step)  # Update the validation loss
            
            # Early stopping
            if self.config.patience is not None:
                if self.early_stopping_counter >= self.config.patience:
                    print(f"Early stopping at step {step + 1} due to no improvement in validation loss.")
                    break

        # Close the progress bar after training completes
        progress_bar.close()

    def _training_step_pretraining(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Performs a single training step: forward pass, loss calculation, and backpropagation.

        Args:
            batch (Dict[str, torch.Tensor]): A batch of data.

        Returns:
            torch.Tensor: The computed loss.
        """
        self.optimizer.zero_grad()
        output = self.model(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'])
        mask = (batch['labels_pretraining'] != -100)

        if batch['embeddings'].size(1) > 0:
            logits = torch.bmm(batch['embeddings'], output.transpose(1, 2))
            logits = logits * mask
            labels = batch['labels_pretraining'] * mask
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='sum')
            loss = loss / mask.sum()
        else:
            loss = torch.tensor(0.0, device=labels.device)

        self.accelerator.backward(loss)
        return loss

    def _evaluate_pretraining(self, step:int) -> float:
        """
        Evaluates the model on the validation set and logs the validation loss.

        Args:
            step (int): The current training step.
            
        Returns:
            float: The computed validation loss.
        """
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                output = self.model(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'])
                mask = (batch['labels_pretraining'] != -100)

                if batch['embeddings'].size(1) > 0:
                    logits = torch.bmm(batch['embeddings'], output.transpose(1, 2))
                    logits = logits * mask
                    labels = batch['labels_pretraining'] * mask
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='sum')
                    loss = loss / mask.sum()
                else:
                    loss = torch.tensor(0.0, device=labels.device)

                val_loss += loss.item()

        self.val_loss = val_loss / len(self.val_dataloader)
        # Create record
        self.history[step+1] = {
            "Train Loss": self.train_loss,
            "Val Loss": self.val_loss,
            "Body LR": self.optimizer.param_groups[0]['lr'],
            "Head LR": self.optimizer.param_groups[1]['lr']
        }

        # Update the table below the progress bar
        self._print_metrics_table(step + 1)

        # Save best ckpt
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_model(stage='pretraining')
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        self.model.train()

        return val_loss

    def _reset_params(self) -> None:
        self.train_loss = None
        self.val_loss = None
        self.history = {}
        self.show_header = True
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.console = Console(style="black")

    def _print_metrics_table(self, step: int) -> None:
        """
        Prints a table with the current training and validation metrics.

        Args:
            step (int): The current training step.
        """
        if not hasattr(self, 'table'):  # Create the table if it doesn't exist
            self.table = Table(show_header=True, header_style="bold white")
            self.table.add_column("Step", justify="right", style="white")
            self.table.add_column("Train Loss", justify="right", style="white")
            self.table.add_column("Val Loss", justify="right", style="white")
            self.table.add_column("Body LR", justify="right", style="white")
            self.table.add_column("Head LR", justify="right", style="white")
    
        # Add a new row to the existing table
        self.table.add_row(
            str(step),
            f"{self.history[step]['Train Loss']:.6f}",
            f"{self.history[step]['Val Loss']:.6f}" if self.history[step]['Val Loss'] is not None else "N/A",
            f"{self.history[step]['Body LR']:.6f}",
            f"{self.history[step]['Head LR']:.6f}"
        )
    
        # Clear the console and print the updated table
        self.console.clear()
        self.console.print(self.table)

    def _save_model(self, stage:str):
        """
        Saves the model checkpoint, removing any previous checkpoint if it exists.

        Args:
            step (int): The current training step.
            stage (str): Whether the training phase is 'pretraining' or 'ner'
        """
        # Define the output directory for saving the model
        output_dir = os.path.join(self.config.output_dir, stage)

        # Remove previous checkpoint if it exists
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            os.rmdir(output_dir)

        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the model and tokenizer
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.base_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def _setup_ner_checkpoint(self) -> Union[AutoModelForTokenClassification, PeftModel]:
        if isinstance(self.model.base_model, PeftModel):
            self.model = PeftModel.from_pretrained(
                model=AutoModelForTokenClassification.from_pretrained(self.config.model_name, num_labels=len(self.train_dataloader.dataset.id2label)),
                model_id=os.path.join(self.config.output_dir, 'pretraining'),
                task_type=TaskType.TOKEN_CLS,
                is_trainable=True
            )
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(os.path.join(self.config.output_dir, 'pretraining'), num_labels=len(self.train_dataloader.dataset.id2label))