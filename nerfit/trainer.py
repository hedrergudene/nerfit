# Libraries
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from peft import PeftModel, TaskType
import numpy as np
from typing import Optional, List, Callable, Dict, Any, Union
from nerfit.args import nerfitArguments
from nerfit.callbacks import SavePeftModelCallback
from nerfit.collator import nerfitDataCollator
from nerfit.dataset import nerfitDataset
from nerfit.model import nerfitModel
from nerfit.utils import build_lookup_table, build_lookup_table_from_string


# Main class
class nerfitTrainer:
    def __init__(self, config: nerfitArguments, args_pretraining:Optional[TrainingArguments], args_ner: Optional[TrainingArguments]):
        """
        Args:
            config (TrainerConfig): Configuration object containing all necessary training parameters.
        """
        self.config = config
        self.tokenizer = self._prepare_tokenizer()
        self.ent2emb = self._prepare_embeddings(config.ent2emb, config.train_annotations, config.val_annotations)
        self.train_dataset, self.val_dataset = self._prepare_dataset(config.train_annotations, config.val_annotations)
        self.model = self._prepare_model(self.config.peft_lora, self.config.peft_config)
        self.collate_fn = self._prepare_data_collator()
        self.args_pretraining = self._prepare_pretraining_config(args_pretraining)
        self.args_ner = self._prepare_ner_config(args_ner)


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
    ) -> List[Dict[str,Union[str, int]]]:
        raise NotImplementedError(f"This is a base class; you must build your own parsing strategy for your dataset.")


    def _prepare_embeddings(
            self,
            ent2emb:Optional[Dict[str, Union[torch.Tensor, str]]],
                train_annotations: Optional[
                    List[
                        Union[
                            Dict[str,str],
                            Dict[str,List[List[str]]],
                            Dict[str,List[Dict[str,Union[int,str]]]],
                            str
                        ]
                    ]
                ],
                val_annotations: Optional[
                    List[
                        Union[
                            Dict[str,str],
                            Dict[str,List[List[str]]],
                            Dict[str,List[Dict[str,Union[int,str]]]],
                            str
                        ]
                    ]
                ]        
    ) -> torch.Tensor:
        if ent2emb is None:
            return build_lookup_table(self._parse_annotation(self._parse_annotation(train_annotations) + self._parse_annotation(val_annotations)))
        elif all([isinstance(v,str) for _,v in ent2emb.items()]):
            return build_lookup_table_from_string(ent2emb)
        elif all([isinstance(v,torch.Tensor) for _,v in ent2emb.items()]):
            return ent2emb
        else:
            raise ValueError(f"`ent2emb` must either be None, a dictionary with label-description pairs, or label-tensor pairs.")


    def _prepare_dataset(self, train_annotations:List[Dict[str,Union[str, int]]], val_annotations:List[Dict[str,Union[str, int]]]) -> tuple[nerfitDataset, nerfitDataset]:
        """
        Prepares the Dataset objects for training and validation splits.

        Returns:
            tuple: A tuple containing the training and validation Dataset objects.
        """
        train_dataset = nerfitDataset(
            self._parse_annotation(train_annotations),
            self.ent2emb,
            self.tokenizer
        )
        val_dataset = nerfitDataset(
            self._parse_annotation(val_annotations),
            self.ent2emb,
            self.tokenizer
        )
        return train_dataset, val_dataset


    def _prepare_data_collator(self) -> Callable:
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

        model = nerfitModel(
            model_name=model_name,
            projection_dim=projection_dim,
            peft_lora=peft_lora,
            peft_config=peft_config
        )

        return model


    def _prepare_pretraining_config(self, args_pretraining:Optional[TrainingArguments]) -> TrainingArguments:
        if args_pretraining is None:
            args_pretraining = TrainingArguments(
                output_dir="./nerfit/pretraining",
                dataloader_num_workers=4,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=16,
                learning_rate=1e-4,
                weight_decay=1e-2,
                lr_scheduler_type='cosine',
                warmup_steps=500,
                fp16=True,
                gradient_accumulation_steps=1,
                max_grad_norm=1.,
                seed=123,
                max_steps=1500,
                eval_strategy="steps",
                eval_steps=250,
                logging_strategy="steps",
                logging_steps=250,
                save_strategy="steps",
                save_steps=250,
                load_best_model_at_end=True,
                greater_is_better=False,
                save_total_limit=1,
                remove_unused_columns=False,
                push_to_hub=False
            )
            return args_pretraining
        else:
            return args_pretraining


    def _prepare_ner_config(self, args_ner:Optional[TrainingArguments]) -> TrainingArguments:
        if args_ner is None:
            args_ner = TrainingArguments(
                output_dir="./nerfit/ner",
                dataloader_num_workers=4,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=16,
                learning_rate=1e-4,
                weight_decay=1e-2,
                lr_scheduler_type='cosine',
                warmup_steps=500,
                bf16=False,
                fp16=True,
                gradient_accumulation_steps=1,
                max_grad_norm=1.,
                seed=123,
                max_steps=2000,
                eval_strategy="steps",
                eval_steps=250,
                logging_strategy="steps",
                logging_steps=250,
                save_strategy="steps",
                save_steps=250,
                metric_for_best_model='eval_overall_f1',
                load_best_model_at_end=True,
                greater_is_better=True,
                save_total_limit=1,
                remove_unused_columns=False,
                push_to_hub=False
            )
            return args_ner
        else:
            return args_ner


    def train(self):
        # Pretraining stage
        self._fit_pretraining()
        # Prepare NER model
        self.model = self._setup_ner_checkpoint()
        # Trainer NER
        self._fit_ner()


    def _fit_pretraining(self) -> None:
        """
        The main training loop that iterates through training steps, logs metrics, evaluates the model, and saves checkpoints.
        """
        trainer = CustomPreTrainer(
            self.model,
            self.args_pretraining,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.collate_fn,
            tokenizer=self.tokenizer,
            callbacks=[SavePeftModelCallback]
        )
        trainer.train()


    def _fit_ner(self) -> None:
        trainer =  Trainer(
            self.model,
            self.args_ner,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.collate_fn,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
            callbacks=[SavePeftModelCallback]
        )
        trainer.train()


    def _compute_metrics(self, eval_preds) -> Dict:
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[self.train_dataset.id2label[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.train_dataset.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = self.metric.compute(predictions=true_predictions, references=true_labels)
        metrics = {}
        for k,v in all_metrics.items():
            if isinstance(v,dict):
                for v_k, v_v in v.items():
                    metrics[f"{k}_{v_k}"] = v_v
            else:
                metrics[k] = v
        return metrics 


    def _setup_ner_checkpoint(self) -> Union[AutoModelForTokenClassification, PeftModel]:
        if isinstance(self.model.base_model, PeftModel):
            model = PeftModel.from_pretrained(
                model=AutoModelForTokenClassification.from_pretrained(self.config.model_name, num_labels=len(self.train_dataset.id2label)),
                model_id=self.args_pretraining.output_dir,
                task_type=TaskType.TOKEN_CLS,
                is_trainable=True
            )
        else:
            model = AutoModelForTokenClassification.from_pretrained(self.args_pretraining.output_dir, num_labels=len(self.train_dataset.id2label))
        return model


# Huggingface wrapper for pretraining stage
class CustomPreTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        inputs.pop("labels_ner")
        outputs = model(**inputs)
        return (outputs['loss'], outputs['logits']) if return_outputs else outputs['loss']