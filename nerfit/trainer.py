# Libraries
import torch
import os
import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel, PeftModelForTokenClassification, TaskType
import numpy as np
from typing import Optional, List, Callable, Dict, Union, Tuple, Any
import evaluate
from nerfit.args import TrainingArguments
from nerfit.callbacks import SavePeftModelCallback
from nerfit.collator import nerfitDataCollator, nerDataCollator
from nerfit.dataset import nerfitDataset, nerDataset
from nerfit.model import nerfitModel
from nerfit.utils import build_lookup_table, build_lookup_table_from_string


# Main class
class Trainer:
    def __init__(self, args: TrainingArguments):
        """
        Args:
            config (TrainingArguments): Configuration object containing all necessary training parameters.
        """
        self.args = args
        self.tokenizer = self._prepare_tokenizer()
        self.ent2emb = self._prepare_embeddings(args.ent2emb, args.train_annotations, args.val_annotations)
        self.train_dataset_pretraining, self.val_dataset_pretraining = self._prepare_dataset_pretraining(args.train_annotations, args.val_annotations)
        self.train_dataset_ner, self.val_dataset_ner = self._prepare_dataset_ner(args.train_annotations, args.val_annotations)
        self.model = self._prepare_model()
        self.collate_fn = self._prepare_data_collator_pretraining()
        self.collate_fn_ner = self._prepare_data_collator_ner()
        self.args_pretraining = self._prepare_pretraining_config(self.args)
        self.args_ner = self._prepare_ner_config(self.args)
        self.best_ckpt_pretraining_path = None
        self.best_ckpt_ner_path = None
        self.metric = evaluate.load('seqeval')


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
            return build_lookup_table(
                self._parse_annotation(train_annotations) + self._parse_annotation(val_annotations),
                self.args.st_model_name,
                self.args.llm
            )
        elif all([isinstance(v,str) for _,v in ent2emb.items()]):
            return build_lookup_table_from_string(ent2emb, self.args.st_model_name)
        elif all([isinstance(v,torch.Tensor) for _,v in ent2emb.items()]):
            return ent2emb
        else:
            raise ValueError(f"`ent2emb` must either be None, a dictionary with label-description pairs, or label-tensor pairs.")


    def _prepare_dataset_pretraining(self, train_annotations:List[Dict[str,Union[str, int]]], val_annotations:List[Dict[str,Union[str, int]]]) -> tuple[nerfitDataset, nerfitDataset]:
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


    def _prepare_dataset_ner(self, train_annotations:List[Dict[str,Union[str, int]]], val_annotations:List[Dict[str,Union[str, int]]]) -> tuple[nerfitDataset, nerfitDataset]:
        """
        Prepares the Dataset objects for training and validation splits.

        Returns:
            tuple: A tuple containing the training and validation Dataset objects.
        """
        train_dataset = nerDataset(
            self._parse_annotation(train_annotations),
            self.ent2emb,
            self.tokenizer
        )
        val_dataset = nerDataset(
            self._parse_annotation(val_annotations),
            self.ent2emb,
            self.tokenizer
        )
        return train_dataset, val_dataset


    def _prepare_data_collator_pretraining(self) -> Callable:
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


    def _prepare_data_collator_ner(self) -> Callable:
        """
        Returns the data collator function for padding and batching the data.

        Returns:
            Callable: The data collator function.
        """
        return nerDataCollator(
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
        return AutoTokenizer.from_pretrained(self.args.model_name)


    def _prepare_model(self) -> torch.nn.Module:
        """
        Prepares the nerfitModel based on the provided configuration.

        Returns:
            torch.nn.Module: The prepared model.
        """
        model_name = self.args.model_name
        projection_dim = next(iter(self.ent2emb.values())).shape[-1]
        peft_lora = self.args.peft_lora
        peft_config = self.args.peft_config

        model = nerfitModel(
            model_name=model_name,
            projection_dim=projection_dim,
            peft_lora=peft_lora,
            peft_config=peft_config
        )

        return model


    def _prepare_pretraining_config(self, args:Optional[TrainingArguments]) -> transformers.TrainingArguments:
        args_pretraining = transformers.TrainingArguments(
            output_dir=os.path.join(args.output_dir,'pretraining'),
            dataloader_num_workers=args.dataloader_num_workers[0],
            per_device_train_batch_size=args.per_device_train_batch_size[0],
            per_device_eval_batch_size=args.per_device_eval_batch_size[0],
            learning_rate=args.learning_rate[0],
            weight_decay=args.weight_decay[0],
            lr_scheduler_type=args.lr_scheduler_type[0],
            warmup_steps=args.warmup_steps[0],
            fp16=args.fp16[0],
            gradient_accumulation_steps=args.gradient_accumulation_steps[0],
            max_grad_norm=args.max_grad_norm[0],
            seed=args.seed[0],
            max_steps=args.max_steps[0],
            eval_strategy=args.eval_strategy[0],
            eval_steps=args.eval_steps[0],
            logging_strategy=args.logging_strategy[0],
            logging_steps=args.logging_steps[0],
            save_strategy='no',
            load_best_model_at_end=False,
            greater_is_better=False,
            save_total_limit=1,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=args.report_to
        )
        return args_pretraining
    

    def _prepare_ner_config(self, args:Optional[TrainingArguments]) -> transformers.TrainingArguments:
        args_ner = transformers.TrainingArguments(
            output_dir=os.path.join(args.output_dir,'ner'),
            dataloader_num_workers=args.dataloader_num_workers[1],
            per_device_train_batch_size=args.per_device_train_batch_size[1],
            per_device_eval_batch_size=args.per_device_eval_batch_size[1],
            learning_rate=args.learning_rate[1],
            weight_decay=args.weight_decay[1],
            lr_scheduler_type=args.lr_scheduler_type[1],
            warmup_steps=args.warmup_steps[1],
            fp16=args.fp16[1],
            gradient_accumulation_steps=args.gradient_accumulation_steps[1],
            max_grad_norm=args.max_grad_norm[1],
            seed=args.seed[1],
            max_steps=args.max_steps[1],
            eval_strategy=args.eval_strategy[1],
            eval_steps=args.eval_steps[1],
            logging_strategy=args.logging_strategy[1],
            logging_steps=args.logging_steps[1],
            save_strategy=args.save_strategy[1],
            save_steps=args.save_steps[1],
            metric_for_best_model='eval_overall_f1',
            load_best_model_at_end=True,
            greater_is_better=True,
            save_total_limit=1,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=args.report_to
        )
        return args_ner


    def train(self):
        # Pretraining stage
        self._fit_pretraining()
        # Prepare NER model
        self.model = self._setup_ner_checkpoint()
        # Trainer NER
        self._fit_ner()


    def _fit_pretraining(self) -> str:
        trainer = CustomPreTrainer(
            self.model,
            self.args_pretraining,
            train_dataset=self.train_dataset_pretraining,
            eval_dataset=self.val_dataset_pretraining,
            data_collator=self.collate_fn,
            tokenizer=self.tokenizer,
            callbacks=[SavePeftModelCallback] if self.args.peft_lora else []
        )
        trainer.train()
        trainer.model.base_model.save_pretrained(self.args_pretraining.output_dir)
        self.best_ckpt_pretraining_path = self.args_pretraining.output_dir


    def _fit_ner(self) -> None:
        trainer =  transformers.Trainer(
            self.model,
            self.args_ner,
            train_dataset=self.train_dataset_ner,
            eval_dataset=self.val_dataset_ner,
            data_collator=self.collate_fn_ner,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
            callbacks=[SavePeftModelCallback] if self.args.peft_lora else []
        )
        trainer.train()
        self.best_ckpt_ner_path = trainer.state.best_model_checkpoint


    def _compute_metrics(self, eval_preds) -> Dict:
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[self.train_dataset_ner.id2label[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.train_dataset_ner.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
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


    def _setup_ner_checkpoint(self) -> Union[AutoModelForTokenClassification, PeftModelForTokenClassification]:
        if self.args.peft_lora:
            model = PeftModelForTokenClassification.from_pretrained(
                model=AutoModelForTokenClassification.from_pretrained(self.args.model_name, num_labels=len(self.train_dataset_ner.id2label)),
                model_id=self.best_ckpt_pretraining_path,
                is_trainable=True
            )
        else:
            model = AutoModelForTokenClassification.from_pretrained(self.best_ckpt_pretraining_path, num_labels=len(self.train_dataset_ner.id2label))
        return model


# Huggingface wrapper for pretraining stage
class CustomPreTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)
        return (outputs['loss'], outputs['logits']) if return_outputs else outputs['loss']