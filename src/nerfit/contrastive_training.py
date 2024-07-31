# Libraries
from tqdm.auto import tqdm
import os
from pathlib import Path
import json
import logging as log
import sys
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingLR
from itertools import cycle
import fire
from .dataset import nerfitDataset
from .collator import nerfitDataCollator
from .model import nerfitModel

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


# Main method. Fire automatically allign method arguments with parse commands from console
def main(
    input_path,
    sentence_transformers_model_name:str = "sentence-transformers/LaBSE",
    encoder_model_name:str = "PlanTL-GOB-ES/roberta-large-bne",
    backbone_lr:float = 2e-5,
    projection_lr:float = 1e-4,
    batch_size:int = 16,
    num_steps:int = 500,
    callback_steps:int = 25,
    patience:int=3,
    output_path:str = './output'
):

    # Folder structure
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Sample data
    with open(input_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    entity_descriptions = {
        "Torneo": "Competencia deportiva organizada por la AFC.",
        "Organismo": "Confederación Asiática de Fútbol.",
        # Add other labels and descriptions here...
    }

    st_model = SentenceTransformer(sentence_transformers_model_name)
    projection_dim = st_model.get_sentence_embedding_dimension()
    tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)

    train_dts = nerfitDataset([v for _,v in annotations.items()][:80], st_model, tokenizer, entity_descriptions)
    val_dts = nerfitDataset([v for _,v in annotations.items()][80:100], st_model, tokenizer, entity_descriptions)

    train_dtl  = torch.utils.data.DataLoader(train_dts, batch_size=batch_size, collate_fn=nerfitDataCollator(pad_token_id=tokenizer.pad_token_id), shuffle=True)
    val_dtl  = torch.utils.data.DataLoader(val_dts, batch_size=2*batch_size, collate_fn=nerfitDataCollator(pad_token_id=tokenizer.pad_token_id), shuffle=True)

    entity_embeddings = train_dts.entity_embeddings
    model = nerfitModel(encoder_model_name, projection_dim, entity_embeddings)

    optimizer = torch.optim.Adam([
        {'params': model.bert_model.parameters(), 'lr': backbone_lr},
        {'params': model.projection_layer.parameters(), 'lr': projection_lr}
    ])

    def evaluate(model, val_loader):
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                cosine_sim = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = model.compute_contrastive_loss(cosine_sim, labels)
                val_loss += loss.item()
        model.train()
        return val_loss / len(val_loader)

    accelerator = Accelerator()
    model, optimizer, train_dtl, val_dtl = accelerator.prepare(model, optimizer, train_dtl, val_dtl)

    step = 0
    best_val_loss = float('inf')
    no_improve_steps = 0
    loss_values = []
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps)

    model.train()
    train_iter = cycle(train_dtl)

    for step in tqdm(range(num_steps), desc="Training", total=num_steps):

        #
        # Training step
        #

        batch = next(train_iter)

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        cosine_sim = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = model.compute_contrastive_loss(cosine_sim, labels)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        loss_values.append(loss.item())


        #
        # Validation
        #

        if (step + 1) % callback_steps == 0:
            avg_train_loss = sum(loss_values[-callback_steps:]) / callback_steps
            avg_val_loss = evaluate(model, val_dtl)

            lr_body = optimizer.param_groups[0]['lr']
            lr_head = optimizer.param_groups[1]['lr']

            log.info(f"Step {step + 1}/{num_steps} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - LR Body: {lr_body:.6f} - LR Head: {lr_head:.6f}")

            #
            # Callbacks
            #
            # (1) Save logs
            with open(os.path.join(output_path, 'training_logs.json'), 'a') as f:
                json.dump({
                    'step': step + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'lr_body': lr_body,
                    'lr_head': lr_head
                }, f)
                f.write('\n')
            # (2) Save the model if validation loss decreases
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.bert_model.save_pretrained(output_path)
                print(f"Model saved at step {step + 1}")

                # Reset no_improve_steps counter
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            # (3) Stop training if validation loss does not improve over `patience` iterations
            if no_improve_steps >= patience:
                print(f"No improvement in validation loss for {patience} validation steps. Stopping training.")
                break


if __name__=="__main__":
    fire.Fire(main)