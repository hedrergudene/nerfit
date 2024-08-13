# Libraries
import os
import logging as log
import sys
from typing import List, Dict
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
import torch
from litellm import completion
from .dataset import nerfitDataset


# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


# Helper method to get sentence embeddings for entities classes
def build_lookup_table(
    annotations:List[str],
    st_model_name:str="sentence-transformers/LaBSE",
    llm:str = "gpt-4o-mini",
) -> Dict[str,torch.Tensor]:

    # Check env variable has been set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(f"Environment variable 'OPENAI_API_KEY' must be set.")

    # Build label-samples mapping
    ent2samples = {}
    for annotation in annotations:
        annot = nerfitDataset._parse_annotation(annotation)
        for _, _, entity in annot['entities']:
            if entity not in ent2samples.keys():
                ent2samples[entity] = [annotation]
            elif ((entity in ent2samples.keys()) & (annotation not in ent2samples[entity]) & (len(ent2samples[entity])<10)):
                ent2samples[entity].append(annotation)
            else:
                continue

    # Create description using LLMs
    ent2emb = {}
    st_model = SentenceTransformer(st_model_name)
    for label, samples in ent2samples.items():
        samples = '\n* '.join(samples)
        response = completion(
            model=llm,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"As an expert NER data labeller, provide a description of the label {label} based on the following samples, in the original language those samples are written:{samples}"},
            ]
        )
        with torch.no_grad():
            ent2emb[label] = st_model.encode([response.choices[0].message.content], normalize_embeddings=True, show_progress_bar=False, convert_to_tensor=True).flatten()
    del st_model
    # Output
    return ent2emb


# Helper method to generate sentence embeddings from natural descriptions of labels
def build_lookup_table_from_string(
    ent2emb:Dict[str,str],
    st_model_name:str="sentence-transformers/LaBSE",
) -> Dict[str,torch.Tensor]:

    # Create description using LLMs
    st_model = SentenceTransformer(st_model_name)
    output = {}
    for k,v in ent2emb.items():
        with torch.no_grad():
            output[k] = st_model.encode([v], normalize_embeddings=True, show_progress_bar=False, convert_to_tensor=True).flatten()
    del st_model
    # Output
    return output


# Helper method to save weights of NER models without last linear layer
def save_updated_fill_mask_model(model:AutoModelForTokenClassification, save_directory:str):

    # Extract the base model (original fill-mask model)
    base_model = getattr(model, model.base_model_prefix)  # Generic way to access the base model

    # Load the corresponding masked language model (fill-mask model)
    fill_mask_model = AutoModelForMaskedLM.from_pretrained(model)

    # Replace the base model with the updated one
    setattr(fill_mask_model, fill_mask_model.base_model_prefix, base_model)

    # Save the updated fill-mask model
    fill_mask_model.save_pretrained(save_directory)