# Libraries
import os
import logging as log
import sys
from typing import List
from sentence_transformers import SentenceTransformer
import torch
from litellm import completion
from safetensors.torch import save_file
from .dataset import nerfitDataset
import fire


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
    annotations:List[str],
    llm:str = "gpt-4o-mini",
    st_model:str = "sentence-transformers/LaBSE",
    output_path:str = './input'
) -> None:

    # Check input folder exists
    os.makedirs(output_path, exist_ok=True)

    # Check env variable has been set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(f"Environment variable 'OPENAI_API_KEY' must be set.")

    # Load model
    model = SentenceTransformer(st_model)

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
            ent2emb[label] = model.encode([response.choices[0].message.content], normalize_embeddings=True, show_progress_bar=False, convert_to_tensor=True).flatten()
    
    # Save tensors
    save_file(ent2emb, os.path.join(output_path, "ent2emb.safetensors"))


# Launch script
if __name__=='__main__':
    fire.Fire(main)