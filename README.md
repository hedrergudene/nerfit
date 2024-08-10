# nerFit: Few-shot entity recognition representation learning

## Table of Contents

1. [Introduction](#introduction)
2. [Description](#description)
3. [Methods](#methods)

## Introduction

This repository contains a contrastive Named Entity Recognition (NER) system designed to work with a classical NER setup. The project leverages state-of-the-art models from Hugging Face's `transformers` library and `SentenceTransformers` for entity description embeddings. This README provides an overview of the project's structure and functionalities.

## Description

The system is designed to identify and label entities in text using a contrastive learning approach. Unlike traditional NER systems where entity labels are directly predicted, this system uses embeddings to capture the semantic meaning of entity descriptions and computes similarities between token embeddings and entity description embeddings.

### Key Features

- **Classical NER Setup**: The system uses a finite set of labels with consistent descriptions, simplifying the data processing.
- **Contrastive Learning**: The model computes a contrastive loss to differentiate between positive and negative samples.
- **Pretrained Models**: Utilizes pretrained models from Hugging Face and SentenceTransformers for encoding and tokenization.

## Methods

### Data Processing

The dataset class (`nuNERDataset`) comprises most of the required operations to turn our natural language data into training inputs. However, a previous standarisation is demanded given the variety of NER datasets, in order to obtain a unified input with this format:

```
{
    'text': 'set an alarm for two hours from now',
    'entities: [17,35,'time']
}
```

The following example if based on Alexa massive dataset:

```python
def parse_annotation(annotation: str):
    pattern = re.compile(r'\[(.*?): (.*?)\]')
    matches = pattern.finditer(annotation)
    text = annotation
    entities = []
    offset = 0
    for m in matches:
        entity = m.group(2).strip()
        label = m.group(1)
        start_idx = m.start() - offset
        end_idx = start_idx + len(entity)
        entities.append([start_idx, end_idx, label])
        # Replace the annotated part with the entity name in the text
        annotated_text = m.group(0)
        text = text[:m.start()-offset] + entity + text[m.end()-offset:]
        # Update the offset to account for the removed annotation
        offset += len(annotated_text) - len(entity)
    return {
        "text": text,
        "entities": entities
    } 
```

This method has to be included in `nerfitDataset` class.

### Embedding lookup table

Next step is to build a mapping between entity labels and a vector representation. To that end, a description of the label should be provided to a sentence transformers model, that is already annotated in the data in case of zero-shot "open" datasets, and must be provided otherwise. The first case is trivial; the second is solved by using an LLM together with some samples, and it's implemented in `build_lookup_table.py`.

### Data Collator

The data collator (`NuNERDataCollator`) handles padding of input sequences and concatenation of entity embeddings. It constructs a padded label tensor for each batch.

### Model

The model (`BERTWithProjection`) wraps a BERT model with an additional projection layer to align token embeddings with entity description embeddings. It computes cosine similarities between token embeddings and entity embeddings, and a contrastive loss function is used for training.

### Training Loop

The training loop leverages the `Accelerate` library to handle distributed training. It includes steps for model evaluation, learning rate scheduling, and saving training metrics and the trained model.

## How to Use

1. **Install Dependencies**: Install the required libraries using `pip install -r requirements.txt`.
2. **Prepare Dataset**: Ensure the dataset is in the correct format and update the `annotations` variable accordingly.
3. **Run Training**: Execute the training script to train the NER model.
4. **Evaluate Model**: Use the evaluation functions to assess the model's performance on validation data.
5. **Inference**: Utilize the trained model to perform NER on new text data.

## Requirements

- `torch`
- `transformers`
- `sentence-transformers`
- `fuzzywuzzy`
- `accelerate`

Ensure all dependencies are installed before running the scripts.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss improvements or bugs.

## License

This project is licensed under the MIT License.
