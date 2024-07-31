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

The dataset class (`nuNERDataset`) precomputes embeddings for entity descriptions and creates a lookup embedding matrix. It uses exact and fuzzy matching to find entity positions in the text and constructs a binary label matrix indicating the presence of entities in the text.

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
