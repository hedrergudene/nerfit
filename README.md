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

- **Classical NER Setup**: The system utilizes a predefined set of entity labels, each with a consistent and comprehensive description, simplifying the data processing and model training.
- **Contrastive Learning**: Instead of directly predicting labels, the model computes a contrastive loss to differentiate between positive and negative samples, improving the accuracy of entity recognition.
- **Pretrained Models**: The system leverages pretrained models from Hugging Face and SentenceTransformers for efficient and accurate encoding and tokenization, ensuring state-of-the-art performance.


## Methods

The `Trainer` object is responsible for managing the entire training pipeline, from data preparation to model optimization and evaluation. It encapsulates all the necessary components, including the model, tokenizer, data loaders, optimizer, and training loop. The `Trainer` is configured via the `TrainerConfig` class, which allows you to specify various parameters such as learning rates, batch size, and more.

However, given the rich variety of NER dataset formats available, it's not possible to rule them all. Instead, you have to prepare a small block of code to convert your annotations into the following schema, and encapsulate it into the `_parse_annotation` static method within the `Trainer`:

```
{
    'text': 'set an alarm for two hours from now',
    'entities: [17,35,'time']
}
```

Here you can find some templates that cover most of the NER dataset templates:

<details>
<summary>
[Alexa massive dataset](https://huggingface.co/datasets/AmazonScience/massive):
</summary>

Annotations have this structure:

```text
[ORG: OpenAI] is based in [LOC: San Francisco].
```

Therefore, `_parse_annotation` method should be like:

```python
# Libraries
import re
from typing import List, Union, Dict
from nerfit import Trainer

# Child class
class CustomTrainer(Trainer):
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
    ) -> List[Dict[str,Union[str,int]]]:
        output = []
        for annotation in annotations:
            pattern = re.compile(r'\[(.*?): (.*?)\]')
            matches = pattern.finditer(annotation)
            text = annotation
            entities = []
            offset = 0
            for m in matches:
                entity = m.group(2).strip()
                label = m.group(1).strip()
                start_idx = m.start() - offset
                end_idx = start_idx + len(entity)
                entities.append([start_idx, end_idx, label])
                # Replace the annotated part with the entity name in the text
                annotated_text = m.group(0)
                text = text[:m.start()-offset] + entity + text[m.end()  -offset:]
                # Update the offset to account for the removed annotation
                offset += len(annotated_text) - len(entity)
            output.append({"text": text,"entities": entities})
        return output
```
</details>


## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss improvements or bugs.

## License

This project is licensed under the MIT License.