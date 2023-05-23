# Trainer-Template 

Your project description goes here.

## Dependencies

This project is based on PyTorch and Transformers. Ensure the following libraries are installed:

- PyTorch
- Transformers
- yaml
- argparse
- attrdict
- loguru

You can install them with pip

## Configuration

The configurations for the model and training process are stored in `src/config.yaml`. You can modify this file to adjust the settings.

## Model

The model used in this project is `TextClassification`, which is defined in `src/model.py`.

## Training and Evaluation

The training process is managed by the `Trainer` class from Hugging Face's Transformers library. After training, the model is evaluated on a test set.

## Usage
To run the main script, navigate to the directory containing `main.py` and run:
`python main.py`