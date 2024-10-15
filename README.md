# RNN-based Language Model using PyTorch

This project implements a Recurrent Neural Network (RNN) based language model using Long Short-Term Memory (LSTM) in PyTorch. The model is trained on the Penn Treebank dataset and generates text by sampling from the learned language model.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Overview](#model-overview)
- [Running the Code](#running-the-code)
- [Text Generation](#text-generation)
- [References](#references)

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/shubham-dayma/language_model_pytorch.git
    cd language_model_pytorch
    ```

2. **Install required dependencies**:
    Make sure you have [PyTorch](https://pytorch.org/get-started/locally/) installed. You can install other dependencies via:

    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare the dataset**:
    - Downloaded the Penn Treebank dataset and place the training text file (`train.txt`) in the `data/` directory:
      ```bash
      mkdir data
      # Add your `train.txt` file here
      ```

## Dataset

This project uses the [Penn Treebank dataset](https://catalog.ldc.upenn.edu/LDC99T42), a popular dataset for language modeling tasks.

- Place the dataset text file (`train.txt`) in the `data/` directory.
- The data is processed using a `Corpus` class, which converts text to word IDs and helps batch the data efficiently.

## Model Overview

The language model is built using a Recurrent Neural Network (RNN) with LSTM units:

- **Embedding Layer**: Maps word IDs to dense vectors (embedding size = 128).
- **LSTM Layer**: Single LSTM layer with 1024 hidden units to learn temporal dependencies.
- **Linear Layer**: Maps the hidden state output of the LSTM to the vocabulary size for predicting the next word.

### Hyperparameters:
- Embedding Size: 128
- Hidden Size: 1024
- Number of Layers: 1
- Sequence Length: 30
- Batch Size: 20
- Learning Rate: 0.002
- Number of Epochs: 5

## Running the Code

The code to train the model and generate text is contained in a single script.

1. **Train the model**:
    Simply run the main Python script:

    ```bash
    python main.py
    ```

    During training, the script will output the following metrics for every 100 steps:
    - **Loss**: Cross-entropy loss between predicted and true words.
    - **Perplexity**: A measure of how well the model predicts the next word, calculated as `exp(loss)`.

    Example output:
    ```
    Epoch [1/5], Step [100/500], Loss: 5.3467, Perplexity: 210.99
    Epoch [1/5], Step [200/500], Loss: 4.9812, Perplexity: 145.73
    ...
    ```

2. **Checkpoints**:
    The model’s checkpoints are saved to `model.ckpt` at the end of training.

## Text Generation

After training, the script generates text by sampling from the trained model. 

The generated text is written to a file called `sample.txt`.

1. **How it works**:
    - The script randomly selects a starting word and iteratively predicts the next word based on the hidden state of the RNN.
    - Words are sampled from the output probability distribution of the network using `torch.multinomial`.

2. **Example usage**:
    Simply run the script as mentioned in [Running the Code](#running-the-code). After training, the model will generate text and save it in the file `sample.txt`. Every 100 words generated will be logged to the console:

    ```bash
    Sampled [100/1000] words and saved to sample.txt
    ```

3. **Generated text**:
    The generated text will be in `sample.txt`. Words are sampled until the specified `num_samples` limit is reached.

## References

- Parts of this code were referenced from the [PyTorch word language model example](https://github.com/pytorch/examples/tree/master/word_language_model).