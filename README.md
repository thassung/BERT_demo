# S-BERT Demo
 
CMU book summary dataset from https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset/data
SNLI dataset from https://huggingface.co/datasets/snli
MNLI dataset from https://huggingface.co/datasets/glue/viewer/mnli

## Overview

   Welcome to the Sentence BERT Demo App! This web-based application demonstrates the basic functionality of a sentence BERT model, allowing users to input multiple (in this demo, two) sentences and compute its similarity.

## Dataset

   The BERT model is trained with CMU book summary data from [kaggle](https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset/data) and further trained with [SNLI dataset](https://huggingface.co/datasets/snli) and [MNLI dataset](https://huggingface.co/datasets/glue/viewer/mnli) from HuggingFace.

## Features

   - **Input Premised Sentence:** User can enter an English sentence (up to 512 words).
   - **Input Hypothesized Sentence:** User can enter another English sentence (up to 512 words).
   - **Submit Button:** User clicks *submit* after typing the prompt. The app will interpret whether the two above sentences are similar. The app will send a text output whether the two sentences are similar (entailment), neutral, or contradict. There are two output from two model. One is from a model trained from scratch from [ipynb in this repo](https://github.com/thassung/BERT_demo/blob/main/S-BERT%20copy.ipynb) and another is a pretrained model from [sbert.net](https://www.sbert.net/)

## Application

### Prerequisites

- Ensure you have Python installed

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/thassung/BERT_demo.git
   ```

2. Install the required Python dependencies:

   ```bash
   pip install flask torch spacy sklearn sentence_transformers
   ```

3. Navigate to the app directoty:
   ```bash
   cd BERT_demo/app
   ```

4. Start the flask application:
   ```bash
   python app.py
   ```

   You can access the application via http://localhost:8080

   Here is what the app looks like
   ![image](https://github.com/thassung/BERT_demo/assets/105700459/66d4292f-9af5-4119-bb64-411ce99028fa)

## Evaluation

The two model (one from scratch and one pretrained) are evaluated with the validation dataset with its label using Spearman correlation. The result is as following.

- My SBERT Spearman correlation:              0.0079             
- Pretrained SBERT Spearman correlation:      0.1344

The correlation from pretrained model is much higher. In fact, by observing test cases, my SBERT most likely to give a result of entailment which implies there is little to no variance in the latest embedding layer from my model. The problem can be the result of training dataset class imbalance.

*n_layers*, *n_heads*, d_ff, and *d_model* (number of encoder layers, number of attention heads, feed forward layers, and embedding size) can directly affect the amount of information a model can capture from the data. Increasing these parameters will increase the potential of the model but also risk overfitting, increase model's complexity, and would require more memory and computational power to train the model. *max_len* (max sequence length) also directly affects the capability of model to problem short or long sentences. But, higher *max_len* means bigger embedding layer and more memory and computational power are required to train the model.
