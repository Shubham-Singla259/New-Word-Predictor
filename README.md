# 🌟 New Word Predictor Project 🌟

## Overview
Welcome to the **New Word Predictor**! This project leverages deep learning to predict the next word in a given sequence of text. Utilizing a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM), it generates human-like text. Additionally, the project features a user-friendly graphical interface built with Tkinter.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Features
- 📚 **Deep Learning:** Utilizes LSTM networks for accurate word prediction.
- 🖥️ **Graphical Interface:** Easy-to-use GUI for text input and predictions.
- 🛠️ **Customizable:** Train on any text dataset of your choice.
- 📊 **Evaluation Metrics:** Includes metrics like perplexity and accuracy.

## Installation

### Prerequisites
- Python 3.7 or above
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Keras
- Tkinter

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/new-word-predictor.git
   cd new-word-predictor
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Preparing the Dataset
1. Place your text dataset in the `data/` directory.
2. Run the preprocessing script to clean and tokenize the text:
   ```bash
   python preprocess.py
   ```

### Training the Model
1. Train the model by running the training script:
   ```bash
   python train.py
   ```
   This will save the trained model in the `models/` directory.

### Running the GUI Application
1. Start the GUI application:
   ```bash
   python app.py
   ```

### Generating Predictions
1. Enter the initial text in the input field and click the **"Generate Predictions"** button to see the generated text.

## Model Architecture
The model is built using an LSTM network, which is well-suited for sequence prediction tasks. The architecture consists of:
- **Embedding Layer:** Converts words into dense vectors.
- **LSTM Layers:** Captures temporal dependencies.
- **Dense Layer:** Uses softmax activation to predict the next word.

## Dataset
The model can be trained on any text corpus. Some commonly used datasets include:
- 📜 Shakespeare's works
- 📚 Text8 dataset (Wikipedia articles)
- 📝 Any custom text data relevant to your application

## Training
The training process involves:
- Tokenizing the text data
- Creating sequences of words
- Splitting the data into training and validation sets
- Training the LSTM model with a suitable optimizer and loss function

### Example Command
```bash
python train.py --epochs 10 --batch_size 64 --data_path data/your_dataset.txt
```

## Evaluation
Evaluate the model's performance using metrics like:
- 🔍 Perplexity
- ✔️ Accuracy on the validation set

### Example Command
```bash
python evaluate.py --model_path models/your_model.h5 --data_path data/your_dataset.txt
```

## Examples
Here are some examples of how to use the word predictor:

### Example 1
```bash
python predict.py --input "To be or not to be"
```

### Example 2
```bash
python predict.py --input "Once upon a time"
```
