# British Sign Language (BSL) Recognition System

This project implements a real-time British Sign Language (BSL) recognition system using MediaPipe and a Multilayer Perceptron model.
This is the page for the One-Hand BSL Model for the numbers zero to ten

## ⚙️ Requirements

- Python 3.10 (recommended)
- `pip` (Python package installer)

> ⚠️ Note: The project was developed using Python 3.10. If you have multiple versions of Python installed, be sure to use Python 3.10 specifically.

## 🔧 Setup Instructions

### 1. Create a virtual environment using Python 3.10

In VSCode terminal:

py -3.10 -m venv venv
venv\Scripts\activate

### 2. Install the required librairies

pip install -r requirements.txt

### 3. Run the model training

python model.py

### 4. Run the code to evaluate the model generated

python evaluate.py

### 5. Run the main BSL-to-text prototype

python app.py
