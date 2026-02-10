# DeepVision

MNIST Digit Recognizer with Flask & PyTorch

A web application that recognizes handwritten digits (0-9) using a Convolutional Neural Network (CNN) built with PyTorch and deployed via Flask.

Project Structure

mnist_project/
├── model.py           # Defines the Neural Network architecture
├── train.py           # Script to train the model and save weights
├── app.py             # Flask server for image processing & inference
├── requirements.txt   # List of dependencies
└── templates/
    └── index.html     # Frontend UI for image upload


Installation

Clone the repository (or create the folder structure manually).

Install dependencies:

pip install -r requirements.txt


Usage

1. Train the Model

Before running the app, you must generate the model weights file (mnist_cnn.pth).

python train.py


What this does: Downloads the MNIST dataset, trains the CNN for 3 epochs, and saves the learned parameters to mnist_cnn.pth.

2. Run the Web Application

Start the Flask development server:

python app.py


Access: Open your browser and go to http://127.0.0.1:5000.

3. Test

Upload an image of a handwritten digit.

Note: The app automatically handles image preprocessing (resizing to 28x28 pixels, converting to grayscale, and inverting colors to match MNIST format).

Model Details

Architecture: SimpleCNN (2 Convolutional Layers + 2 Fully Connected Layers).

Input: 28x28 Grayscale images.

Normalization: Mean=0.1307, Std=0.3081 (Standardizing data to improve training stability).

Requirements

Python 3.x

torch, torchvision (Deep Learning framework)

flask (Web framework)

pillow (Image processing library)
