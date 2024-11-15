# Image Caption Generation using ResNet50 and LSTM
This repository contains the implementation of an image caption generation system that uses ResNet50 as a feature extractor and Long Short-Term Memory (LSTM) networks for generating captions. The goal of the project is to automate the process of generating meaningful and contextually relevant captions for images. The model is trained and evaluated using the Flickr8k dataset, which consists of 8,000 images, each with five corresponding captions.

## Dataset
The model is trained and evaluated using the Flickr8k dataset, which contains 8,000 images with 5 captions for each image. You can download the dataset from the following link:
https://www.kaggle.com/datasets/adityajn105/flickr8k

## Packages required to run the project:

tensorflow (for model building and training)
keras (for deep learning functionalities)
numpy (for numerical operations)
pandas (for data handling)
matplotlib (for plotting and visualizations)
PIL or Pillow (for image processing)
h5py (for saving model weights)
scikit-learn (for splitting the dataset)

## Model Architecture
The model consists of the following two primary components:

ResNet50:

Pre-trained on the ImageNet dataset, ResNet50 extracts features from the input images.
The output of the ResNet50 network is a high-dimensional feature vector representing the visual information in the image.
LSTM:

The extracted image features from ResNet50 are passed into an LSTM network.
The LSTM processes the sequential data and generates a caption by predicting the next word at each timestep based on the context of the previous words.
The entire system is trained end-to-end, meaning both ResNet50 and LSTM are fine-tuned during the training process.

## Methodology for Image Captioning
1. Data Preprocessing
Extract image features
Text preprocessing
Train-Test split
Data generator
2. Encoder-Decoder Architecture
Load VGG16 model
Encoder : Image feature layer Sequence feature layer
Decoder
3. Training & Optimization
Training model
Evaluation of model
4. Frontend
User interface using streamlit.
## Results:
The Resnet50-LSTM model was trained for 20 epochs, achieving a low training loss of 2.5493.
Evaluated the model using the BLEU score, with a focus on BLEU-1 score (0.560692).
