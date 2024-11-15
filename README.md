# Image Caption Generation using ResNet50 and LSTM
This repository implements an image caption generation system using ResNet50 as a feature extractor and Long Short-Term Memory (LSTM) networks for generating natural language captions. The goal of this project is to automate the generation of meaningful and contextually relevant captions for images. The model is trained and evaluated using the Flickr8k dataset, which consists of 8,000 images, each paired with five captions.

### Dataset
The model is trained and evaluated using the Flickr8k dataset, which contains 8,000 images with 5 captions for each image. You can download the dataset from the following link:
[https://www.kaggle.com/datasets/adityajn105/flickr8k]

### Packages required to run the project:

- tensorflow (for model building and training)
- keras (for deep learning functionalities)
- numpy (for numerical operations)
- pandas (for data handling)
- matplotlib (for plotting and visualizations)
- PIL or Pillow (for image processing)
- h5py (for saving model weights)
- scikit-learn (for splitting the dataset)

### Model Architecture
The model consists of the following two primary components:
ResNet50: 
- Pre-trained on the ImageNet dataset, ResNet50 is used to extract features from the input images.
- The output of ResNet50 is a high-dimensional feature vector that represents the visual information within an image.

![Resnet50](https://github.com/user-attachments/assets/5176860c-466c-49ac-a960-fc37f17a91bc)

LSTM:
- The image features extracted by ResNet50 are fed into an LSTM network.
- The LSTM network processes the sequential data and generates a caption by predicting the next word at each timestep based on the context of previous words.
- 
![1_laH0_xXEkFE0lKJu54gkFQ](https://github.com/user-attachments/assets/1578a3bd-54bf-4d92-83aa-1a1d95758a8f)

### Methodology for Image Captioning
1. Data Preprocessing
- Image Feature Extraction: Extract high-level features from each image using the ResNet50 model.
- Text Preprocessing: Tokenize the captions, create vocabulary, and pad the sequences for LSTM input.
- Train-Test Split: Split the dataset into training and validation sets.
- Data Generator: Create a data generator to efficiently feed the data into the model during training.
2. Encoder-Decoder Architecture
- Encoder: The ResNet50 model acts as the encoder, extracting features from input images.
- Decoder: The LSTM model acts as the decoder, generating captions word by word based on the encoded features.

![LSTM](https://github.com/user-attachments/assets/3491b532-4f9e-4938-9cd5-b5ff47b2aee7)

![Screenshot 2024-11-12 000157](https://github.com/user-attachments/assets/60f05d9e-e948-45db-ae66-4fd52fc2e761)

3. Training & Optimization
- Train the model for several epochs, optimizing for loss using techniques such as EarlyStopping and ModelCheckpoint.
- Evaluate the model's performance using standard captioning metrics like BLEU-1 and BLEU-2.

![Screenshot 2024-11-15 231829](https://github.com/user-attachments/assets/0800fd87-9b83-41cb-a46b-9af1c8029b19)

4. Frontend
- A Streamlit frontend is provided for generating captions from new images. This allows users to interact with the trained model easily.
### Results:
The model was trained for 20 epochs and achieved the following:
- Training Loss: 2.5493
- BLEU-1 Score: 0.5607
- BLEU-2 Score: 0.3421

![Screenshot 2024-11-12 005408](https://github.com/user-attachments/assets/f472f987-3481-4e11-b1a6-be6cf724808d)

![Screenshot 2024-11-12 005516](https://github.com/user-attachments/assets/f7240430-05d6-43c9-9a7c-eeecefc30d37)
