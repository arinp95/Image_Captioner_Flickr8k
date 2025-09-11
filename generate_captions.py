import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Optional
import pickle
import logging
import os

logger = logging.getLogger(__name__)

class CaptionGenerator:
    def __init__(self, model_path: str, tokenizer_path: str, config_path: str):
        self.resnet = ResNet50(include_top=False, pooling="avg", input_shape=(224, 224, 3))
        self.model = load_model(model_path)

        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        self.max_length = config["max_length"]
        self.index_word = {v: k for k, v in self.tokenizer.word_index.items()}
        logger.info("CaptionGenerator initialized")

    def extract_features(self, img_path: str) -> np.ndarray:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = self.resnet.predict(img_array, verbose=0)
        return features

    def generate_caption(self, photo: np.ndarray) -> str:
        in_text = 'startseq'
        for _ in range(self.max_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            yhat = self.model.predict([photo, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = self.index_word.get(yhat)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'endseq':
                break
        caption = in_text.replace('startseq', '').replace('endseq', '').strip()
        return caption
