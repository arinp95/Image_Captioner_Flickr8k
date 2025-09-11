# 🧠 Image Caption Generator using ResNet50 and LSTM

This project is an end-to-end deep learning solution that automatically generates natural language descriptions for images. It integrates a powerful feature extractor (DenseNet201) with a language modeling component (LSTM), trained on the **Flickr30k** dataset. The final model is deployed using a **Flask** web app for real-time image captioning.

---

## 📂 Project Structure
```
.
├── static/                   # Static assets (CSS, JS, images, etc.)
├── templates/                # HTML templates for the web app
├── .gitignore                # Git ignore rules
├── 01-data-preparation-8k.ipynb    # Data preprocessing notebook
├── 02-model-training-8k.ipynb      # Model training notebook
├── 03-model-evaluation-8k.ipynb    # Model evaluation notebook
├── app.py                    # Main FastAPI/Flask app script
├── caption_model.weights.h5  # Trained model weights (HDF5 format)
├── captions.pkl              # Preprocessed captions data
├── config.pkl                # Configuration parameters
├── features.pkl              # Extracted image features
├── generate_captions.py      # Script to generate captions from images
├── history.pkl               # Training history (loss/accuracy)
├── model.keras               # Saved Keras model
├── requirements.txt          # Python dependencies
├── test.pkl                  # Test dataset
└── tokenizer.pkl             # Tokenizer for text preprocessing

```

---

## 🧠 Model Architecture

### 🔹 Encoder (Feature Extractor)
- Pretrained **ResNet50** from Keras Applications
- Global Average Pooling applied to last convolutional layer output

### 🔹 Decoder (Language Generator)
- Embedding Layer initialized randomly
- LSTM Layer with 256 hidden units
- Dense Layer with ReLU followed by Softmax over vocabulary

The model is trained to predict the next word given the image embedding and previously generated words.

---

## 📈 Evaluation Metrics
Model performance is evaluated using BLEU-n metrics over a held-out validation set:
```
BLEU-1: 0.65
BLEU-2: 0.40
```
> Scores were calculated using 5 reference captions per image with NLTK’s BLEU implementation and smoothing function.

---

## 🌐 Deployment with Flask

A simple web UI is included for testing the model by uploading an image.

### Features
- Upload any image from your device
- Automatically view the predicted caption

### Run the App Locally
```bash
# Step 1: Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Launch app
python app.py
```

Then navigate to `http://127.0.0.1:5000` in your browser.

---

## 📦 Dependencies
- `tensorflow`
- `flask`
- `numpy`, `pandas`, `matplotlib`, `Pillow`
- `nltk`, `tqdm`

To install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🗃️ Dataset
- [Flickr30k Dataset](https://www.kaggle.com/datasets/eeshawn/flickr30k)

---

## 👤 Author
**Arindam Phatowali**  
B.Tech + M.Tech (Mathematics & Data Science), MANIT Bhopal

---

## 📬 Contact & Links
📧 Email: arindamphatowali@gmail.com  
🐍 GitHub: [github.com/arinp95](https://github.com/arinp95)

---

## ⭐ Like this project?
Please consider starring ⭐ the repository to show your support and help others discover it!


# 🖼️ Image Captioning with ResNet50 + LSTM

This project explores how **deep learning** can connect **computer vision** and **natural language processing** to describe images in natural language. The goal is simple but powerful: **given an image, automatically generate a meaningful caption**.

To achieve this, the project combines:
- **ResNet50 (CNN)** → to extract rich visual features from an image.
- **LSTM (RNN)** → to generate a descriptive sentence word by word.
- **Flickr8k dataset** → 8,000 images with 5 captions each, providing a diverse training ground.

On top of the model, we built a **FastAPI-powered web app**, where users can upload images and receive captions instantly.

---

## 📖 Problem Statement

Humans can easily look at an image and describe it: *“a dog running across a field”*. For a computer, this is challenging because it requires:
1. **Understanding the visual content** (objects, actions, relationships).
2. **Translating that understanding into language** (grammar, word order, fluency).

This project aims to bridge that gap by training a model that learns **visual–linguistic mappings**.

---

## 🔍 Approach

1. **Feature Extraction (Vision / Encoder)**
   - Uses **ResNet50** (a pretrained CNN) to process input images.
   - The network outputs a **2048-dimensional feature vector** summarizing the image.

2. **Caption Generation (Language / Decoder)**
   - An **LSTM network** takes the encoded features and sequentially predicts words.
   - Captions start with a `<start>` token and end with an `<end>` token.
   - During training, the model learns from human-written captions.

3. **Training Pipeline**
   - Captions are cleaned and tokenized.
   - Maximum caption length and vocabulary size are saved in `config.pkl`.
   - Extracted features (`features.pkl`) speed up training.
   - Training history (loss, accuracy) is logged in `history.pkl`.

4. **Evaluation**
   - Quality measured using **BLEU scores**, which compare generated captions with ground truth references.
   - BLEU-1 captures word overlap; BLEU-4 considers longer phrase overlaps.

5. **Deployment (Web App)**
   - A **FastAPI server** (`app.py`) powers the web interface.
   - Upload an image → model predicts caption → result displayed in the browser.

---

## 📂 Repository Structure

```
.
├── static/                   # Static assets (CSS, JS, uploaded images)
├── templates/                # HTML templates for the web app
├── .gitignore                 # Git ignore rules
├── 01-data-preparation-8k.ipynb    # Preprocessing images & captions
├── 02-model-training-8k.ipynb      # Model architecture & training loop
├── 03-model-evaluation-8k.ipynb    # Caption evaluation & BLEU scoring
├── app.py                     # FastAPI web server
├── caption_model.weights.h5   # Trained model weights
├── captions.pkl               # Preprocessed captions
├── config.pkl                 # Config (max caption length, vocab size, etc.)
├── features.pkl               # Precomputed image features
├── generate_captions.py       # Core CaptionGenerator class (ResNet50 + LSTM)
├── history.pkl                # Training history (loss/accuracy curves)
├── model.keras                # Full Keras model (saved format)
├── requirements.txt           # Project dependencies
├── test.pkl                   # Test dataset
└── tokenizer.pkl              # Tokenizer (word ↔ index mapping)
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/image-captioning.git
cd image-captioning
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Data & Pretrained Files
- **Dataset**: [Flickr8k](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k)
- Place the following in the project root:
  - `model.keras` or `caption_model.weights.h5`
  - `tokenizer.pkl`
  - `config.pkl`

---

## ▶️ Running the Application

Start the FastAPI server:
```bash
uvicorn app:app --reload
```

Open your browser and visit:
```
http://127.0.0.1:8000
```

1. Upload an image.  
2. The model extracts features with **ResNet50**.  
3. The **LSTM decoder** generates a caption.  
4. The caption is displayed alongside the uploaded image.  

---

## 🧠 Model Architecture

**Encoder (ResNet50):**
- Pretrained on ImageNet.
- Outputs 2048-d feature vector for each image.

**Decoder (LSTM):**
- Input: Encoded image + partial caption sequence.
- Embedding layer for word vectors.
- LSTM generates next word probabilities.
- Dense layer maps outputs to vocabulary size.

**Training Objective:**
- **Categorical Cross-Entropy Loss** between predicted and true next words.

---

## 📊 Example Results

| Input Image | Generated Caption |
|-------------|------------------|
| ![dog](https://i.imgur.com/jxWqIBf.jpg) | "a dog running across a grassy field" |
| ![kids](https://i.imgur.com/62Qp6uh.jpg) | "two children playing in the park" |
| ![beach](https://i.imgur.com/E8mQl2O.jpg) | "people walking along the beach" |

---

## 🔮 Future Improvements

- Use **beam search** instead of greedy decoding for more natural captions.
- Replace LSTM with **Transformers** (e.g., attention-based decoders).
- Train on larger datasets like **MS-COCO** for richer vocabulary.
- Deploy on **Docker, Hugging Face Spaces, or AWS Lambda** for easier access.

---

## 🙌 Acknowledgements

- Dataset: [Flickr8k](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k)
- Libraries: [TensorFlow/Keras](https://www.tensorflow.org/), [FastAPI](https://fastapi.tiangolo.com/)
- Pretrained ResNet50 from [Keras Applications](https://keras.io/api/applications/).

---

## 📜 License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute it with attribution.

