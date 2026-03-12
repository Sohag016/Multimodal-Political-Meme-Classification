# Multimodal-Political-Meme-Classification
A deep learning based multimodal classification system that detects whether a meme is Political or NonPolitical by combining image features and text extracted from the meme.

This project was developed for the Poli Meme Decode Contest (CUET CSE Fest).

The model integrates:

Vision Transformer (ViT) for image understanding

DistilBERT for text representation

Tesseract OCR for extracting text from meme images

Multimodal feature fusion for final classification

Project Overview

Political memes often contain important information both in the image and the text written on the image.
Using only image or only text often leads to poor results.

This project uses a multimodal deep learning architecture that combines both modalities to improve prediction accuracy.

Pipeline:

Input Meme Image
        │
        ▼
     OCR (Tesseract)
        │
        ▼
   Extracted Text
        │
        ▼
  DistilBERT Encoder
        │
        ▼
    Text Feature
        │
        │
Image → Vision Transformer (ViT)
        │
        ▼
    Image Feature
        │
        ▼
   Feature Concatenation
        │
        ▼
      Classifier
        │
        ▼
 Political / NonPolitical
Model Architecture

The system uses two pretrained deep learning models.

Image Encoder

Model:
vit_base_patch16_384

Library:
timm

Purpose:

Extract visual patterns from meme images

Capture political symbols, faces, and scene context

Text Encoder

Model:
distilbert-base-uncased

Used for:

Encoding OCR extracted meme text

Generating contextual text embeddings

To reduce training cost, the text encoder is frozen during training.

Feature Fusion

Image and text features are projected into the same dimension.

Image Feature → Linear → 512
Text Feature → Linear → 512

Then concatenated:

512 + 512 = 1024
Classifier

Final classification network:

Linear (1024 → 256)
ReLU
Dropout (0.2)
Linear (256 → 2)

Output classes:

Political

NonPolitical

Dataset

Dataset used in this project:

Poli Meme Decode Dataset (CUET CSE Fest)

Dataset structure:

Dataset
│
├── Train
│   ├── Image
│   └── Labels
│
└── Test
    ├── Image

Label mapping:

Political = 1
NonPolitical = 0
OCR Processing

Memes usually contain text embedded in images.

We use Tesseract OCR to extract the text.

Pipeline:

Image → OCR → Extracted Text → DistilBERT

To speed up training, OCR results are cached using pickle.

Cache files:

ocr_train.pkl
ocr_test.pkl
Training Configuration
Parameter	Value
Image Size	384
Batch Size	16
Epochs	5
Learning Rate	2e-4
Optimizer	Adam
Seed	42

Device:

CUDA if available
otherwise CPU
Model Performance

Training result:

Train F1 Score = 0.79
Validation F1 Score = 0.62

Multimodal learning improves performance compared to single modality models.

Inference Pipeline

Steps:

1 Load trained model
2 Extract OCR text
3 Encode text using DistilBERT
4 Extract image features using ViT
5 Fuse text and image features
6 Predict label

Output file:

submission.csv

Format:

filename,label
image1.jpg,Political
image2.jpg,NonPolitical
Installation

Clone the repository:

git clone https://github.com/yourusername/multimodal-meme-classification.git
cd multimodal-meme-classification

Install dependencies:

pip install torch torchvision
pip install transformers
pip install timm
pip install pytesseract
pip install albumentations
pip install pandas numpy scikit-learn

Install Tesseract OCR.

Ubuntu:

sudo apt install tesseract-ocr
Running the Project

Train the model:

python train.py

Run inference:

python inference.py

Generate prediction file:

submission.csv
Project Structure
multimodal-meme-classification
│
├── train.py
├── inference.py
├── dataset.py
├── model.py
│
├── best_model.pth
│
├── ocr_train.pkl
├── ocr_test.pkl
│
└── README.md
Future Improvements

Possible improvements:

CLIP based multimodal architecture

Cross attention feature fusion

Advanced OCR text cleaning

Larger dataset training

Transformer based multimodal fusion

Author

Md. Sohag Hossain
CSE Student
Machine Learning and Data Science Enthusiast

Acknowledgements

CUET CSE Fest

PyTorch

HuggingFace Transformers

TIMM Vision Models

Tesseract OCR
