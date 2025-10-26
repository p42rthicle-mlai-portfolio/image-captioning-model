# Image Captioning Model

This project demonstrates a from-scratch implementation of an Image Captioning model using deep learning concepts.
The model generates captions for images by combining CNNs for visual feature extraction and RNNs for sequence generation.

The implementation builds the pipeline entirely using TensorFlow and Keras, without relying on any pre-built captioning APIs for conceptual clarity

---

## Requirements

[tensorflow
keras
numpy
matplotlib
Pillow
tqdm
pickle
]

---

## Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Overview

1. Feature Extraction using a pre-trained CNN model (Xception).
2. Text Processing – cleaning, tokenization, and sequence generation.
3. Dataset Preparation – mapping image features to corresponding text captions.
4. Model Architecture – a custom encoder-decoder model combining CNN and LSTM.
5. Training – using a generator-based pipeline to feed large datasets efficiently.
6. Caption Generation – generating descriptive text for unseen images.

---

## Dataset
The dataset used is Flickr8k, which consists of:
~ 8,000 images paired with five captions each, describing visual content.
Captions are provided in the file Flickr8k.token.txt.
Training, validation, and test splits are defined in text files (Flickr_8k.trainImages.txt, etc.).

---

## Model Architecture

1. Feature Extraction (CNN Encoder)

The Xception model, pre-trained on ImageNet, is used as a feature extractor.
The final fully-connected layer is removed, and the output of the penultimate layer (a 2048-dimensional vector) is saved for each image.

Illustration:

<img width="550" height="295" alt="Image" src="https://github.com/user-attachments/assets/fd97516d-add4-45d5-af73-ed40bf6a1f37" />

---

2. Caption Processing
Each image is associated with multiple textual captions.
The text is:
Cleaned (lowercased, punctuation removed),
Tokenized (each word assigned a unique index),
Mapped into input-output pairs for supervised learning.

Each caption is represented as:

Input:   Image feature + partial sequence ("a cat on the")
Output:  next word ("mat")

Illustration: 

<img width="550" height="245" alt="Image" src="https://github.com/user-attachments/assets/b7f59aff-abbe-46c0-b272-52e012c1f16c" />

---

3. Combined Model (Encoder–Decoder)
The final model merges two inputs:
Input 1: Image feature vector from Xception (dense representation)
Input 2: Caption sequence (tokenized text)

These are concatenated and passed into an LSTM decoder to predict the next word.

Model Diagram: 

<img width="500" height="242" alt="Image" src="https://github.com/user-attachments/assets/3fa93837-ecdd-464f-8e36-9bb53267ab4a" />

---

## Key Concepts
Encoder–Decoder Architecture

CNN Encoder: Extracts image features.

LSTM Decoder: Generates the sequence of words.

Sequence-to-Sequence Learning

Each training step predicts the next word given the image and previous words.

Loss Function

Categorical Crossentropy between predicted and actual word indices.

Optimization

Adam optimizer with gradient updates on both text and image embeddings.

---

## Training Procedure

The dataset is large and cannot fit in memory, so a data generator is used:
```
dataset = data_generator(descriptions, features, tokenizer, max_length)
model.fit(dataset, epochs=10, steps_per_epoch=steps, verbose=1)
```

Model checkpoints are saved after every epoch.
Loss convergence is visualized to monitor learning stability.

Output example (training log):
```
Epoch 1/10
400/400 [==============================] - 310s - loss: 4.723
...
Epoch 10/10
400/400 [==============================] - 275s - loss: 2.891
```

---

## Results

Once trained, the model can generate meaningful captions for unseen images:

Image	

<img width="500" height="280" alt="Image" src="https://github.com/user-attachments/assets/82faf97f-688c-4518-8c17-d642e9a12ffe" />

Predicted Caption:

A dog running in a park	“a brown dog is running through field”
<img width="500" height="120" alt="Image" src="https://github.com/user-attachments/assets/f8a8a2e3-3414-4302-9203-1c647a171225" />

---

## References

Dataset: Flickr8k

Base CNN: Xception (Keras Applications)

Frameworks: TensorFlow, Keras, NumPy

---

## Acknowledgment

This project was developed as part of an academic demonstration to understand the internal mechanics of deep learning–based image captioning systems.
It is nowhere close to production grade, lots of scope of improvement.
