# Imports
# dealing with strings - captions
# os - get dataset from files
# pickles - training takes time so to not lose anything
import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, get_file  # for hadrcode encoding
from keras.layers import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout

# small library for seeing the progress of loops.
from tqdm import tqdm_notebook as tqdm

tqdm().pandas()

# Transfer Learning
# you have model - trained on something, trained feature used for other model
# downstream in finetuning - so in narrow dataset performance in better

# Input = images - converted to features, each image has some features
# Next part of input = caption sequence (text)
# Output = next work - character by character or word by word


# Loading a text file into memory
def load_doc(filename):
    # Opening the file as read only
    file = open(filename, "r")
    text = file.read()
    file.close()
    return text


# get all imgs with their captions
def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.split("\n")
    descriptions = {}
    for caption in captions[:-1]:
        img, caption = caption.split("\t")
        if img[:-2] not in descriptions:
            descriptions[img[:-2]] = [caption]
        else:
            descriptions[img[:-2]].append(caption)
    return descriptions


# Data cleaning- lower casing, removing puntuations and words containing numbers
def cleaning_text(captions):
    table = str.maketrans("", "", string.punctuation)
    for img, caps in captions.items():
        for i, img_caption in enumerate(caps):
            img_caption.replace("-", " ")
            desc = img_caption.split()

            # converts to lowercase
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [word.translate(table) for word in desc]
            # remove hanging 's and a
            desc = [word for word in desc if (len(word) > 1)]
            # remove tokens with numbers in them
            desc = [word for word in desc if (word.isalpha())]
            # convert back to string
            img_caption = " ".join(desc)
            captions[img][i] = img_caption
    return captions


# updating the vocabulary - assing each word a num for tokenization
def text_vocabulary(descriptions):
    # build vocabulary of all unique words
    vocab = set()

    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]

    return vocab


# Save all descriptions in one file
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + "\t" + desc)
    data = "\n".join(lines)
    file = open(filename, "w")
    file.write(data)
    file.close()


# Set these path according to project folder in you system
dataset_text = "/Users/darthwithap/Development/mlai/portfolio/image-captioning-model/data/Flickr8k_text"
dataset_images = "/Users/darthwithap/Development/mlai/portfolio/image-captioning-model/data/Flicker8k_Dataset"

# we prepare our text data
filename = dataset_text + "/" + "Flickr8k.token.txt"
# #loading the file that contains all data
# #mapping them into descriptions dictionary img to 5 captions
descriptions = all_img_captions(filename)
print("Length of descriptions =", len(descriptions))

# #cleaning the descriptions
clean_descriptions = cleaning_text(descriptions)

# #building vocabulary
vocabulary = text_vocabulary(clean_descriptions)
print("Length of vocabulary = ", len(vocabulary))

# #saving each description to file
save_descriptions(clean_descriptions, "descriptions.txt")

# Run the file to get descriptions.txt file
