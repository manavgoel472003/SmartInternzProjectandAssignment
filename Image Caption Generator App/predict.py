from tensorflow import keras
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

model = keras.models.load_model("/Users/manav/Downloads/flickr8k/final_model.h5")
cnn_model = Xception(weights="imagenet", include_top=False, pooling="max")

with open("/Users/manav/Downloads/flickr8k/tokenizer.pkl", 'rb') as f:
    tk = pickle.load(f)

def read_image(path):
  img = load_img(path, target_size=(240, 240))
  img = img_to_array(img)
  img /= 255.
  return img

def index_to_word(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
    if index == integer:
      return word
  return None

def predict_caption(model, image, tokenizer, max_length):
  in_text = 'startseq' # to start generation
  for i in range(max_length):
    sequence = tokenizer.texts_to_sequences([in_text])[0]
    sequence = pad_sequences([sequence], max_length)
    y_pred = model.predict([image, sequence])
    y_pred = np.argmax(y_pred)
    word = index_to_word(y_pred, tokenizer)
    if word is None:
      break
    in_text += " " + word
    if word == "endseq":
      break
  return in_text

def predict_cap(path):
    image = read_image(path)
    image = np.expand_dims(image, axis=0)
    feature = cnn_model.predict(image, verbose=0)
    y_pred = predict_caption(model, feature, tk, 34)  # found during testing phase
    y_pred = " ".join(y_pred.split(" ")[1:-1])
    return y_pred