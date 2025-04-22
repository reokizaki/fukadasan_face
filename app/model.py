# app/model.py
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np

# モデルのロードを遅延ロードするためのグローバル変数
model = None

def load_model():
    global model
    if model is None:
        base_model = MobileNetV2(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output)
    return model

def extract_features(img_path):
    model = load_model()  # モデルを遅延ロード
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features[0]

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)
