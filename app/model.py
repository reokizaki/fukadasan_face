import dlib
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# 顔検出器とランドマーク予測器の初期化
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('app/models/shape_predictor_68_face_landmarks.dat')

# MobileNetV2モデルをグローバルで遅延ロード
model = None
def load_model():
    global model
    if model is None:
        base_model = MobileNetV2(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output)
    return model

# 顔ランドマークを抽出
def extract_face_landmarks(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    landmarks_list = []

    for face in faces:
        landmarks = landmark_predictor(gray, face)
        coords = np.array([[p.x, p.y] for p in landmarks.parts()])
        landmarks_list.append(coords)

    return landmarks_list

# 顔のパーツ特徴量を抽出（目、鼻、口、輪郭）
def extract_features_of_face_parts(img_path):
    landmarks_list = extract_face_landmarks(img_path)
    if not landmarks_list:
        return None

    landmarks = landmarks_list[0]  # 最初の顔だけを使用

    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    nose = landmarks[27:36]
    mouth = landmarks[48:60]
    face_outline = landmarks[0:17]

    # 各パーツの特徴を数値で表現
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    nose_shape = np.mean(nose, axis=0)
    mouth_width = np.linalg.norm(mouth[0] - mouth[6])
    face_width = np.linalg.norm(face_outline[0] - face_outline[16])

    return {
        "left_eye_center": left_eye_center,
        "right_eye_center": right_eye_center,
        "nose_shape": nose_shape,
        "mouth_width": mouth_width,
        "face_width": face_width,
    }

# 顔パーツごとの類似度（差）を計算
def face_part_similarity(a, b):
    return {
        "eye_similarity": np.linalg.norm(a['left_eye_center'] - b['left_eye_center']) + np.linalg.norm(a['right_eye_center'] - b['right_eye_center']),
        "nose_similarity": np.linalg.norm(a['nose_shape'] - b['nose_shape']),
        "mouth_similarity": abs(a['mouth_width'] - b['mouth_width']),
        "face_similarity": abs(a['face_width'] - b['face_width']),
    }

# モバイルネットによる画像特徴量抽出
def extract_features(img_path):
    model = load_model()
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features[0]

# 2つの特徴ベクトル間のコサイン類似度を計算
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))