import os
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from app.model import (
    extract_features,
    cosine_similarity,
    extract_features_of_face_parts,
    face_part_similarity,
    extract_face_landmarks
)

# 基本設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_IMAGES_DIR = os.path.join(BASE_DIR, 'face')  # 参照画像が格納されているディレクトリ
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

# Flask初期化
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# アップロードを許可する拡張子
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# ファイル拡張子のチェック関数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 複数の顔画像から特徴量を抽出して平均化
def extract_features_from_multiple_images(img_dir):
    features_list = []
    for filename in os.listdir(img_dir):
        img_path = os.path.join(img_dir, filename)
        if os.path.isfile(img_path):
            # 特徴量抽出処理
            features = extract_features(img_path)
            features_list.append(features)
    # 複数の特徴量の平均を取る
    if len(features_list) > 0:
        average_features = np.mean(features_list, axis=0)
        return average_features
    return None

# 複数の顔画像から顔パーツ特徴量を抽出して平均化
def extract_face_parts_from_multiple_images(img_dir):
    parts_list = []
    for filename in os.listdir(img_dir):
        img_path = os.path.join(img_dir, filename)
        if os.path.isfile(img_path):
            parts = extract_features_of_face_parts(img_path)
            if parts:
                parts_list.append(parts)
    
    if not parts_list:
        return None

    # 各パーツごとに平均を取る
    averaged_parts = {}
    part_keys = parts_list[0].keys()
    for key in part_keys:
        part_vectors = [d[key] for d in parts_list if key in d]
        averaged_parts[key] = np.mean(part_vectors, axis=0)

    return averaged_parts

# 全体特徴量とパーツ特徴量の両方を事前に定義
face_features = extract_features_from_multiple_images(FACE_IMAGES_DIR)
face_features_parts = extract_face_parts_from_multiple_images(FACE_IMAGES_DIR)

@app.route('/', methods=['GET', 'POST'])
def index():
    similarity = None
    face_part_similarity_scores = None  # 顔パーツの類似度
    uploaded_file_url = None
    error_message = None  # エラーメッセージ用の変数

    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename:
            if allowed_file(file.filename):  # ファイルが許可された形式であることを確認
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # 顔が検出されるかチェック
                landmarks_list = extract_face_landmarks(filepath)
                if not landmarks_list:  # 顔が検出できなかった場合
                    similarity = 0  # 類似度を0%に設定
                    error_message = "似顔絵か写真をアップロードしてください"
                else:
                    # 類似度を計算（顔が検出された場合）
                    uploaded_features = extract_features(filepath)
                    similarity_raw = cosine_similarity(face_features, uploaded_features)
                    similarity = (similarity_raw + 1) / 2  # -1〜1 → 0〜1 に変換

                    # 顔パーツの類似度を計算
                    uploaded_face_parts = extract_features_of_face_parts(filepath)
                    if uploaded_face_parts:
                        # 顔パーツごとの類似度を計算
                        face_part_similarity_scores = face_part_similarity(face_features_parts, uploaded_face_parts)
            else:
                error_message = "許可されていないファイル形式です。画像ファイル（png, jpg, jpeg, gif）をアップロードしてください。"

            uploaded_file_url = os.path.join('static', 'uploads', filename)

    return render_template(
        'index.html',
        similarity=similarity,
        image_path=uploaded_file_url,
        face_part_similarity_scores=face_part_similarity_scores,
        error_message=error_message  # エラーメッセージをテンプレートに渡す
    )

if __name__ == '__main__':
    app.run(debug=True)
