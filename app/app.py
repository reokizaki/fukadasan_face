import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from .model import extract_features, cosine_similarity

# 基本設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_IMAGE_PATH = os.path.join(BASE_DIR, 'face.jpg')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

# Flask初期化
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 顔画像の特徴量を1度だけ抽出
face_features = extract_features(FACE_IMAGE_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    similarity = None
    uploaded_file_url = None

    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 類似度を計算
            uploaded_features = extract_features(filepath)
            similarity = cosine_similarity(face_features, uploaded_features)
            uploaded_file_url = os.path.join('static', 'uploads', filename)

    return render_template('index.html', similarity=similarity, image_path=uploaded_file_url)

if __name__ == '__main__':
    app.run(debug=True)
