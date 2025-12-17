from flask import Flask, request, jsonify
import numpy as np
import cv2
import io
from PIL import Image

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Sudoku API is running!"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        img_bytes = file.read()
        
        # Bild einlesen (nur um zu prüfen, ob es klappt)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image'}), 400

        # --- HIER KOMMT SPÄTER DIE ECHTE KI/OCR REIN ---
        # Aktuell: Wir geben einfach ein hartcodiertes Test-Sudoku zurück
        # (Das gleiche aus deinem Buch)
        # 0 = leeres Feld
        detected_grid = [
            5, 3, 0, 0, 7, 0, 0, 0, 0,
            6, 0, 0, 1, 9, 5, 0, 0, 0,
            0, 9, 8, 0, 0, 0, 0, 6, 0,
            8, 0, 0, 0, 6, 0, 0, 0, 3,
            4, 0, 0, 8, 0, 3, 0, 0, 1,
            7, 0, 0, 0, 2, 0, 0, 0, 6,
            0, 6, 0, 0, 0, 0, 2, 8, 0,
            0, 0, 0, 4, 1, 9, 0, 0, 5,
            0, 0, 0, 0, 8, 0, 0, 7, 9
        ]
        
        return jsonify({
            'status': 'success',
            'grid': detected_grid
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
