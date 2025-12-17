import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import io

# --- BLITZ-TEST VERSION (Ohne TensorFlow) ---
# Wir testen nur, ob das Bild ankommt und der Server antwortet.

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Sudoku BLITZ-TEST Ready"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
        
        # 1. Bild empfangen (Beweist, dass Upload klappt)
        file = request.files['file']
        in_memory_file = io.BytesIO(file.read())
        file_bytes = np.frombuffer(in_memory_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        print(f"Bild empfangen! Größe: {img.shape}")

        # 2. DUMMY-ANTWORT (Beweist, dass Server antwortet)
        # Wir tun so, als hätten wir erkannt, aber schicken sofort ein Muster zurück.
        # Das dauert 0.0 Sekunden.
        
        # Ein Muster: 1 bis 9, wiederholt
        grid = [1,2,3,4,5,6,7,8,9] * 9 

        print("Sende Dummy-Grid zurück...")
        return jsonify({'status': 'success', 'grid': grid})

    except Exception as e:
        print(f"SERVER ERROR: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
