import os

# --- WICHTIG: Speicher-Begrenzung für Render Free Tier ---
# Muss VOR dem Import von TensorFlow passieren!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import io
import gc  # Garbage Collector für Speicher-Bereinigung

app = Flask(__name__)

# --- Modell laden ---
print("Starte Server (Low-Memory Mode)...")
try:
    model_path = 'digit_model.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Modell erfolgreich geladen!")
    else:
        print(f"WARNUNG: {model_path} nicht gefunden!")
        model = None
except Exception as e:
    print(f"KRITISCHER FEHLER: {e}")
    model = None

@app.route('/', methods=['GET'])
def home():
    return "Sudoku AI (Safe Mode) Ready"

@app.route('/analyze', methods=['POST'])
def analyze():
    if not model: 
        return jsonify({'grid': [0]*81, 'status': 'error_no_model'})
    
    try:
        if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
        
        # 1. Bild lesen
        file = request.files['file']
        in_memory_file = io.BytesIO(file.read())
        file_bytes = np.frombuffer(in_memory_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # 2. ZUSCHNEIDEN (Dynamisch 20% Rand weg)
        # Das passt für VGA, QVGA und alles andere
        h, w, _ = img.shape
        crop_h = int(h * 0.20)
        crop_w = int(w * 0.20)
        
        if h > 2*crop_h and w > 2*crop_w:
            img = img[crop_h:h-crop_h, crop_w:w-crop_w]
            
        # 3. AUFBEREITUNG
        img = cv2.resize(img, (450, 450))
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgThresh = cv2.adaptiveThreshold(imgGray, 255, 1, 1, 11, 2)
        
        # 4. ZERSCHNEIDEN & ERKENNEN (Schritt für Schritt)
        grid = []
        rows = np.vsplit(imgThresh, 9)
        
        print("Starte Einzel-Erkennung...")
        
        for r in rows:
            cols = np.hsplit(r, 9)
            for box in cols:
                # Bild vorbereiten
                h_box, w_box = box.shape
                crop = box[4:h_box-4, 4:w_box-4]
                crop = cv2.resize(crop, (28, 28))
                
                # Einzeln vorhersagen (schont RAM)
                blob = crop.reshape(1, 28, 28, 1) / 255.0
                prediction = model.predict(blob, verbose=0)
                
                classIndex = np.argmax(prediction)
                prob = np.amax(prediction)
                
                if prob > 0.7:
                    grid.append(int(classIndex))
                else:
                    grid.append(0)
        
        # Speicher aufräumen
        gc.collect()
        
        print("Fertig!")
        return jsonify({'status': 'success', 'grid': grid})

    except Exception as e:
        print(f"SERVER ERROR: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
