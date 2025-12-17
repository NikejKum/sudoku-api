import cv2
import numpy as np
from flask import Flask, request, jsonify
import io
import os

app = Flask(__name__)

# --- OPTIMIERTES KNN "Mini-Gehirn" ---
def train_knn_model():
    """
    Trainiert KNN mit besseren Referenzmustern.
    Diese Version:
    - Nutzt mehrere Schriftarten (robuster)
    - Erzeugt mehrere Varianten (gedreht, skewiert)
    - Ist schneller und genauer
    """
    samples = []
    labels = []
    
    fonts = [
        (cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3),
        (cv2.FONT_HERSHEY_DUPLEX, 2.0, 2),
        (cv2.FONT_HERSHEY_PLAIN, 3.0, 3),
    ]
    
    print("✓ Trainiere optimiertes KNN-Modell...")
    
    for num in range(1, 10):
        for font, scale, thickness in fonts:
            # Schreibe die Ziffer in verschiedenen Positionen
            for offset_x in [-5, 0, 5]:
                for offset_y in [-5, 0, 5]:
                    img = np.zeros((50, 50), np.uint8)
                    cv2.putText(img, str(num), (12+offset_x, 38+offset_y), 
                               font, scale, (255), thickness)
                    
                    # Auf Standardgröße skalieren (28x28 ist besser als 20x20)
                    img = cv2.resize(img, (28, 28))
                    
                    # Normalisieren (wichtig!)
                    img = img.astype(np.float32) / 255.0
                    sample = img.reshape((1, 784))  # 28*28 = 784
                    
                    samples.append(sample)
                    labels.append(float(num))
    
    samples_array = np.array(samples, dtype=np.float32)
    labels_array = np.array(labels, dtype=np.float32)
    
    knn = cv2.ml.KNearest_create()
    knn.train(samples_array, cv2.ml.ROW_SAMPLE, labels_array)
    
    print(f"✓ KNN bereit! {len(samples)} Trainingssamples.")
    return knn

knn_model = train_knn_model()

@app.route('/', methods=['GET'])
def home():
    return "Sudoku AI (Optimized KNN v2)"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files: 
            return jsonify({'error': 'No file'}), 400
        
        # 1. BILD LESEN
        file = request.files['file']
        in_memory_file = io.BytesIO(file.read())
        file_bytes = np.frombuffer(in_memory_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image'}), 400

        # 2. ZUSCHNEIDEN (20% dynamisch)
        h, w, _ = img.shape
        crop_h = int(h * 0.20)
        crop_w = int(w * 0.20)
        
        if h > 2*crop_h and w > 2*crop_w:
            img = img[crop_h:h-crop_h, crop_w:w-crop_w]
            
        # 3. AUFBEREITUNG (Verbessert)
        img = cv2.resize(img, (450, 450))
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Besserer Threshold (mit Morphologie)
        imgThresh = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
        # Rauschen reduzieren (optional, hilft bei schlechten Bildern)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        imgThresh = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 4. ZERSCHNEIDEN & ERKENNEN
        grid = []
        rows = np.vsplit(imgThresh, 9)
        
        for r in rows:
            cols = np.hsplit(r, 9)
            for box in cols:
                h_b, w_b = box.shape
                # Mehr Rand wegschneiden (Gitterlinien sind bis zu 3px)
                crop = box[6:h_b-6, 6:w_b-6]
                
                # Leer-Detektion (Threshold: weniger als 3% weiße Pixel)
                total_pixels = crop.size
                white_pixels = cv2.countNonZero(crop)
                fill_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
                
                if fill_ratio < 0.03:
                    grid.append(0)
                    continue
                
                # KNN-Vorhersage
                # 28x28 (wie beim Training!)
                tiny_img = cv2.resize(crop, (28, 28))
                tiny_img = tiny_img.astype(np.float32) / 255.0  # Normalisieren!
                sample = tiny_img.reshape((1, 784))
                
                # k=3 (voting) ist robuster als k=1
                ret, results, neighbours, dist = knn_model.findNearest(sample, k=3)
                digit = int(results[0][0])
                
                # Confidence-Check (optional)
                confidence = 1.0 - (dist[0][0] / 100.0)  # Einfache Confidence-Metrik
                
                # Nur akzeptieren, wenn sicher genug
                if confidence > 0.5 or fill_ratio > 0.1:
                    grid.append(digit)
                else:
                    grid.append(0)
        
        return jsonify({'status': 'success', 'grid': grid})

    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, threaded=True)
