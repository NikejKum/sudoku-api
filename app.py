import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import io

app = Flask(__name__)

# KNN Training (Unverändert robust)
def create_knn():
    samples = []
    labels = []
    for digit in range(1, 10):
        # Wir trainieren 3 Größen, um robust zu sein
        for size in [1.5, 2.0, 2.5]: 
            for thickness in [2, 3]:
                img = np.zeros((50, 50), np.uint8)
                # Text zentrieren
                (w, h), _ = cv2.getTextSize(str(digit), cv2.FONT_HERSHEY_SIMPLEX, size, thickness)
                x = (50 - w) // 2
                y = (50 + h) // 2
                cv2.putText(img, str(digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, 255, thickness)
                
                img_small = cv2.resize(img, (20, 20))
                samples.append(img_small.flatten())
                labels.append(digit)
    
    samples = np.array(samples, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    knn = cv2.ml.KNearest_create()
    knn.train(samples, cv2.ml.ROW_SAMPLE, labels)
    return knn

knn = create_knn()

@app.route('/', methods=['GET'])
def home(): return "Sudoku AI V2"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['file']
        img_bytes = np.frombuffer(io.BytesIO(file.read()).read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        # 1. Bild vorbereiten
        # 20% Rand abschneiden (gegen LEGO)
        h, w = img.shape[:2]
        border_h, border_w = int(h*0.2), int(w*0.2)
        img = img[border_h:h-border_h, border_w:w-border_w]
        
        # Auf 450x450 bringen
        img = cv2.resize(img, (450, 450))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Adaptive Threshold (macht Papier schwarz, Tinte weiß)
        # BlockSize 15 ist besser gegen Schatten
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 15, 2)
        
        # Rauschen entfernen (kleine Punkte weg)
        kernel = np.ones((2,2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        grid = []
        cell_size = 50
        
        for row in range(9):
            for col in range(9):
                y1, y2 = row * cell_size, (row + 1) * cell_size
                x1, x2 = col * cell_size, (col + 1) * cell_size
                
                # Zelle ausschneiden
                cell = thresh[y1:y2, x1:x2]
                
                # Ränder großzügig entfernen (7px), damit Gitterlinien weg sind
                cell_inner = cell[7:-7, 7:-7]
                
                # Check: Ist da überhaupt was?
                # Wir zählen weiße Pixel nur in der Mitte
                white_pixels = cv2.countNonZero(cell_inner)
                total_pixels = cell_inner.size
                
                # Schwellenwert erhöht auf 6% (filtert Schatten besser)
                if white_pixels < (total_pixels * 0.06):
                    grid.append(0)
                else:
                    # Wenn Inhalt da ist: Zentrieren und Erkennen
                    # Wir suchen das Bounding Rect der Ziffer
                    pts = cv2.findNonZero(cell_inner)
                    if pts is not None:
                        x, y, w, h = cv2.boundingRect(pts)
                        # Ziffer ausschneiden
                        digit_crop = cell_inner[y:y+h, x:x+w]
                        # Auf 20x20 skalieren (passend zum KNN)
                        digit_ready = cv2.resize(digit_crop, (20, 20))
                        
                        sample = digit_ready.flatten().reshape(1, -1).astype(np.float32)
                        ret, results, _, _ = knn.findNearest(sample, k=1)
                        grid.append(int(results[0][0]))
                    else:
                        grid.append(0)

        return jsonify({'status': 'success', 'grid': grid})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
