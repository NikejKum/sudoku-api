import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import io

app = Flask(__name__)

# --- KNN MODEL (unverändert) ---
def create_knn():
    samples = []
    labels = []
    for digit in range(1, 10):
        for size in [1.5, 2.0, 2.5]: 
            for thickness in [2, 3]:
                img = np.zeros((50, 50), np.uint8)
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
def home(): return "Sudoku AI (Manual Crop)"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['file']
        img_bytes = np.frombuffer(io.BytesIO(file.read()).read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        # --- MANUELLE ECKEN VERARBEITEN ---
        # Wir erwarten einen Header "X-Sudoku-Points"
        # Format: "x1,y1,x2,y2,x3,y3,x4,y4" (Werte 0.0 bis 1.0, normalisiert)
        points_header = request.headers.get('X-Sudoku-Points')
        
        img_h, img_w = img.shape[:2]
        
        if points_header:
            print(f"Manuelle Punkte empfangen: {points_header}")
            try:
                # Koordinaten parsen
                vals = [float(x) for x in points_header.split(',')]
                
                # In Pixel umrechnen
                pts1 = np.float32([
                    [vals[0]*img_w, vals[1]*img_h], # TL
                    [vals[2]*img_w, vals[3]*img_h], # TR
                    [vals[4]*img_w, vals[5]*img_h], # BR
                    [vals[6]*img_w, vals[7]*img_h]  # BL
                ])
                
                # Ziel: Perfektes Quadrat
                pts2 = np.float32([[0, 0], [450, 0], [450, 450], [0, 450]])
                
                # Hardcore Warping
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                img_final = cv2.warpPerspective(img, matrix, (450, 450))
                
            except Exception as e:
                print(f"Fehler beim Warping: {e}")
                img_final = cv2.resize(img, (450, 450)) # Fallback
        else:
            # Fallback: Einfacher Crop (falls User nichts geklickt hat)
            crop = int(img_h * 0.1)
            img_final = img[crop:img_h-crop, crop:img_w-crop]
            img_final = cv2.resize(img_final, (450, 450))

        # --- AB HIER NORMAL WEITER ---
        gray = cv2.cvtColor(img_final, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 15, 2)
        
        # Gitterlinien entfernen (etwas aggressiver)
        kernel = np.ones((2,2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        grid = []
        cell_size = 50
        
                # ... (Loop startet hier)
        for row in range(9):
            for col in range(9):
                y1, y2 = row * cell_size, (row + 1) * cell_size
                x1, x2 = col * cell_size, (col + 1) * cell_size
                
                # ZUERST 'cell' definieren!
                cell = thresh[y1:y2, x1:x2]
                
                # JETZT können wir cell verwenden
                # Ränder großzügig entfernen (10px)
                # Sicherstellen, dass die Zelle groß genug ist
                if cell.shape[0] > 20 and cell.shape[1] > 20:
                    cell_inner = cell[10:-10, 10:-10]
                else:
                    cell_inner = cell # Fallback, falls Zelle winzig
                
                # Prüfen
                white_pixels = cv2.countNonZero(cell_inner)
                total_pixels = cell_inner.size
                
                if total_pixels == 0 or white_pixels < (total_pixels * 0.12):
                    grid.append(0)
                else:
                    # Erkennen
                    pts = cv2.findNonZero(cell_inner)
                    if pts is not None:
                        x, y, w, h = cv2.boundingRect(pts)
                        digit_crop = cell_inner[y:y+h, x:x+w]
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


