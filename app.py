import cv2
import numpy as np
from flask import Flask, request, jsonify
import io
import os
import urllib.request

app = Flask(__name__)

# --- 1. INTELLIGENTES MODELL LADEN ---
knn = None

def load_smart_knn():
    global knn
    print("Lade intelligentes KNN-Modell...")
    
    # Wir laden einen kleinen Datensatz mit echten Ziffern (MNIST-ähnlich)
    # Da wir keine Datei haben, generieren wir ein besseres Set on-the-fly, 
    # aber diesmal mit Verzerrungen, um echte Kamera-Bilder zu simulieren.
    
    samples = []
    labels = []
    
    # Mehr Schriftarten, mehr Variationen
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX]
    
    for digit in range(1, 10):
        for font in fonts:
            for scale in [1.0, 1.5, 2.0, 2.5]:
                for thickness in [1, 2, 3, 4]:
                    # Basis-Bild
                    img = np.zeros((50, 50), np.uint8)
                    
                    # Text zentrieren (ungefähr)
                    text_size = cv2.getTextSize(str(digit), font, scale, thickness)[0]
                    text_x = (50 - text_size[0]) // 2
                    text_y = (50 + text_size[1]) // 2
                    
                    cv2.putText(img, str(digit), (text_x, text_y), font, scale, 255, thickness)
                    
                    # WICHTIG: Random Noise & Rotation simulieren (Robustheit!)
                    # Das macht das Modell schlau gegen schlechte Kameras
                    center = (25, 25)
                    for angle in [-10, 0, 10]:
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated = cv2.warpAffine(img, M, (50, 50))
                        
                        # Resize auf 20x20 (Standard für schnelle OCR)
                        small = cv2.resize(rotated, (20, 20))
                        
                        samples.append(small.flatten())
                        labels.append(digit)

    samples = np.array(samples, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    knn = cv2.ml.KNearest_create()
    knn.train(samples, cv2.ml.ROW_SAMPLE, labels)
    print(f"✓ Smart-KNN trainiert mit {len(samples)} Varianten.")

# --- 2. BILDVERARBEITUNGS-FUNKTIONEN ---

def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    # Adaptive Threshold ist der Schlüssel für Sudokus
    imgThresh = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    return imgThresh

def find_sudoku_contour(img):
    # Alles schwarz-weiß machen
    thresh = preProcess(img)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    biggest = np.array([])
    max_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000: # Muss groß genug sein
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest

def reorder_points(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

# --- 3. SERVER LOGIK ---

load_smart_knn()

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files: return jsonify({'error': 'no file'}), 400
        
        file = request.files['file']
        img_bytes = np.frombuffer(io.BytesIO(file.read()).read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        # 1. Bild verkleinern (Speed)
        img = cv2.resize(img, (640, 480)) # Standard VGA Größe nutzen
        
        # 2. Sudoku suchen (Kontur)
        contour = find_sudoku_contour(img)
        
        if contour.size != 0:
            # GEFUNDEN! -> Entzerren (Warp)
            contour = reorder_points(contour)
            pts1 = np.float32(contour)
            pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarp = cv2.warpPerspective(img, matrix, (450, 450))
            imgDisplay = imgWarp.copy() # Zum Arbeiten
        else:
            # NICHT GEFUNDEN -> Fallback (Mitte ausschneiden)
            h, w = img.shape[:2]
            crop = int(h*0.1) # 10% Rand weg
            imgWarp = img[crop:h-crop, crop:w-crop]
            imgWarp = cv2.resize(imgWarp, (450, 450))
            imgDisplay = imgWarp.copy()

        # 3. Bild für OCR vorbereiten
        imgWarpGray = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2GRAY)
        imgWarpThresh = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 11, 2)
        
        # 4. Gitter zerlegen & Erkennen
        grid = []
        rows = np.vsplit(imgWarpThresh, 9)
        
        for r in rows:
            cols = np.hsplit(r, 9)
            for box in cols:
                # Rand entfernen (Gitterlinien weg!)
                h_b, w_b = box.shape
                # Wir schneiden 6 Pixel Rand weg. Bei 50x50 Boxen bleibt 38x38
                crop = box[6:h_b-6, 6:w_b-6]
                
                # Prüfen ob leer
                white_pixels = cv2.countNonZero(crop)
                total_pixels = crop.size
                
                # Wenn weniger als 10% gefüllt -> Leer
                if white_pixels < (total_pixels * 0.1):
                    grid.append(0)
                else:
                    # Erkennen
                    small_img = cv2.resize(crop, (20, 20))
                    sample = small_img.reshape((1, 400)).astype(np.float32)
                    
                    ret, results, _, dist = knn.findNearest(sample, k=1)
                    digit = int(results[0][0])
                    grid.append(digit)

        return jsonify({'status': 'success', 'grid': grid})

    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
