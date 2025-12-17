import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import io
import os

app = Flask(__name__)

# --- Modell laden ---
print("Starte Server & lade Modell...")
try:
    # Hier der korrekte Dateiname mit Unterstrich!
    model = load_model('digit_model.h5')
    print("Modell erfolgreich geladen!")
except Exception as e:
    print(f"WARNUNG: Modell-Fehler: {e}")
    model = None

def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    return imgThreshold

def splitBoxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            h, w = box.shape
            crop = box[4:h-4, 4:w-4]
            crop = cv2.resize(crop, (28, 28))
            boxes.append(crop)
    return boxes

def getPredection(boxes, model):
    result = []
    if model is None: return [0]*81 # Fallback
    
    boxes_array = np.array(boxes)
    boxes_array = boxes_array.reshape(boxes_array.shape[0], 28, 28, 1)
    boxes_array = boxes_array / 255.0
    
    # Vorhersage
    predictions = model.predict(boxes_array)
    
    for i in range(81):
        prob = np.amax(predictions[i])
        classIndex = np.argmax(predictions[i])
        if prob > 0.6: # Etwas toleranter
            result.append(int(classIndex))
        else:
            result.append(0)
    return result

@app.route('/', methods=['GET'])
def home():
    return "Sudoku AI Ready (Light Version)"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
        
        file = request.files['file']
        in_memory_file = io.BytesIO(file.read())
        file_bytes = np.frombuffer(in_memory_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # 1. Bild verarbeiten
        img = cv2.resize(img, (450, 450))
        
        # 2. Wir nehmen an, das Bild vom ESP32 ist schon grob zugeschnitten 
        # oder wir nutzen das ganze Bild, da wir die Konturen sparen wollen für Speed.
        # (Für beste Ergebnisse: Kamera nah ranhalten!)
        
        imgThreshold = preProcess(img)
        
        # Versuchen Konturen zu finden (Sudoku ausschneiden)
        contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest = np.array([])
        max_area = 0
        
        for i in contours:
            area = cv2.contourArea(i)
            if area > 50:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area

        if biggest.size != 0:
            # Wenn Gitter gefunden -> Entzerren
            def reorder(myPoints):
                myPoints = myPoints.reshape((4, 2))
                myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
                add = myPoints.sum(1)
                myPointsNew[0] = myPoints[np.argmin(add)]
                myPointsNew[3] = myPoints[np.argmax(add)]
                diff = np.diff(myPoints, axis=1)
                myPointsNew[1] = myPoints[np.argmin(diff)]
                myPointsNew[2] = myPoints[np.argmax(diff)]
                return myPointsNew

            biggest = reorder(biggest)
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0, 0],[450, 0],[0, 450],[450, 450]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarp = cv2.warpPerspective(img, matrix, (450, 450))
            imgProcessed = preProcess(imgWarp) # Nochmal Threshold auf das entzerrte
        else:
            # Fallback: Kein Gitter gefunden? Nimm das ganze Bild!
            imgProcessed = imgThreshold

        # 3. Ziffern erkennen (Nur 1x Durchlauf!)
        boxes = splitBoxes(imgProcessed)
        grid = getPredection(boxes, model)

        return jsonify({'status': 'success', 'grid': grid})

    except Exception as e:
        print(f"SERVER ERROR: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
