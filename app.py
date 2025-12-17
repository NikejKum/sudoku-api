import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import io

app = Flask(__name__)

# --- Modell laden (nur 1x beim Start) ---
try:
    model = load_model('digit_model.h5')
    print("Modell erfolgreich geladen")
except:
    print("FEHLER: model.h5 nicht gefunden!")
    model = None

# --- Hilfsfunktionen ---

def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    return imgThreshold

def biggestContour(contours):
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
    return biggest

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

def splitBoxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            # Ränder abschneiden (wichtig für OCR)
            h, w = box.shape
            crop = box[4:h-4, 4:w-4]
            crop = cv2.resize(crop, (28, 28))
            boxes.append(crop)
    return boxes

def getPredection(boxes, model):
    result = []
    confidence_sum = 0
    
    # Batch prediction ist schneller
    boxes_array = np.array(boxes)
    boxes_array = boxes_array.reshape(boxes_array.shape[0], 28, 28, 1)
    boxes_array = boxes_array / 255.0 # Normieren
    
    predictions = model.predict(boxes_array)
    
    for i in range(81):
        prob = np.amax(predictions[i])
        classIndex = np.argmax(predictions[i])
        
        # Schwellenwert: Nur wenn > 70% sicher
        if prob > 0.7:
            result.append(int(classIndex))
            confidence_sum += prob
        else:
            result.append(0)
            
    return result, confidence_sum

def try_orientations(img, model):
    """
    Probiert 4 Orientierungen (0, 90, 180, 270)
    Gibt das Grid zurück, das die höchste Konfidenz hatte.
    """
    best_grid = []
    best_conf = -1
    
    # 1. Bild vorbereiten (Warping muss vorher passiert sein)
    # Wir nehmen an, img ist schon 450x450 Sudoku
    
    orientations = [0, 1, 2, 3] # 0=Original, 1=90, 2=180, 3=270 (CounterClockwise)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.adaptiveThreshold(img_gray, 255, 1, 1, 11, 2)
    
    for rot in orientations:
        # Rotieren
        rotated_img = img_thresh.copy()
        if rot == 1: rotated_img = cv2.rotate(img_thresh, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rot == 2: rotated_img = cv2.rotate(img_thresh, cv2.ROTATE_180)
        elif rot == 3: rotated_img = cv2.rotate(img_thresh, cv2.ROTATE_90_CLOCKWISE)
        
        # Splitten
        boxes = splitBoxes(rotated_img)
        
        # Erkennen
        grid, confidence = getPredection(boxes, model)
        
        print(f"Rotation {rot*90}° - Konfidenz: {confidence}")
        
        if confidence > best_conf:
            best_conf = confidence
            best_grid = grid
            
    return best_grid

# --- Routes ---

@app.route('/', methods=['GET'])
def home():
    return "Sudoku AI is ready!"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
        if model is None: return jsonify({'error': 'Model not loaded'}), 500

        file = request.files['file']
        in_memory_file = io.BytesIO(file.read())
        file_bytes = np.frombuffer(in_memory_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # 1. Sudoku finden
        img = cv2.resize(img, (450, 450))
        imgThreshold = preProcess(img)
        contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest = biggestContour(contours)

        if biggest.size != 0:
            biggest = reorder(biggest)
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0, 0],[450, 0],[0, 450],[450, 450]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarp = cv2.warpPerspective(img, matrix, (450, 450))
            
            # 2. Orientierung testen & Erkennen
            final_grid = try_orientations(imgWarp, model)
            
            return jsonify({'status': 'success', 'grid': final_grid})
        else:
            return jsonify({'error': 'No Sudoku found'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

