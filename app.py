import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import io
import os

app = Flask(__name__)

# --- Modell laden ---
print("Starte Server...")
try:
    # Versuche das Modell zu laden (Dateiname mit Unterstrich beachten!)
    model_path = 'digit_model.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Modell erfolgreich geladen!")
    else:
        print(f"WARNUNG: {model_path} nicht gefunden! (Prüfe Dateinamen auf GitHub)")
        model = None
except Exception as e:
    print(f"WARNUNG: Kritischer Fehler beim Laden des Modells: {e}")
    model = None

@app.route('/', methods=['GET'])
def home():
    return "Sudoku AI Turbo-Mode (Ready)"

@app.route('/analyze', methods=['POST'])
def analyze():
    # Fallback, falls Modell fehlt
    if not model: 
        print("Anfrage erhalten, aber kein Modell geladen.")
        return jsonify({'grid': [0]*81, 'status': 'error_no_model'})
    
    try:
        if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
        
        # 1. Bild aus dem Request lesen
        file = request.files['file']
        in_memory_file = io.BytesIO(file.read())
        file_bytes = np.frombuffer(in_memory_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # 2. ZUSCHNEIDEN (Der "LEGO-Filter")
        # Wir gehen davon aus, dass das Bild VGA (640x480) ist.
        # Wir schneiden die bunten Ränder weg, um nur das Gitter zu behalten.
        h, w, _ = img.shape
        
        # Werte für VGA (640x480)
        crop_x = 120  # Links & Rechts je 120px weg (bleiben 400px Breite)
        crop_y = 60   # Oben & Unten je 60px weg (bleiben 360px Höhe)
        
        # Sicherheitscheck, falls Bild kleiner ist
        if h > 2*crop_y and w > 2*crop_x:
            img = img[crop_y:h-crop_y, crop_x:w-crop_x]
        
        # 3. AUFBEREITUNG
        # Jetzt skalieren wir den Ausschnitt auf unser Standard-Format
        img = cv2.resize(img, (450, 450))
        
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Adaptiver Threshold macht aus grauem Papier hartes Schwarz-Weiß
        imgThresh = cv2.adaptiveThreshold(imgGray, 255, 1, 1, 11, 2)
        
        # 4. ZERSCHNEIDEN & ERKENNEN
        grid = []
        rows = np.vsplit(imgThresh, 9)
        
        # Wir sammeln alle 81 Kästchen in einem Stapel für die KI
        batch_images = []
        
        for r in rows:
            cols = np.hsplit(r, 9)
            for box in cols:
                h_box, w_box = box.shape
                # Rand vom Kästchen entfernen (Gitterlinien wegmachen)
                # 4 Pixel Rand weg ist ein guter Standardwert
                crop = box[4:h_box-4, 4:w_box-4]
                
                # Auf 28x28 skalieren (Input-Größe für das KI-Modell)
                crop = cv2.resize(crop, (28, 28))
                
                # Normieren (0..1) und richtige Form (28,28,1)
                crop = crop / 255.0
                crop = crop.reshape(28, 28, 1)
                batch_images.append(crop)
        
        # KI Vorhersage für alle 81 Bilder auf einmal (schneller!)
        batch_array = np.array(batch_images)
        predictions = model.predict(batch_array, verbose=0)
        
        # Ergebnisse auswerten
        for i in range(81):
            prob = np.amax(predictions[i])     # Wie sicher ist die KI?
            classIndex = np.argmax(predictions[i]) # Welche Zahl ist es?
            
            # Schwellenwert: Nur wenn > 70% sicher, sonst ist es leer (0)
            if prob > 0.7:
                grid.append(int(classIndex))
            else:
                grid.append(0)

        print("Analyse fertig. Grid erkannt.")
        return jsonify({'status': 'success', 'grid': grid})

    except Exception as e:
        print(f"SERVER ERROR: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Wichtig für Render: Port aus Umgebungsvariable oder 10000
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
