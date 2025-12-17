        print("Starte Test-Modus (OHNE KI)...")
        # Wir simulieren eine Erkennung in 0 Sekunden
        grid = [1,2,3,4,5,6,7,8,9] * 9  # Einfach Muster zurückgeben
        
        print("Fertig!")
        return jsonify({'status': 'success', 'grid': grid})
