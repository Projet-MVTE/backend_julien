from flask import Flask, jsonify, send_file, request

app = Flask(__name__)

# Définir une route pour la racine "/"
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "IA-MVTE Application, developped at ENSTA Bretagne in 2024 by Dina, Johan, Julien, Jawad and Naim"}), 200
    
# Route pour télécharger un fichier JSON
@app.route("/download_tree", methods=["GET"])
def download_json():
    json_file_path = "arbre.json"
    try:
        return send_file(json_file_path, as_attachment=True, mimetype='application/json')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/set_tree', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    file.save("arbre.json")  # Enregistre directement dans la racine
    return "File uploaded successfully", 200

@app.route('/askIA', methods=['GET'])
def calculate():
    # Récupération des paramètres de la requête
    a = request.args.get('a', type=float)
    b = request.args.get('b', type=float)
    operation = request.args.get('operation', type=str)

    # Vérification des paramètres
    if a is None or b is None or operation is None:
        return jsonify({"error": "Missing parameters. Please provide 'a', 'b', and 'operation'."}), 400
    # Retourner le résultat
    return jsonify({"result": "test"}), 200

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Utilisation du port fourni par Railway
    app.run(host="0.0.0.0", port=port)
