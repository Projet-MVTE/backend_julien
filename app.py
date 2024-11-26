"""
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Hello, World!"}), 200

@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        # Recevoir les données envoyées dans la requête
        data = request.json
        script_name = data.get("script_name", "default_script")
        params = data.get("params", {})

        # Exemple : Exécuter un script Python (ici, il faut que le script existe sur le serveur)
        if script_name == "example":
            result = example_script(**params)
            return jsonify({"status": "success", "result": result}), 200
        else:
            return jsonify({"status": "error", "message": "Script non trouvé"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def example_script(param1=None, param2=None):
    # Exemple de logique dans un script Python
    return {"param1": param1, "param2": param2}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
"""

from flask import Flask, jsonify, send_file

app = Flask(__name__)

# Définir une route pour la racine "/"
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Coucou dina et jojo"}), 200
# Route pour télécharger un fichier JSON
@app.route("/download-json", methods=["GET"])
def download_json():
    # Chemin vers votre fichier JSON local
    json_file_path = "arbre.json"  # Assurez-vous que le fichier est dans le même répertoire ou spécifiez le chemin absolu

    try:
        # Envoyer le fichier JSON
        return send_file(json_file_path, as_attachment=True, mimetype='application/json')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Utilisation du port fourni par Railway
    app.run(host="0.0.0.0", port=port)
