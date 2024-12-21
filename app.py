from flask import Flask, jsonify, send_file

app = Flask(__name__)

# Définir une route pour la racine "/"
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "test 1"}), 200
    
# Route pour télécharger un fichier JSON
@app.route("/download_three", methods=["GET"])
def download_json():
    json_file_path = "arbre.json"
    try:
        return send_file(json_file_path, as_attachment=True, mimetype='application/json')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

"""
@app.route("/set_three/<string:input_string>", methods=["GET"])
def set_three(input_string):
    try:
        f = open("arbre_decisions.txt","w")
        f.write(input_string)
        f.close()
        return jsonify({"Fichier modifié": chaine}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
"""

@app.route('/set_tree', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    file.save("arbre.json")  # Enregistre directement dans la racine
    return "File uploaded successfully", 200

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Utilisation du port fourni par Railway
    app.run(host="0.0.0.0", port=port)
