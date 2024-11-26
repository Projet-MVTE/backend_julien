from flask import Flask, request, jsonify

app = Flask(__name__)

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
    app.run(host="0.0.0.0", port=5000)