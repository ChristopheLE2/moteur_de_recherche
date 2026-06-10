from flask import Flask, request, render_template, redirect, url_for
import os
import time
import json

app = Flask(__name__)
INPUT_DIR = "/input"

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]
    image_path = os.path.join(INPUT_DIR, "image.jpg")
    file.save(image_path)

    return {"success": True}

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()

    config = {
        "descriptors": data.get("descriptors", []),
        "distance": data.get("distance"),
        "nb_results": data.get("nb_results")
    }

    config_file = os.path.join(
        INPUT_DIR,
        "search_config.json"
    )

    with open(config_file, "w") as f:
        json.dump(config, f)

    # déclenchement du job
    open(os.path.join(INPUT_DIR, "process.job"), "w").close()

    # attente de la fin de l'indexation
    done_file = os.path.join(INPUT_DIR, "done.flag")

    timeout = 60
    start = time.time()

    while not os.path.exists(done_file):
        if time.time() - start > timeout:
            return {"success": False}, 500
        time.sleep(1)

    os.remove(done_file)

    return {"success": True}

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)