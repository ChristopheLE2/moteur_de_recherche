from flask import Flask, request, render_template, redirect, url_for
import os
import time

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

    # déclenchement du job
    open(os.path.join(INPUT_DIR, "process.job"), "w").close()

    # attente de la fin de l'indexation
    done_file = os.path.join(INPUT_DIR, "done.flag")

    timeout = 60
    start = time.time()

    while not os.path.exists(done_file):
        if time.time() - start > timeout:
            return "timeout", 500
        time.sleep(1)

    os.remove(done_file)

    return redirect(url_for('home'))

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)