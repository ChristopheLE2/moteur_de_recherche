from flask import Flask, render_template, request
import plotly.express as px
import plotly.io as pio
from statistics import mean

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    # Vous pouvez enregistrer le fichier ou faire quelque chose avec ici
    return f'File {file.filename} uploaded successfully'

@app.route('/nosInfos')
def showNosInfos():
    return render_template('nosInfos.html')

if __name__ == '__main__':
    app.run(debug=True)