import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model_filename = 'xgboost_model.pkl'
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(request.form["Cedera"]),
                    float(request.form["Kondisi_Pasien"]),
                    float(request.form["Riwayat_Medis"]),
                    float(request.form["Usia"]),
                    float(request.form["Gejala_Nyeri_perut"]),
                    float(request.form["Gejala_Luka_ringan"]),
                    float(request.form["Gejala_Pusing"]),
                    float(request.form["Gejala_Nyeri_dada"]),
                    float(request.form["Gejala_Sesak_napas"]),
                    float(request.form["Gejala_Batuk"]),
                    float(request.form["Gejala_Mual"]),
                    float(request.form["Gejala_Pendarahan_hebat"]),
                    float(request.form["Gejala_Pingsan_tiba-tiba"]),
                    float(request.form["Gejala_Pusing_hebat"]),
                    float(request.form["Gejala_Demam"]),
                    float(request.form["Gejala_Muntah"])]

        # Reshape the features into a 2D array
        reshaped_features = np.array(features).reshape(1, -1)

        prediction = loaded_model.predict(reshaped_features)
        predicted_class = prediction[0]
        class_names = ["Tinggi", "Sedang", "Rendah"]
        predicted_class_name = class_names[predicted_class]
        return render_template('index.html', prediction_result=f'Prediksi: {predicted_class_name}')


if __name__ == '__main__':
    app.run(debug=True)
