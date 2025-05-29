from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Importa CORS
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas

# Cargar el modelo
try:
    with open('modelo_bicicletas.pkl', 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    original_columns = model_data['original_columns']
    categorical_mapping = model_data['categorical_mapping']
    drop_first = model_data['drop_first']
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    model = None

# Ruta principal que sirve el formulario HTML
@app.route('/')
def home():
    return render_template('formulario.html')

# Ruta para procesar las predicciones
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Usa get_json() en lugar de request.json
        print("Datos recibidos del frontend:", data)

        # Validación de campos
        required_fields = ['gear_count', 'weight_kg', 'frame_material', 
                         'year', 'wheel_diameter_mm', 'electronic_shifting', 
                         'brake_type', 'bike_brand', 'gear_brand']
        
        missing = [field for field in required_fields if field not in data]
        if missing:
            return jsonify({'error': f'Campos faltantes: {missing}'}), 400

        # 1. Crear DataFrame
        input_df = pd.DataFrame([data])

        # 2. Aplicar get_dummies (como en entrenamiento)
        input_encoded = pd.get_dummies(input_df, 
                                     columns=['bike_brand', 'gear_brand'], 
                                     drop_first=drop_first)

        # 3. Lista de columnas esperadas (EXCLUYENDO price_usd)
        expected_columns = [col for col in original_columns if col != 'price_usd']

        # 4. Añadir columnas faltantes (solo las necesarias)
        for col in expected_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0  # Rellena con 0 (o False si son dummy)

        # 5. Ordenar columnas como el modelo espera
        input_encoded = input_encoded[expected_columns]

        # 6. Predecir
        prediction = model.predict(input_encoded)[0]
        price_eur = round(prediction * 0.85)

        return jsonify({
            'predicted_price': price_eur,
            'status': 'success'
        })

    except Exception as e:
        print("Error completo en el servidor:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)