from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Importa CORS para permitir solicitudes entre dominios
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas (necesario para Insomnia y frontend)

# Cargar el modelo al iniciar la aplicación
try:
    with open('modelo_bicicletas.pkl', 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    original_columns = model_data['original_columns']
    categorical_mapping = model_data['categorical_mapping']
    drop_first = model_data['drop_first']
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print(f"❌ Error cargando el modelo: {e}")
    model = None

# Ruta principal que sirve el formulario HTML
@app.route('/')
def home():
    return render_template('formulario.html')

# Ruta para procesar las predicciones
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos JSON del frontend o de Insomnia
        data = request.get_json()
        print("📊 Datos recibidos del frontend:", data)

        # Validación de campos obligatorios
        required_fields = ['gear_count', 'weight_kg', 'frame_material', 
                         'year', 'wheel_diameter_mm', 'electronic_shifting', 
                         'brake_type', 'bike_brand', 'gear_brand']
        
        missing = [field for field in required_fields if field not in data]
        if missing:
            return jsonify({'error': f'Campos faltantes: {missing}'}), 400

        # 1. Crear DataFrame con los datos de entrada
        input_df = pd.DataFrame([data])

        # 2. Aplicar one-hot encoding (como se hizo durante el entrenamiento)
        input_encoded = pd.get_dummies(input_df, 
                                     columns=['bike_brand', 'gear_brand'], 
                                     drop_first=drop_first)

        # 3. Lista de columnas esperadas (excluyendo el precio)
        expected_columns = [col for col in original_columns if col != 'price_usd']

        # 4. Añadir columnas faltantes con valor 0
        for col in expected_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # 5. Ordenar columnas como el modelo espera
        input_encoded = input_encoded[expected_columns]

        # 6. Realizar predicción
        prediction = model.predict(input_encoded)[0]
        price_eur = round(prediction * 0.85)  # Convertir USD a EUR

        return jsonify({
            'predicted_price': price_eur,
            'status': 'success'
        })

    except Exception as e:
        print("❌ Error completo en el servidor:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Configuración para PythonAnywhere
    app.run(host='0.0.0.0', port=5000)