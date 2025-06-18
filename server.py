from flask import Flask, request, jsonify
from transformers import AutoImageProcessor, MobileNetV2ForImageClassification
from PIL import Image
import torch
import json
import io
import requests

# Configura tu API KEY de Spoonacular aquí:
SPOONACULAR_KEY = "751ea2230a71467f8ecb1f61cbc76380"

# Cargar modelo y processor de Hugging Face
processor = AutoImageProcessor.from_pretrained("paolinox/mobilenet-finetuned-food101")
model = MobileNetV2ForImageClassification.from_pretrained("paolinox/mobilenet-finetuned-food101")
model.eval()

# Cargar las calorías de tu archivo local
try:
    with open("calorie_data.json", "r") as f:
        calorie_data = json.load(f)
except FileNotFoundError:
    calorie_data = {}

app = Flask(__name__)

def buscar_calorias_spoonacular(nombre_platillo, api_key):
    url = "https://api.spoonacular.com/recipes/guessNutrition"
    params = {
        "title": nombre_platillo.replace('_', ' '),
        "apiKey": api_key
    }
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        cal = data.get("calories", {}).get("value")
        unit = data.get("calories", {}).get("unit")
        return cal, unit
    return None, None

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Leer imagen enviada
    image_bytes = request.files['image'].read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Preprocesar imagen
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        class_name = model.config.id2label[predicted_class_idx]
        class_name_norm = class_name.lower().replace(' ', '_')

    # Buscar calorías primero localmente
    cal_info = calorie_data.get(class_name_norm, {})
    cal = cal_info.get("calories")
    unit = cal_info.get("unit")

    if cal is None or unit is None:
        # Consultar Spoonacular
        cal, unit = buscar_calorias_spoonacular(class_name_norm, SPOONACULAR_KEY)
        # Guardar en JSON local si se encontraron calorías
        if cal is not None and unit is not None:
            calorie_data[class_name_norm] = {"calories": cal, "unit": unit}
            with open("calorie_data.json", "w") as f:
                json.dump(calorie_data, f, indent=2)

    return jsonify({
        "comida": class_name_norm,
        "calorias": cal,
        "unidad": unit,
        "etiqueta_original": class_name
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

