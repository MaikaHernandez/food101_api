from flask import Flask, request, jsonify
from transformers import AutoImageProcessor, MobileNetV2ForImageClassification
from PIL import Image
import torch
import json
import io

# Cargar modelo y processor de Hugging Face
processor = AutoImageProcessor.from_pretrained("paolinox/mobilenet-finetuned-food101")
model = MobileNetV2ForImageClassification.from_pretrained("paolinox/mobilenet-finetuned-food101")
model.eval()

# Cargar las calorías de tu archivo local
with open("calorie_data.json", "r") as f:
    calorie_data = json.load(f)

app = Flask(__name__)

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
    
    # Buscar calorías
    cal_info = calorie_data.get(class_name_norm, {})
    return jsonify({
        "comida": class_name_norm,
        "calorias": cal_info.get("calories"),
        "unidad": cal_info.get("unit"),
        "etiqueta_original": class_name
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
