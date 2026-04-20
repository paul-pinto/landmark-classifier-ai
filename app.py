import gradio as gr
import torch
from pathlib import Path

# Importamos tu función original desde la carpeta src
from src.predictor import predict_landmarks

# 1. Configuración de Rutas
# Usamos rutas relativas para que funcionen tanto en tu PC local como en la nube
MODEL_PATH = 'models/transfer_best_gpu.pt'
CLASS_MAP = 'outputs/transfer_run_gpu/class_to_idx.json'

# 2. Función de Inferencia (Wrapper para Gradio)
def clasificar_imagen(img_path):
    """
    Recibe la imagen subida por el usuario, la procesa con la red neuronal
    y devuelve las probabilidades en el formato que requiere la interfaz visual.
    """
    if img_path is None:
        return {}
        
    try:
        # Extraemos el Top-5 de predicciones usando tu código
        predicciones = predict_landmarks(img_path, MODEL_PATH, CLASS_MAP, k=5)
        
        # Gradio (gr.Label) necesita un diccionario: {'Torre Eiffel': 0.88, ...}
        resultados = {clase: prob for clase, prob in predicciones}
        return resultados
    except Exception as e:
        return {"Error en la predicción": 0.0}

# 3. Construcción de la Interfaz Web
interfaz = gr.Interface(
    fn=clasificar_imagen,
    inputs=gr.Image(type="filepath", label="📸 Sube o arrastra una foto aquí"),
    outputs=gr.Label(num_top_classes=5, label="📊 Top 5 Predicciones de la Red"),
    title="🏛️ Clasificador de Landmarks (Deep Learning)",
    description=(
        "Esta aplicación utiliza una red neuronal **ResNet18** entrenada mediante Transfer Learning "
        "para identificar 50 hitos arquitectónicos globales.\n\n"
        "**Análisis de Sesgos:** Te invito a subir fotografías de infraestructura local "
        "(ej. Amazonía Boliviana) para evaluar cómo responde el modelo ante escenarios "
        "ausentes en su set de datos original (alucinación estadística)."
    ),
    flagging_mode="never" # Mantiene la interfaz limpia sin el botón de reportar
)

# 4. Lanzamiento de la Aplicación
if __name__ == "__main__":
    # Al ejecutar localmente o en Hugging Face, esto levanta el servidor web
    interfaz.launch()