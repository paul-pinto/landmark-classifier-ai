# Clasificador de Landmarks con Deep Learning

Este proyecto implementa un pipeline completo de visión artificial para la clasificación de 50 clases de puntos de referencia (landmarks) globales, utilizando PyTorch. Desarrollado como parte de la Maestría en Ciencia de Datos e IA de la Universidad Católica Boliviana.

## 🚀 Estructura del Proyecto

```text
landmark-classifier/
├── cnn_from_scratch.ipynb   # Fase 1 y 2: Arquitectura propia (Accuracy: 49.92%)
├── transfer_learning.ipynb  # Fase 3: Fine-tuning con ResNet18 (Accuracy: 72.40%)
├── app.ipynb                # Fase 4: Predicción, Gradio UI y Generación de PDF
├── src/                     # Scripts modulares (Data, Model, Train, Predictor)
├── models/                  # Modelos exportados en formato TorchScript (.pt)
├── inference_images/        # Imágenes de prueba (incluyendo casos locales)
├── outputs/                 # Gráficas, checkpoints y métricas JSON
├── run_pipeline.ps1         # Script de automatización (PowerShell)
├── report.pdf               # Reporte analítico final autogenerado
└── requirements.txt         # Dependencias del sistema