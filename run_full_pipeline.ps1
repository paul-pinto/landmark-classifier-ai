param(
  [string]$DataRoot = "..\Proyecto 1 Data set\landmark_images",
  [int]$ScratchEpochs = 35,
  [int]$TransferEpochs = 20,
  [int]$ScratchBatch = 16,
  [int]$TransferBatch = 32
)

$ErrorActionPreference = "Stop"
$Py = ".\.venv_cuda\Scripts\python.exe"

Write-Host "======================================================"
Write-Host "🚀 INICIANDO PIPELINE DE ENTRENAMIENTO (GTX 1050 Ti)"
Write-Host "======================================================"

# Limpieza de variables de entorno conflictivas para Windows
if ([Environment]::GetEnvironmentVariable("PYTORCH_CUDA_ALLOC_CONF")) {
    [Environment]::SetEnvironmentVariable("PYTORCH_CUDA_ALLOC_CONF", $null)
    Write-Host "Variable PYTORCH_CUDA_ALLOC_CONF limpiada."
}

Write-Host "`n== 1) Análisis Exploratorio (EDA) =="
& $Py -m src.eda --data-root $DataRoot --out-dir outputs/eda --num-samples 6

Write-Host "`n== 2) Fase 2: Scratch Training (Arquitectura Propia) =="
# Batch Size reducido a 16 y sin AMP para evitar CUDA Out of Memory
& $Py -u -m src.train `
  --data-root $DataRoot `
  --model-type scratch `
  --epochs $ScratchEpochs `
  --batch-size $ScratchBatch `
  --lr 3e-4 `
  --label-smoothing 0.1 `
  --use-cosine-scheduler `
  --num-workers 2 `
  --checkpoint-every 5 `
  --output-dir outputs/final_run_gpu `
  --export-path models/scratch_best_gpu.pt

Write-Host "`n== 3) Fase 3: Transfer Training (ResNet18) =="
# Batch Size a 32, ResNet18 por eficiencia y 20 épocas para convergencia rápida
& $Py -u -m src.train `
  --data-root $DataRoot `
  --model-type transfer `
  --backbone resnet18 `
  --epochs $TransferEpochs `
  --batch-size $TransferBatch `
  --lr 1e-3 `
  --weight-decay 1e-4 `
  --use-cosine-scheduler `
  --num-workers 2 `
  --checkpoint-every 5 `
  --output-dir outputs/transfer_run_gpu `
  --export-path models/transfer_best_gpu.pt

Write-Host "`n== 4) Fase 4: Comparación y Verificación de Rúbrica =="
& $Py -m src.compare_models `
  --scratch-summary outputs/final_run_gpu/summary.json `
  --transfer-summary outputs/transfer_run_gpu/summary.json `
  --scratch-threshold 0.40 `
  --transfer-threshold 0.70 `
  --out-dir outputs/comparison

Write-Host "`n======================================================"
Write-Host "✅ PIPELINE FINALIZADO CON ÉXITO."
Write-Host "======================================================"