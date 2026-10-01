param(
    [string]$Destination = ""
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = if ($env:ONE_FACE_MODELS_PATH) { $env:ONE_FACE_MODELS_PATH } else { Join-Path $repositoryRoot "data\face-models" }
}
$modelDirectory = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $modelDirectory | Out-Null

$models = @(
    @{
        Name = "face_detection_yunet_2023mar.onnx"
        Url = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    },
    @{
        Name = "face_recognition_sface_2021dec.onnx"
        Url = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    }
)

foreach ($model in $models) {
    $target = Join-Path $modelDirectory $model.Name
    if ((Test-Path -LiteralPath $target) -and (Get-Item -LiteralPath $target).Length -gt 0) {
        Write-Host "Already present: $target"
        continue
    }
    $temporary = "$target.part"
    Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    Write-Host "Downloading $($model.Name)"
    Invoke-WebRequest -Uri $model.Url -OutFile $temporary -UseBasicParsing
    if ((Get-Item -LiteralPath $temporary).Length -le 0) {
        throw "Downloaded file is empty: $target"
    }
    Move-Item -LiteralPath $temporary -Destination $target -Force
}

Write-Host "Face models are ready in $modelDirectory"
