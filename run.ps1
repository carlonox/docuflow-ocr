# ============================================================================
# DocuFlow Docker convenience wrapper for Windows (PowerShell)
# ============================================================================
#
# Usage:
#   .\run.ps1 build                     # Build minimal image
#   .\run.ps1 cascade                   # Build with RapidOCR + Surya
#   .\run.ps1 full                      # Build everything
#   .\run.ps1 shell                     # Interactive shell
#   .\run.ps1 process <image>           # Process a single document
#   .\run.ps1 batch <directory>         # Process all documents in a directory
#   .\run.ps1 test                      # Run the test suite
#
# ============================================================================

param(
    [Parameter(Position=0)]
    [ValidateSet("build", "cascade", "full", "shell", "process", "batch", "test", "help")]
    [string]$Command = "help",
    
    [Parameter(Position=1)]
    [string]$Arg1 = "",
    
    [Parameter(Position=2)]
    [string]$Arg2 = "cascade"
)

$ImageName = if ($env:DOCUFLOW_IMAGE) { $env:DOCUFLOW_IMAGE } else { "docuflow" }
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Show-Usage {
    @"
DocuFlow Docker Wrapper (Windows)

Usage:
  .\run.ps1 <command> [args...]

Commands:
  build             Build minimal image (framework only)
  cascade           Build with RapidOCR + Surya (recommended)
  full              Build with all providers
  shell             Interactive shell in the container
  process <file>    Process a single document
  batch <dir>       Process all documents in a directory
  test              Run the test suite
  help              Show this message

Environment variables:
  DOCUFLOW_IMAGE    Image name (default: docuflow)

Examples:
  .\run.ps1 cascade                         # Build recommended stack
  .\run.ps1 process C:\docs\invoice.png     # Process a document
  .\run.ps1 batch C:\documents              # Process a directory
  .\run.ps1 shell                           # Debug interactively
"@
}

function Build-Minimal {
    Write-Host "Building minimal image (framework only)..." -ForegroundColor Cyan
    docker build -t "${ImageName}:minimal" $ScriptDir
    Write-Host "Built ${ImageName}:minimal" -ForegroundColor Green
}

function Build-Cascade {
    Write-Host "Building cascade image (RapidOCR + Surya + preprocessing)..." -ForegroundColor Cyan
    docker build `
        --build-arg INSTALL_RAPID=true `
        --build-arg INSTALL_SURYA=true `
        --build-arg INSTALL_PREPROCESS=true `
        -t "${ImageName}:cascade" `
        $ScriptDir
    Write-Host "Built ${ImageName}:cascade" -ForegroundColor Green
}

function Build-Full {
    Write-Host "Building full image (all providers)..." -ForegroundColor Cyan
    docker build `
        --build-arg INSTALL_RAPID=true `
        --build-arg INSTALL_SURYA=true `
        --build-arg INSTALL_VISION=true `
        --build-arg INSTALL_PREPROCESS=true `
        --build-arg INSTALL_TESSERACT=true `
        -t "${ImageName}:full" `
        $ScriptDir
    Write-Host "Built ${ImageName}:full" -ForegroundColor Green
}

function Invoke-Shell {
    param([string]$Tag = "cascade")
    Write-Host "Starting interactive shell in ${ImageName}:${Tag}..." -ForegroundColor Cyan
    
    $dataDir = Join-Path $ScriptDir "data"
    $modelsDir = Join-Path $ScriptDir "models"
    $outputDir = Join-Path $ScriptDir "output"
    
    New-Item -ItemType Directory -Force -Path $dataDir, $modelsDir, $outputDir | Out-Null
    
    docker run --rm -it `
        -v "${dataDir}:/data" `
        -v "${modelsDir}:/models" `
        -v "${outputDir}:/output" `
        "${ImageName}:${Tag}" `
        bash
}

function Process-Document {
    param([string]$File, [string]$Tag = "cascade")
    
    if (-not (Test-Path $File)) {
        Write-Error "File not found: $File"
        return
    }
    
    $absPath = Resolve-Path $File
    $outputDir = Join-Path $ScriptDir "output"
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    
    Write-Host "Processing: $File" -ForegroundColor Cyan
    docker run --rm `
        -v "$(Split-Path -Parent $absPath):/input:ro" `
        -v "${outputDir}:/output" `
        -v "$(Join-Path $ScriptDir 'data'):/data" `
        -v "$(Join-Path $ScriptDir 'models'):/models" `
        "${ImageName}:${Tag}" `
        python examples/quick_start.py "/input/$(Split-Path -Leaf $absPath)"
}

function Process-Batch {
    param([string]$Dir, [string]$Tag = "cascade")
    
    if (-not (Test-Path $Dir -PathType Container)) {
        Write-Error "Directory not found: $Dir"
        return
    }
    
    $absDir = Resolve-Path $Dir
    $outputDir = Join-Path $ScriptDir "output"
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    
    Write-Host "Processing batch: $Dir" -ForegroundColor Cyan
    Write-Host "Output will be saved to: $outputDir\batch_results.json" -ForegroundColor DarkGray
    
    docker run --rm `
        -v "${absDir}:/input:ro" `
        -v "${outputDir}:/output" `
        -v "$(Join-Path $ScriptDir 'data'):/data" `
        -v "$(Join-Path $ScriptDir 'models'):/models" `
        "${ImageName}:${Tag}" `
        python -c @"
from pathlib import Path
from docuflow.ocr.cascade import CascadeOCR, CascadeConfig
from docuflow.ocr.providers.rapid_ocr import RapidOCRProvider
from docuflow.ocr.providers.surya import SuryaProvider
from docuflow.examples.id_card_extractor import IDCardExtractor
import json

cascade = CascadeOCR(
    fast_provider=RapidOCRProvider(),
    accurate_provider=SuryaProvider(),
    config=CascadeConfig(fast_threshold=0.90),
    context='batch processing',
)
extractor = IDCardExtractor(cascade)

input_dir = Path('/input')
output_dir = Path('/output')
results = []

for img in sorted(input_dir.glob('*.png')) + sorted(input_dir.glob('*.jpg')):
    print(f'Processing: {img.name}')
    try:
        result = extractor.process(str(img), document_id=img.name)
        results.append({
            'file': img.name,
            'success': result.success,
            'fields': result.fields,
            'validation': result.validation,
            'provider_chain': result.provider_chain,
        })
    except Exception as e:
        results.append({'file': img.name, 'error': str(e)})

output_file = output_dir / 'batch_results.json'
output_file.write_text(json.dumps(results, indent=2, default=str))
print(f'Results saved to {output_file}')
"@
}

function Invoke-Tests {
    param([string]$Tag = "cascade")
    Write-Host "Running tests in ${ImageName}:${Tag}..." -ForegroundColor Cyan
    docker run --rm "${ImageName}:${Tag}" python -m pytest tests/ -v
}

# Main dispatcher
switch ($Command) {
    "build"   { Build-Minimal }
    "cascade" { Build-Cascade }
    "full"    { Build-Full }
    "shell"   { Invoke-Shell -Tag $Arg2 }
    "process" {
        if ([string]::IsNullOrWhiteSpace($Arg1)) {
            Write-Error "Usage: .\run.ps1 process <image_file> [tag]"
            exit 1
        }
        Process-Document -File $Arg1 -Tag $Arg2
    }
    "batch" {
        if ([string]::IsNullOrWhiteSpace($Arg1)) {
            Write-Error "Usage: .\run.ps1 batch <directory> [tag]"
            exit 1
        }
        Process-Batch -Dir $Arg1 -Tag $Arg2
    }
    "test"    { Invoke-Tests -Tag $Arg2 }
    "help"    { Show-Usage }
    default   {
        Write-Error "Unknown command: $Command"
        Show-Usage
        exit 1
    }
}
