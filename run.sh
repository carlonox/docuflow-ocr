#!/usr/bin/env bash
# ============================================================================
# DocuFlow convenience wrapper
# ============================================================================
#
# Usage:
#   ./run.sh                           # Build and run with default config
#   ./run.sh build                     # Build only
#   ./run.sh cascade                   # Build with RapidOCR + Surya
#   ./run.sh full                      # Build everything
#   ./run.sh shell                     # Interactive shell
#   ./run.sh process <image>           # Process a single document
#   ./run.sh batch <directory>         # Process all documents in a directory
#   ./run.sh test                      # Run the test suite
#
# ============================================================================

set -euo pipefail

IMAGE_NAME="${DOCUFLOW_IMAGE:-docuflow}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
DocuFlow Docker Wrapper

Usage:
  $0 [command] [args...]

Commands:
  build           Build minimal image (framework only)
  cascade         Build with RapidOCR + Surya (recommended)
  full            Build with all providers
  shell           Interactive shell in the container
  process <file>  Process a single document
  batch <dir>     Process all documents in a directory
  test            Run the test suite
  help            Show this message

Environment variables:
  DOCUFLOW_IMAGE  Image name (default: docuflow)

Examples:
  $0 cascade                              # Build recommended stack
  $0 process /path/to/document.png        # Process a document
  $0 batch /path/to/documents             # Process a directory
  $0 shell                                # Debug interactively

EOF
}

build_minimal() {
    echo "Building minimal image (framework only)..."
    docker build -t "${IMAGE_NAME}:minimal" "$SCRIPT_DIR"
    echo "✓ Built ${IMAGE_NAME}:minimal"
}

build_cascade() {
    echo "Building cascade image (RapidOCR + Surya + preprocessing)..."
    docker build \
        --build-arg INSTALL_RAPID=true \
        --build-arg INSTALL_SURYA=true \
        --build-arg INSTALL_PREPROCESS=true \
        -t "${IMAGE_NAME}:cascade" \
        "$SCRIPT_DIR"
    echo "✓ Built ${IMAGE_NAME}:cascade"
}

build_full() {
    echo "Building full image (all providers)..."
    docker build \
        --build-arg INSTALL_RAPID=true \
        --build-arg INSTALL_SURYA=true \
        --build-arg INSTALL_VISION=true \
        --build-arg INSTALL_PREPROCESS=true \
        --build-arg INSTALL_TESSERACT=true \
        -t "${IMAGE_NAME}:full" \
        "$SCRIPT_DIR"
    echo "✓ Built ${IMAGE_NAME}:full"
}

run_shell() {
    local tag="${1:-cascade}"
    echo "Starting interactive shell in ${IMAGE_NAME}:${tag}..."
    docker run --rm -it \
        -v "${SCRIPT_DIR}/data:/data" \
        -v "${SCRIPT_DIR}/models:/models" \
        -v "${SCRIPT_DIR}/output:/output" \
        "${IMAGE_NAME}:${tag}" \
        bash
}

process_document() {
    local file="$1"
    local tag="${2:-cascade}"
    
    if [[ ! -f "$file" ]]; then
        echo "Error: file not found: $file" >&2
        exit 1
    fi
    
    local abs_path
    abs_path="$(cd "$(dirname "$file")" && pwd)/$(basename "$file")"
    local output_dir="${SCRIPT_DIR}/output"
    mkdir -p "$output_dir"
    
    echo "Processing: $file"
    docker run --rm \
        -v "$(dirname "$abs_path"):/input:ro" \
        -v "${output_dir}:/output" \
        -v "${SCRIPT_DIR}/data:/data" \
        -v "${SCRIPT_DIR}/models:/models" \
        "${IMAGE_NAME}:${tag}" \
        python examples/quick_start.py "/input/$(basename "$abs_path")"
}

process_batch() {
    local dir="$1"
    local tag="${2:-cascade}"
    
    if [[ ! -d "$dir" ]]; then
        echo "Error: directory not found: $dir" >&2
        exit 1
    fi
    
    local abs_dir
    abs_dir="$(cd "$dir" && pwd)"
    local output_dir="${SCRIPT_DIR}/output"
    mkdir -p "$output_dir"
    
    echo "Processing batch: $dir"
    docker run --rm \
        -v "${abs_dir}:/input:ro" \
        -v "${output_dir}:/output" \
        -v "${SCRIPT_DIR}/data:/data" \
        -v "${SCRIPT_DIR}/models:/models" \
        "${IMAGE_NAME}:${tag}" \
        python -c "
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
print(f'✓ Results saved to {output_file}')
"
}

run_tests() {
    local tag="${1:-cascade}"
    echo "Running tests in ${IMAGE_NAME}:${tag}..."
    docker run --rm \
        "${IMAGE_NAME}:${tag}" \
        python -m pytest tests/ -v
}

# Main dispatcher
case "${1:-help}" in
    build)
        build_minimal
        ;;
    cascade)
        build_cascade
        ;;
    full)
        build_full
        ;;
    shell)
        run_shell "${2:-cascade}"
        ;;
    process)
        if [[ $# -lt 2 ]]; then
            echo "Usage: $0 process <image_file> [tag]" >&2
            exit 1
        fi
        process_document "$2" "${3:-cascade}"
        ;;
    batch)
        if [[ $# -lt 2 ]]; then
            echo "Usage: $0 batch <directory> [tag]" >&2
            exit 1
        fi
        process_batch "$2" "${3:-cascade}"
        ;;
    test)
        run_tests "${2:-cascade}"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown command: $1" >&2
        usage
        exit 1
        ;;
esac
