#!/bin/bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MAIN_TEX="main.tex"
INTERMEDIATE_PDF="main.pdf"
FINAL_PDF="dissertation_dewalch.pdf"
BUILD_DIR="build_artifacts"
LOCK_DIR=".build_pdf.lock"

# LaTeX files that should not stay at repo root after a successful build.
ARTIFACT_PATTERNS=(
    "*.aux"
    "*.acn"
    "*.acr"
    "*.alg"
    "*.bbl"
    "*.bcf"
    "*.blg"
    "*.fdb_latexmk"
    "*.fls"
    "*.fot"
    "*.glg"
    "*.glo"
    "*.gls"
    "*.ist"
    "*.lof"
    "*.log"
    "*.lot"
    "*.nlo"
    "*.out"
    "*.run.xml"
    "*.slg"
    "*.syg"
    "*.syi"
    "*.synctex.gz"
    "*.toc"
    "*.xdv"
)

echo -e "${GREEN}=== Dissertation Build ===${NC}"
echo "Working directory: $SCRIPT_DIR"

cleanup_lock() {
    rm -rf "$LOCK_DIR"
}

acquire_lock() {
    if ! mkdir "$LOCK_DIR" 2>/dev/null; then
        echo -e "${RED}ERROR: another dissertation build is already running in ${SCRIPT_DIR}.${NC}"
        echo "Remove $LOCK_DIR only if you are certain no compile is active."
        exit 1
    fi
    trap cleanup_lock EXIT
}

clean_previous_build() {
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"

    # Remove previous output names so each run is deterministic.
    rm -f "$INTERMEDIATE_PDF" "$FINAL_PDF"

    # Remove stale LaTeX artifacts left from interrupted/failed runs.
    shopt -s nullglob
    local stale_files=()
    for pattern in "${ARTIFACT_PATTERNS[@]}"; do
        stale_files+=( $pattern )
    done
    if (( ${#stale_files[@]} > 0 )); then
        rm -f "${stale_files[@]}"
    fi
    shopt -u nullglob
}

move_build_artifacts() {
    shopt -s nullglob
    local files=()
    for pattern in "${ARTIFACT_PATTERNS[@]}"; do
        files+=( $pattern )
    done
    if (( ${#files[@]} > 0 )); then
        mv -f "${files[@]}" "$BUILD_DIR/"
    fi
    shopt -u nullglob
}

run_latexmk() {
    echo -e "${YELLOW}[latexmk] Building ${MAIN_TEX}...${NC}"
    if ! latexmk -pdf -interaction=nonstopmode -halt-on-error "$MAIN_TEX" > /dev/null 2>&1; then
        echo -e "${RED}[latexmk] build failed. Showing tail of ${INTERMEDIATE_PDF%.pdf}.log:${NC}"
        if [[ -f "${INTERMEDIATE_PDF%.pdf}.log" ]]; then
            tail -60 "${INTERMEDIATE_PDF%.pdf}.log"
        fi
        exit 1
    fi
    echo -e "${GREEN}[latexmk] complete.${NC}"
}

acquire_lock
clean_previous_build
run_latexmk

if [[ ! -f "$INTERMEDIATE_PDF" ]]; then
    echo -e "${RED}ERROR: ${INTERMEDIATE_PDF} not generated.${NC}"
    exit 1
fi

# Archive the raw compile output and expose only the stable user-facing name at
# repo root so the dissertation directory stays tidy after a successful build.
mv -f "$INTERMEDIATE_PDF" "$BUILD_DIR/$INTERMEDIATE_PDF"
cp -f "$BUILD_DIR/$INTERMEDIATE_PDF" "$FINAL_PDF"

move_build_artifacts

PDF_SIZE=$(ls -lh "$FINAL_PDF" | awk '{print $5}')
PDF_PAGES=$(pdfinfo "$FINAL_PDF" 2>/dev/null | awk '/Pages:/ {print $2}' || echo "?")

echo ""
echo -e "${GREEN}=== Build Complete ===${NC}"
echo -e "Output: ${GREEN}${FINAL_PDF}${NC} ($PDF_SIZE, $PDF_PAGES pages)"
echo -e "Archived compile PDF: ${GREEN}${BUILD_DIR}/${INTERMEDIATE_PDF}${NC}"
echo -e "Artifacts: ${GREEN}${BUILD_DIR}/${NC}"
echo ""
