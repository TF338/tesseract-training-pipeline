#!/bin/bash
# Tesseract Flexible Training Script
# Prepares data, splits train/test, trains from scratch or pre-trained, evaluates

set -e  # Exit on error

echo "Setting up Tesseract training environment"

########################################
# USER CONFIGURATION
########################################

TRAIN_PERCENT=90			# Percentage of data used for training
MAX_ITERATIONS=10000		# Total training iterations
LEARNING_RATE=0.001			# Learning rate

# Input/Output directories
PY_INPUT_DIR="input_data"     
PY_OUTPUT_BASE="tesstrain/data/train_data"  
PY_VALUE_REGEX='-([0-9]+(?:\.[0-9]+)?BB)\.png$' 

# Model configuration
MODEL_NAME="custom_model"          
CUSTOM_START_MODEL_FILE=""         
TRAIN_FROM_SCRATCH=false            

# Character set for dataset (space-separated)
CHAR_SET="0 1 2 3 4 5 6 7 8 9 . B"

########################################
# 1. Install required packages
########################################

echo "Installing required packages..."
sudo dnf install -y \
    tesseract \
    tesseract-devel \
    tesseract-langpack-eng \
    tesseract-tools \
    git \
    make \
    autoconf \
    automake \
    libtool \
    pkg-config \
    pango-devel \
    cairo-devel \
    icu \
    python3 \
    python3-pip \
    python3-pillow \
    wget \
    bc

########################################
# 2. Clone tesstrain
########################################

if [ ! -d "tesstrain" ]; then
    echo "Cloning tesstrain..."
    git clone https://github.com/tesseract-ocr/tesstrain.git
    cd tesstrain
    make
    cd ..
else
    echo "tesstrain already cloned"
fi

########################################
# 3. Prepare directories
########################################

GT_BASE="$PY_OUTPUT_BASE"
TRAIN_DIR="$GT_BASE/train"
TEST_DIR="$GT_BASE/test"

rm -rf "$GT_BASE"
mkdir -p "$TRAIN_DIR" "$TEST_DIR"

########################################
# 4. Prepare + split data
########################################

python3 << EOF
import re, random
from pathlib import Path
from PIL import Image

INPUT_DIR = Path("$PY_INPUT_DIR")
OUTPUT_BASE = Path("$PY_OUTPUT_BASE")
TRAIN_DIR = OUTPUT_BASE / "train"
TEST_DIR = OUTPUT_BASE / "test"
VALUE_REGEX = re.compile(r"$PY_VALUE_REGEX", re.IGNORECASE)
TRAIN_PERCENT = $TRAIN_PERCENT
MODEL_NAME = "$MODEL_NAME"

samples = []

for img_path in sorted(INPUT_DIR.glob("*.png")):
    if "NO_VALUE" in img_path.name.upper(): continue
    m = VALUE_REGEX.search(img_path.name)
    if not m: continue
    val = m.group(1)
    if val.endswith(".0"): val = val[:-2]
    samples.append((img_path, val))

if not samples:
    raise SystemExit("No valid training samples found")

random.shuffle(samples)

# Handle 100% training case
if TRAIN_PERCENT >= 100:
    train_samples = samples
    # Use a small subset for testing (at least 5% or 10 samples, whichever is smaller)
    test_count = max(min(10, len(samples) // 20), 1)
    test_samples = samples[:test_count]
    print(f"100% training mode: Using {test_count} samples for testing (same as training)")
else:
    split_idx = int(len(samples) * TRAIN_PERCENT / 100)
    if split_idx >= len(samples):
        split_idx = len(samples) - 1
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

def write_samples(samples, out_dir, start_idx):
    written = 0
    skipped = 0
    for i, (img_path, val) in enumerate(samples, start_idx):
        base = f"{MODEL_NAME}_{i:06d}"
        out_img = out_dir / f"{base}.png"
        out_txt = out_dir / f"{base}.gt.txt"
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                im.load()
                clean = Image.new("RGB", im.size)
                clean.paste(im)
                clean.save(out_img, "PNG", optimize=False)
            out_txt.write_text(val, encoding="utf-8")
            written += 1
        except Exception as e:
            print(f"Skipping corrupt image: {img_path.name} ({e})")
            skipped += 1
    print(f"Written to {out_dir}: {written}")
    if skipped: print(f"Skipped corrupt images: {skipped}")

write_samples(train_samples, TRAIN_DIR, 0)
write_samples(test_samples, TEST_DIR, len(train_samples))

print(f"Total samples: {len(samples)}")
print(f"Training: {len(train_samples)}")
print(f"Testing: {len(test_samples)}")
EOF

########################################
# 5. Validate split
########################################

TRAIN_COUNT=$(ls "$TRAIN_DIR"/*.png 2>/dev/null | wc -l)
TEST_COUNT=$(ls "$TEST_DIR"/*.png 2>/dev/null | wc -l)

if [ "$TRAIN_COUNT" -lt 10 ]; then
    echo "Not enough training data (need at least 10 images, got $TRAIN_COUNT)"
    exit 1
fi

if [ "$TEST_COUNT" -lt 1 ]; then
    echo "No test data found (got $TEST_COUNT)"
    exit 1
fi

echo "Data validated: $TRAIN_COUNT training, $TEST_COUNT test images"

########################################
# 6. Evaluation list
########################################

find "$TEST_DIR" -name "*.gt.txt" | sed 's/\.gt\.txt$//' > "$TEST_DIR/list.txt"

# Verify the list file was created
if [ ! -f "$TEST_DIR/list.txt" ] || [ ! -s "$TEST_DIR/list.txt" ]; then
    echo "Failed to create evaluation list file"
    exit 1
fi

echo "Created evaluation list with $(wc -l < "$TEST_DIR/list.txt") entries"

########################################
# 7. Create unicharset / numbers / punc / wordlist
########################################

LANGDATA_DIR="$GT_BASE/langdata"
mkdir -p "$LANGDATA_DIR"

echo -e "$(echo $CHAR_SET | sed 's/ /\n/g')" > "$LANGDATA_DIR/unicharset"
echo -e "0123456789" > "$GT_BASE/numbers"
echo -e "." > "$GT_BASE/punc"
echo -e "BB\n" > "$GT_BASE/wordlist"

########################################
# 8. Start model
########################################

if [ "$TRAIN_FROM_SCRATCH" = true ]; then
    START_MODEL=""
    TESSDATA_DIR=""
elif [ -n "$CUSTOM_START_MODEL_FILE" ]; then
    START_MODEL="$CUSTOM_START_MODEL_FILE"
    TESSDATA_DIR="$(dirname "$CUSTOM_START_MODEL_FILE")"
else
    BEST_MODEL_DIR="/usr/share/tesseract/tessdata_best"
    sudo mkdir -p "$BEST_MODEL_DIR"
    if [ ! -f "$BEST_MODEL_DIR/eng.traineddata" ]; then
        echo "Downloading best English model..."
        sudo wget -O "$BEST_MODEL_DIR/eng.traineddata" \
        https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata
    fi
    START_MODEL="eng"
    TESSDATA_DIR="$BEST_MODEL_DIR"
fi

########################################
# 9. Train
########################################

cd tesstrain

echo ""
echo "   Starting training with:"
echo "   Model: $MODEL_NAME"
echo "   Start model: ${START_MODEL:-from scratch}"
echo "   Max iterations: $MAX_ITERATIONS"
echo "   Training samples: $TRAIN_COUNT"
echo "   Test samples: $TEST_COUNT"
echo ""

make training \
    MODEL_NAME="$MODEL_NAME" \
    START_MODEL="$START_MODEL" \
    TESSDATA="$TESSDATA_DIR" \
    MAX_ITERATIONS="$MAX_ITERATIONS" \
    LEARNING_RATE="$LEARNING_RATE" \
    GROUND_TRUTH_DIR="../$TRAIN_DIR" \
    EVAL_LISTFILE="../$TEST_DIR/list.txt"

########################################
# 10. Install trained model
########################################

TRAINED_MODEL="data/$MODEL_NAME.traineddata"
INSTALL_DIR="/usr/share/tesseract/tessdata"

if [ -f "$TRAINED_MODEL" ]; then
    echo "Training complete! Installing model..."
    sudo cp "$TRAINED_MODEL" "$INSTALL_DIR/$MODEL_NAME.traineddata"
    echo "Model installed to: $INSTALL_DIR/$MODEL_NAME.traineddata"
else
    echo "Trained model not found at: $TRAINED_MODEL"
    exit 1
fi

########################################
# 11. Automatic test evaluation
########################################

cd ..  # Back to main directory

echo ""
echo "Running automatic evaluation on test set..."
echo ""

TOTAL_CHARS=0
CORRECT_CHARS=0
TOTAL_STRINGS=0
CORRECT_STRINGS=0

# Check if test directory has images
if [ ! "$(ls -A "$TEST_DIR"/*.png 2>/dev/null)" ]; then
    echo "No test images found in $TEST_DIR"
    echo "   Skipping evaluation"
else
    for img in "$TEST_DIR"/*.png; do
        gt_file="${img%.png}.gt.txt"
        [ -f "$gt_file" ] || continue
        
        pred=$(tesseract "$img" stdout -l "$MODEL_NAME" --psm 7 2>/dev/null | tr -d '\n' | tr -d ' ')
        gt=$(cat "$gt_file" | tr -d '\n' | tr -d ' ')
        
        # String-level accuracy
        TOTAL_STRINGS=$((TOTAL_STRINGS + 1))
        if [ "$pred" == "$gt" ]; then
            CORRECT_STRINGS=$((CORRECT_STRINGS + 1))
        fi
        
        # Character-level accuracy
        len=$(( ${#pred} > ${#gt} ? ${#pred} : ${#gt} ))
        correct=0
        for ((i=0; i<len; i++)); do
            c1="${pred:i:1}"
            c2="${gt:i:1}"
            [ "$c1" == "$c2" ] && correct=$((correct+1))
        done
        TOTAL_CHARS=$((TOTAL_CHARS + len))
        CORRECT_CHARS=$((CORRECT_CHARS + correct))
    done

    if [ $TOTAL_CHARS -gt 0 ]; then
        CHAR_ACC=$(echo "scale=2; $CORRECT_CHARS * 100 / $TOTAL_CHARS" | bc)
        STRING_ACC=$(echo "scale=2; $CORRECT_STRINGS * 100 / $TOTAL_STRINGS" | bc)
        echo "   Evaluation Results:"
        echo "   Character-level accuracy: $CHAR_ACC% ($CORRECT_CHARS/$TOTAL_CHARS)"
        echo "   String-level accuracy:    $STRING_ACC% ($CORRECT_STRINGS/$TOTAL_STRINGS)"
    else
        echo "No characters evaluated"
    fi
fi

echo ""
echo "Training + evaluation complete!"
echo ""
echo "Training data: $TRAIN_DIR ($TRAIN_COUNT images)"
echo "Test data:     $TEST_DIR ($TEST_COUNT images)"
echo "Model:         $INSTALL_DIR/$MODEL_NAME.traineddata"
echo ""
echo "Test manually with:"
echo "tesseract image.png stdout -l $MODEL_NAME --psm 7"
echo ""
echo "With character whitelist:"
echo "   tesseract image.png stdout -l $MODEL_NAME --psm 7 -c tessedit_char_whitelist=\"$CHAR_SET\""
