# Tesseract Flexible Training Script

## Overview

This repository contains a fully automated Bash script for training a custom Tesseract OCR model.
The script handles dataset preparation, train/test splitting, model training (from scratch or from a pretrained model),
evaluation, and installation of the trained model.

---

## Features

- Automatic dependency installation
- Dataset parsing and validation
- Configurable train/test split (including 100% training mode)
- Training from scratch or from a pretrained model
- Automatic unicharset and language data generation
- Built-in evaluation (character-level and string-level accuracy)
- Automatic installation of the trained model into Tesseract

---

## Requirements

- Fedora / RHEL-based Linux distribution
- sudo privileges
- PNG input images
- Filenames must contain the ground-truth values
- The images need to be in PNG format
---

## Configuration

Edit the USER CONFIGURATION section in the script.

- PY_INPUT_DIR: The directory where the training dataset is located
- PY_VALUE_REGEX: Labels get extraction from the filenames using this regex
- CHAR_SET: The character dataset that the model will get trained on
- TRAIN_PERCENT: Percentage of samples used for training
- MAX_ITERATIONS: Total training iterations
- LEARNING_RATE: Learning rate
- PY_INPUT_DIR: Input directory with labeled images
- PY_OUTPUT_BASE: Output directory for generated training data
- PY_VALUE_REGEX: Regex for label extraction
- MODEL_NAME: Name of the trained model
- TRAIN_FROM_SCRATCH: Train without a pretrained model
- CUSTOM_START_MODEL_FILE: Path to a `.traineddata` file

---

## What the Script Does

1. Installs required system packages
2. Clones and builds tesstrain
3. Prepares output directories
4. Parses images and extracts labels
5. Splits training and test datasets
6. Generates ground-truth files
7. Creates evaluation lists
8. Builds language data files
9. Trains the Tesseract model
10. Installs the trained model
11. Runs automatic evaluation

---

## Running the Script

Make executable:
```
chmod +x train.sh
```

Run:
```
./tesseract_training.sh
```

The trained model will be installed to:
```
/usr/share/tesseract/tessdata/<MODEL_NAME>.traineddata
```

---

## Testing the Model

Basic test:
```
tesseract image.png stdout -l <MODEL_NAME> --psm 7
```

With character whitelist:
```
tesseract image.png stdout -l <MODEL_NAME> --psm 7 \
  -c tessedit_char_whitelist="0123456789.B"
```

---

## Notes

- Minimum 10 training images required
- At least 1 test image required
- Corrupt images are skipped automatically
- Script exits on first error

---

## License

Provided as-is. Use at your own risk.
