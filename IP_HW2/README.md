# Image Processing HW2

This project contains implementations of several basic image processing techniques, including **morphological operations, spatial/frequency-domain filtering, hybrid images, and image denoising**.

## Project Contents

- **Morphology**
  - Dilation
  - Erosion
  - Opening
  - Closing
  - Comparison of different structuring elements

- **Filtering**
  - Custom 2D convolution
  - Spatial-domain filtering
  - Frequency-domain filtering using FFT
  - Hybrid image generation

- **Denoising**
  - Gaussian-noise removal
  - Mean / Gaussian filtering experiments
  - Salt-and-pepper noise removal using morphological operations
  - CAPTCHA preprocessing and OCR testing with EasyOCR
  - SNR-based evaluation

## Project Structure

```text
IP_HW2/
├── demo_v4.ipynb          # Main demonstration notebook
├── morphology.py          # Morphological operations
├── filtering.py           # Spatial/frequency filtering and hybrid images
├── denoising.py           # Gaussian and salt-and-pepper denoising
├── images/                # Input images used in the experiments
├── Image Processing HW2.pdf       # Project report
├── requirements.txt
└── README.md
```

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the notebook from the project root:

```bash
jupyter notebook demo_v4.ipynb
```

The notebook demonstrates the main experiments using the images in the `images/` folder.

## Main Files

### `morphology.py`
Implements custom morphological operations and padding functions.

### `filtering.py`
Implements custom 2D convolution, spatial-domain filtering, frequency-domain filtering, and hybrid image generation.

### `denoising.py`
Implements Gaussian-noise denoising and a morphology-based preprocessing pipeline for salt-and-pepper noisy CAPTCHA images.

## Notes

- Image paths in the notebook are relative to the project root, so run the notebook from the `IP_HW2` directory.
- EasyOCR may download its recognition model automatically the first time it is used.
- The notebook creates `eroded_image.png` temporarily during the OCR experiment.

## Report

See `影像處理_HW2.pdf` for the experiment design, figures, SNR comparisons, and discussion.
