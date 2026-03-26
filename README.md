# FingerprintDeflectometry
This repository contains the codebase for retrieving latent fingerprint using modulated intensity maps from deflectometry recordings

## Features
- Projects sinusoidal fringes on a secondary screen.
- Opens a live camera view to align and focus the surface under test.
- Computes phase-shift, phase (wrapped/unwrapped), and the **modulated-intensity (amplitude) map**.
- Optional phase-compensation step.
- Saves results in a structured folder called Results/surface.

## Requirements-downloads
- Python 3.9+ (recommended)
- Packages: `numpy`, `opencv-python`, `vmbpy` (Allied Vision)

> **Note (Allied Vision):** Ensure the Allied Vision runtime/driver is installed for your camera model.
python -m pip install 'PATH_TO_REPOSITORY/FingerprintDeflectometry/vmbpy-1.1.0-py3-none-win_amd64.whl[numpy,opencv]'


## Contact
   -Julian Perez-Carvajal    (julperezca@unal.edu.co)
   -Carlos A. Buitrago-Duque (cabuitragod@unal.edu.co)
   -Jorge I. Garcia-Sucerquia (jigarcia@unal.edu.co)
  