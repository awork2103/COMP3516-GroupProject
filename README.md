# COMP3516 Group Project: Multi-modal Human Activity Recognition

This project builds human activity recognition classifiers using the OctoNet data provided for COMP3516. The task uses three sensor modalities: infrared array (IRA), WiFi CSI, and IMU. The dataset contains five activity classes: sitting, walking, sleeping, falling down, and jumping.

## Project Structure

```text
.
├── instructions.ipynb              # Main notebook with visualization, preprocessing, training, and inference
├── activity_masked.csv             # Provided CSV with labelled rows and masked test rows
├── data_sources/                   # Provided pickle data files
├── activity_IRA_labelled.csv       # IRA model predictions for masked rows
├── activity_CSI_labelled.csv       # CSI model predictions for masked rows
├── activity_IMU_labelled.csv       # IMU model predictions for masked rows
├── activity_multimodal_labelled.csv # Multimodal model predictions for masked rows
├── ira_model_best.pt               # Best IRA checkpoint
├── csi_model_best.pt               # Best CSI checkpoint
├── imu_model_best.pt               # Best IMU checkpoint
├── multimodal_model_best.pt        # Best multimodal checkpoint
├── output.png                      # IRA visualization output
└── output_wifi.png                 # CSI visualization output
```

## Requirements

The notebook uses Python with the following libraries:

```text
numpy
pandas
matplotlib
torch
jupyter
```

CUDA is used automatically if available; otherwise the notebook falls back to CPU.

Example setup:

```bash
pip install numpy pandas matplotlib torch jupyter
```

## Data Assumptions

Keep the provided project files in the repository root:

```text
activity_masked.csv
data_sources/*.pickle
```

The notebook includes a compatibility unpickler for the provided metadata and NumPy pickle paths. It also aligns variable-length modality streams to fixed lengths before training:

```text
IRA: 24 frames
CSI: 48 frames
IMU: 64 frames
```

## How To Reproduce

1. Open `instructions.ipynb`.
2. Run the notebook cells from top to bottom.
3. Task 1 visualizes one IRA example per class and one CSI sample.
4. The shared preprocessing cells load pickle files, clean metadata, align frames, and define CSV export helpers.
5. Task 2 trains the three single-modality classifiers:
   - IRA classifier
   - CSI classifier
   - IMU classifier
6. Task 3 trains the multimodal classifier using late fusion over IRA, CSI, and IMU encoders.
7. Each single-modality model saves its own best checkpoint and exports its own prediction CSV for the previously masked rows.
8. The multimodal model also saves its best checkpoint and exports the final multimodal prediction CSV.

## Model Summary

The single-modality models each use modality-specific preprocessing and neural network architectures:

- IRA uses a 3D CNN over aligned infrared frames.
- CSI uses a 2D CNN over CSI amplitude and temporal-difference features.
- IMU uses a 3D CNN over a structured IMU tensor with raw and temporal-difference channels.

The multimodal model uses a late-fusion architecture:

- IRA, CSI, and IMU are processed by separate encoders.
- Each encoder projects its output into a 64-dimensional embedding.
- The three embeddings are concatenated into a 192-dimensional fused vector.
- A fully connected classifier predicts one of the five activity classes.

## Training Details

The notebook uses stratified train-validation splits for reproducibility. The training loops include:

- deterministic seeding
- modality-specific normalization
- data augmentation
- AdamW optimization
- learning-rate scheduling
- label smoothing
- gradient clipping
- early stopping based on validation accuracy, with validation loss as a tie-breaker
- best-checkpoint saving

## Output CSV Format

The submitted prediction CSV keeps the same format as `activity_masked.csv`:

```text
filename,activity,activity_id
```

The activity label mapping is:

```text
1: sitting
2: walking
3: sleeping
4: falling down
5: jumping
```

The primary final prediction file is:

```text
activity_multimodal_labelled.csv
```

The single-modality models also produce their own labelled CSV files:

```text
activity_IRA_labelled.csv
activity_CSI_labelled.csv
activity_IMU_labelled.csv
```

These files are included for comparison with the multimodal result.

## Notes

The notebook is intended to be the single source of truth for preprocessing, model definitions, training, inference, and visualization. No separate training scripts are required.
