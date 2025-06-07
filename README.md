# Size-Depth-of-the-Creator

A machine learning project focused on crater detection and analysis using synthetic datasets and deep learning models.

## Repository Structure

- `create_dataset.py` – Script for generating datasets of crater images and associated ellipses using SurRender. Configurable parameters include image count, resolution, field of view, and more.
- `train_model.py` – Script to train a crater detection model. Allows configuration of model backbone, training hyperparameters, dataset path, and loss metric.
- `test_model.py` – Quick script to load and test a trained crater detection model.
- `requirements.txt` – Python dependencies required for the project.
- `blobs/` – Likely stores trained model weights or binary data.
- `data/` – Presumed location for datasets.
- `docs/` – Documentation files.
- `notebooks/` – Jupyter notebooks for experimentation or analysis.
- `src/` – Source code for model, data handling, and utilities.
- `testing.ipynb` – Jupyter notebook for model testing or experimentation.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Creation

To generate a new crater dataset:

```bash
python create_dataset.py --n_train 20000 --n_val 2000 --n_test 1000
```
Additional arguments (with defaults):
- `--identifier`: Set a custom identifier for the dataset.
- `--resolution`, `--fov`: Camera settings.
- `--min_sol_incidence`, `--max_sol_incidence`: Solar incidence angle range.
- `--ellipse_limit`: Maximum ellipticity for ground-truth ellipses.
- `--filled`, `--mask_thickness`: Mask configuration.

## Training

To train the crater detection model:

```bash
python train_model.py --epochs 20 --batch_size 32 --dataset data/dataset_crater_detection_80k.h5
```

Other important arguments:
- `--backbone`: Model backbone (default: resnet50)
- `--ellipse_loss_metric`: Ellipse loss metric (`gaussian-angle`, `kullback-leibler`)
- `--device`: Device to use (`cuda` or `cpu`)
- `--learning_rate`, `--momentum`, `--weight_decay`: Optimizer parameters.

## Testing

To test a trained model:

```bash
python test_model.py
```

Make sure the trained weights (`blobs/CraterRCNN.pth`) are available.

## Requirements

See `requirements.txt` for all dependencies, including:
- numpy, pandas, matplotlib
- torch, torchvision, onnx
- scikit-learn, networkx, tqdm, h5py, astropy, mlflow, etc.
- SurRender API client

## Notes

- You may need to adjust paths and parameters according to your data and environment.
- The project relies on the SurRender software for generating synthetic crater images.

## Further Information

- For more details, explore the `docs/` and `notebooks/` directories.
- For the full file structure, see the [repository on GitHub](https://github.com/jayeshpandey01/Size-Depth-of-the-Creator/tree/main/).

---

*This README was auto-generated based on top-level files and scripts visible from the repository root. For a complete overview, please review all project files and documentation.*
