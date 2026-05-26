# Anomalib CFA Anomaly Detection API

A lightweight REST API for training and running an anomaly detection model on industrial images.  
Built with [FastAPI](https://fastapi.tiangolo.com/) and [Anomalib](https://github.com/openvinotoolkit/anomalib), using the **CFA** (Coupled-hypersphere-based Feature Adaptation) algorithm.

---

## How it works

1. Point the API at your image dataset
2. Call `/train` — the model learns what "normal" looks like from OK images
3. Call `/predict` — the model scores every test image and saves a heatmap overlay showing where defects are

---

## Dataset structure

Your dataset folder must follow this layout:

```
data/
├── train/
│   └── OKImages/       ← normal images used for training
└── test/
    ├── OKImages/       ← normal images used for testing
    └── NGImages/       ← defective images used for testing
```

---

## Quick start

### 1. Install dependencies

```bash
pip install anomalib fastapi uvicorn
```

> Requires Python 3.9+ and PyTorch. GPU is optional but recommended.

### 2. Run the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open Swagger UI

```
http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/status` | Check if model is trained / training |
| `POST` | `/set_dataset_path` | Set the path to your dataset folder |
| `POST` | `/train` | Start training in the background |
| `POST` | `/predict` | Run inference and save result images |

---

## Usage walkthrough

### Step 1 — Set dataset path

Pass the path as a **query parameter** in Swagger UI (not JSON body).  
Both Windows and Unix path styles work:

```
# Windows style  →  C:\Python\Anomalib\data
# Unix style     →  /home/user/data
```

> ⚠️ If you use a JSON body with a Windows path (`C:\data`), the backslash will cause a JSON parse error. Use the query parameter field instead.

---

### Step 2 — Train

```json
POST /train

{
  "data_root": "C:/Python/Anomalib/data",
  "backbone": "resnet18",
  "max_epochs": 10,
  "lr": 0.001,
  "train_batch_size": 4,
  "eval_batch_size": 4,
  "num_workers": 2,
  "val_split_ratio": 0.3
}
```

Training runs in the background. Poll `/status` to check when it finishes.

**Supported backbones:**
- `resnet18` (default, fastest)
- `wide_resnet50_2`
- `vgg19_bn`
- `efficientnet_b5`

---

### Step 3 — Predict

```
POST /predict
```

No body needed. Results are saved as heatmap overlay images:

```
prediction_results/
├── OK/    ← ok_0001_score0.1234.png  ...
└── NG/    ← ng_0001_score0.8765.png  ...
```

The response also includes AUROC and F1 metrics:

```json
{
  "metrics": [{"AUROC": 0.97, "F1Score": 0.93}],
  "total_test_images": 50,
  "ng_count": 12,
  "ok_count": 38,
  "saved_files": ["..."]
}
```

---

## Project structure

```
project/
├── app/
│   ├── main.py                  ← FastAPI application (all endpoints)
│   ├── model/
│   │   └── best_model.ckpt      ← saved after training (auto-generated)
│   └── prediction_results/
│       ├── OK/                  ← heatmaps for normal images
│       └── NG/                  ← heatmaps for defective images
└── data/
    ├── train/OKImages/
    └── test/
        ├── OKImages/
        └── NGImages/
```

---

## Notes

- The model checkpoint is saved automatically after training and reloaded on server restart — no need to retrain every time.
- The server uses `accelerator="auto"`, so it will use a GPU if available and fall back to CPU otherwise.
- `/predict` clears old result images each time it runs.

---

## License

MIT