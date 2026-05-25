"""
Anomalib CFA Anomaly Detection API
====================================
A FastAPI service that trains and runs a CFA (Coupled Flow Attention) anomaly
detection model using the Anomalib library.

Dataset folder structure required:
    data/
    ├── train/OKImages/     <- normal training images
    ├── test/NGImages/      <- defective test images
    └── test/OKImages/      <- normal test images

Endpoints:
    GET  /status            - Check training status
    POST /set_dataset_path  - Point the service to your dataset folder
    POST /train             - Start training in the background
    POST /predict           - Run inference and save heatmap results
"""

import shutil
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from pydantic import Field, field_validator
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel

from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.engine import Engine
from anomalib.metrics import AUROC, Evaluator, F1Score
from anomalib.models import Cfa
from anomalib.callbacks import ModelCheckpoint
from anomalib.visualization import ImageVisualizer
from anomalib.visualization.image.item_visualizer import visualize_image_item


# Allow Evaluator to be deserialized safely from checkpoints
torch.serialization.add_safe_globals([Evaluator])

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
_APP_DIR      = Path(__file__).resolve().parent
DATA_ROOT     = _APP_DIR.parent / "data"          # default dataset location
MODEL_DIR     = _APP_DIR / "model"
RESULT_OK_DIR = _APP_DIR / "prediction_results" / "OK"
RESULT_NG_DIR = _APP_DIR / "prediction_results" / "NG"
CHECKPOINT    = MODEL_DIR / "best_model.ckpt"

for d in (MODEL_DIR, RESULT_OK_DIR, RESULT_NG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Runtime state (in-memory; reset on restart)
# ---------------------------------------------------------------------------
state: dict = {
    "model":      None,
    "engine":     None,
    "datamodule": None,
    "trained":    False,
    "training":   False,
}

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    """Parameters for the /train endpoint."""

    data_root:        str   = str(DATA_ROOT)
    backbone: Literal[
        "vgg19_bn", "resnet18", "wide_resnet50_2", "efficientnet_b5"
    ] = Field(default="resnet18")
    max_epochs:       int   = 1
    lr:               float = 0.001
    train_batch_size: int   = 1
    eval_batch_size:  int   = 1
    num_workers:      int   = 1
    val_split_ratio:  float = 0.3

    @field_validator("data_root", mode="before")
    @classmethod
    def normalize_data_root(cls, v: str) -> str:
        # Replace backslashes so Windows paths survive JSON parsing
        return v.replace("\\", "/")


class FolderPathRequest(BaseModel):
    """Path to the root dataset folder."""

    dataset_path: str

    @field_validator("dataset_path", mode="before")
    @classmethod
    def normalize_dataset_path(cls, v: str) -> str:
        # Replace backslashes so Windows paths survive JSON parsing
        return v.replace("\\", "/")


# ---------------------------------------------------------------------------
# Helpers: builders
# ---------------------------------------------------------------------------

def build_datamodule(root=None, train_batch=2, eval_batch=1,
                     workers=1, val_ratio=0.3):
    """Create an Anomalib Folder datamodule from the expected directory layout."""
    return Folder(
        name="data",
        root=root,
        normal_dir="train/OKImages",
        abnormal_dir="test/NGImages",
        normal_test_dir="test/OKImages",
        val_split_mode=ValSplitMode.FROM_TEST,
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_ratio=val_ratio,
        train_batch_size=train_batch,
        eval_batch_size=eval_batch,
        num_workers=workers,
    )


def build_model(backbone="resnet18"):
    """Instantiate the CFA model with AUROC and F1 evaluation metrics."""
    evaluator = Evaluator(test_metrics=[
        AUROC(fields=["pred_score", "gt_label"]),
        F1Score(fields=["pred_label", "gt_label"]),
    ])
    return Cfa(backbone=backbone, evaluator=evaluator, visualizer=ImageVisualizer())


def build_engine(max_epochs=1, eval_only=False, callbacks=None, lr=0.001):
    """
    Create an Anomalib Engine.
    - eval_only=True  → lightweight engine used only for inference/loading
    - eval_only=False → full training engine with checkpoint saving
    """
    if callbacks is None:
        callbacks = []

    if eval_only:
        return Engine(
            enable_progress_bar=False,
            accelerator="auto",
            callbacks=callbacks,
        )

    return Engine(
        max_epochs=max_epochs,
        lr=lr,
        accelerator="auto",
        callbacks=[
            ModelCheckpoint(
                dirpath=MODEL_DIR,
                filename="best_model",
                monitor=None,
                save_top_k=1,
                save_last=True,
            )
        ] + callbacks,
        enable_progress_bar=False,
    )


# ---------------------------------------------------------------------------
# Helpers: misc
# ---------------------------------------------------------------------------

def _normalize_path(raw: str) -> Path:
    """
    Convert a user-supplied path string to a pathlib.Path safely.

    Problem: on Windows, backslashes in a plain string literal can be
    interpreted as escape sequences (e.g. '\\A' → '\x07').  This function
    replaces all backslashes with forward slashes before constructing the
    Path object, making both styles work:
        'C:\\Python\\Anomalib\\data'  ✓
        'C:/Python/Anomalib/data'     ✓
    """
    return Path(raw.replace("\\", "/"))


def _require_trained():
    """Raise HTTP 400 if no trained checkpoint is available."""
    if CHECKPOINT.exists():
        state["trained"] = True
    if not state["trained"]:
        raise HTTPException(400, "Model not trained yet. Call /train first.")


def _save_result(item, save_path: Path):
    """Render an anomaly heatmap overlay and save it to *save_path*."""
    heatmap = visualize_image_item(
        item,
        text_config={"enable": False},
        fields=["image"],
        overlay_fields=[("image", ["anomaly_map"])],
        overlay_fields_config={
            "anomaly_map": {
                "colormap":  True,
                "normalize": False,
            }
        },
    )
    heatmap.save(save_path)


# ---------------------------------------------------------------------------
# App lifespan: load checkpoint on startup if it exists
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load a previously saved checkpoint so the server is ready to predict."""
    if CHECKPOINT.exists():
        try:
            model      = build_model()
            checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict)

            engine = build_engine(eval_only=True)
            state.update(model=model, engine=engine, trained=True)
            print(f"[startup] Loaded checkpoint from {CHECKPOINT}")
        except Exception as e:
            traceback.print_exc()
            print(f"[startup] Failed to load checkpoint: {e}")
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Anomalib CFA API",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Status ──────────────────────────────────────────────────────────────────

@app.get(
    "/status",
    tags=["Status"],
    summary="Get training status",
)
def status():
    """Return whether the model is trained/training and which device is available."""
    return {
        "trained":    state["trained"],
        "training":   state["training"],
        "checkpoint": str(CHECKPOINT) if CHECKPOINT.exists() else None,
        "device":     "cuda" if torch.cuda.is_available() else "cpu",
    }


# ── Dataset ─────────────────────────────────────────────────────────────────

@app.post(
    "/set_dataset_path",
    tags=["Dataset"],
    summary="Set dataset root path",
    description="""
Point the service to your dataset folder.

Pass the path as a **query parameter** (not JSON body) so Windows backslash  
paths like `C:\\Python\\Anomalib\\data` are sent safely — just type the path  
directly in the `dataset_path` field in Swagger UI.

Expected structure:
```
<dataset_path>/
├── train/OKImages/
├── test/NGImages/
└── test/OKImages/
```
""",
)
def set_data_path(
    dataset_path: str = Query(
        ...,
        description="Absolute path to dataset root. "
                    "Both C:\\\\data and C:/data styles are accepted.",
        example="C:/Python/Anomalib/data",
    )
):
    """
    Validate the supplied path and build a datamodule from it.

    Receives path as a Query parameter (not JSON body) so that Windows
    backslash paths are never parsed as JSON escape sequences.
    _normalize_path() does a second-pass replace just in case.
    """
    root_path = _normalize_path(dataset_path).resolve()

    if not root_path.exists():
        raise HTTPException(400, f"Folder not found: {root_path}")

    try:
        dm = build_datamodule(root=root_path.as_posix())
        dm.setup()
        state["datamodule"] = dm
        return {"message": f"Dataset path set to {root_path}", "success": True}
    except Exception as e:
        raise HTTPException(400, f"Invalid folder structure: {e}")


# ── Training ─────────────────────────────────────────────────────────────────

@app.post(
    "/train",
    tags=["Training"],
    summary="Train the CFA model (background task)",
    description="""
Start model training.  Training runs in the background; poll `/status` to track progress.

Supported backbones:
- `vgg19_bn`
- `resnet18` *(default)*
- `wide_resnet50_2`
- `efficientnet_b5`
""",
)
def train(req: TrainRequest, bg: BackgroundTasks):
    """Kick off a background training job and return immediately."""
    if state["training"]:
        raise HTTPException(409, "Training already in progress.")

    def _run():
        state["training"] = True
        try:
            # Use _normalize_path so Windows backslash paths work here too
            dm = build_datamodule(
                root=_normalize_path(req.data_root).as_posix(),
                train_batch=req.train_batch_size,
                eval_batch=req.eval_batch_size,
                workers=req.num_workers,
                val_ratio=req.val_split_ratio,
            )
            dm.setup()

            model  = build_model(backbone=req.backbone)
            engine = build_engine(max_epochs=req.max_epochs, lr=req.lr)
            engine.train(model, datamodule=dm)

            state.update(datamodule=dm, model=model, engine=engine, trained=True)
            print("[train] Training complete.")
        except Exception:
            traceback.print_exc()
            state["trained"] = False
        finally:
            state["training"] = False

    bg.add_task(_run)
    return {"message": "Training started. Poll /status to check progress."}


# ── Predict ──────────────────────────────────────────────────────────────────

@app.post(
    "/predict",
    tags=["Predict"],
    summary="Run inference on the test set",
    description="""
Run the trained model over the test dataset.  
Results (heatmap overlays) are saved to:
- `prediction_results/OK/` for normal images
- `prediction_results/NG/` for anomalous images
""",
)
def predict():
    """Predict on the current datamodule's test split and save heatmap images."""
    _require_trained()

    # Clear previous results
    for d in (RESULT_NG_DIR, RESULT_OK_DIR):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    try:
        predictions = state["engine"].predict(
            model=state["model"],
            datamodule=state["datamodule"],
        )
        metrics = state["engine"].test(
            model=state["model"],
            datamodule=state["datamodule"],
        )
        print(f"[predict] Metrics: {metrics}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))

    saved, total, ok_count, ng_count = [], 0, 0, 0

    for batch in predictions:
        for item in batch:
            total += 1
            is_ng = bool(item.pred_label)

            if is_ng:
                ng_count += 1
                fname     = f"ng_{ng_count:04d}_score{float(item.pred_score):.4f}.png"
                save_path = RESULT_NG_DIR / fname
            else:
                ok_count += 1
                fname     = f"ok_{ok_count:04d}_score{float(item.pred_score):.4f}.png"
                save_path = RESULT_OK_DIR / fname

            _save_result(item, save_path)
            saved.append(str(save_path))

    return {
        "metrics":           metrics,
        "total_test_images": total,
        "ng_count":          ng_count,
        "ok_count":          ok_count,
        "saved_files":       saved,
        "message": (
            f"Saved {ng_count} NG image(s) to {RESULT_NG_DIR} "
            f"and {ok_count} OK image(s) to {RESULT_OK_DIR}"
        ),
    }
