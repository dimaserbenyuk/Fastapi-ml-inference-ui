from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import joblib
import pandas as pd
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from fastapi.staticfiles import StaticFiles

os.environ["HF_HUB_CACHE"] = "/tmp/huggingface"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/tmp", StaticFiles(directory="/tmp"), name="tmp")

REPO_ID = "DmytroSerbeniuk/my-iris-model"
MODEL_FILENAME = "model.joblib"
METRICS_PATH = "/tmp/iris_metrics.json"
PLOT_PATH = "/tmp/accuracy_plot.png"

model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, cache_dir="/tmp")
model = joblib.load(model_path)

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        metrics_history = json.load(f)
else:
    metrics_history = []


metrics = metrics_history[-1] if metrics_history else {}

def generate_accuracy_plot():
    if metrics_history:
        timestamps = [m["timestamp"] for m in metrics_history]
        accuracies = [m["accuracy"] for m in metrics_history]
        plt.figure(figsize=(8, 4))
        plt.plot(timestamps, accuracies, marker='o', linestyle='-')
        plt.title("Accuracy Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        os.makedirs("/tmp", exist_ok=True)
        plt.savefig(PLOT_PATH)
        plt.close()

generate_accuracy_plot()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("form.html", {
        "request": request,
        "metrics": metrics,
        "metrics_history": metrics_history,
        "plot_path": "/tmp/accuracy_plot.png" if os.path.exists(PLOT_PATH) else None
    })

@app.post("/predict-form", response_class=HTMLResponse)
async def predict_from_form(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...),
):
    data = pd.DataFrame([{
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }])
    prediction = model.predict(data)[0]

    y_true = [prediction] 
    y_pred = [prediction]
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    precision = report["weighted avg"]["precision"]

    new_metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "accuracy": round(acc, 4),
        "precision": round(precision, 4)
    }

    metrics_history.append(new_metrics)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_history, f, indent=2)
    generate_accuracy_plot()

    return templates.TemplateResponse("form.html", {
        "request": request,
        "prediction": prediction,
        "metrics": new_metrics,
        "metrics_history": metrics_history,
        "plot_path": "/tmp/accuracy_plot.png" if os.path.exists(PLOT_PATH) else None
    })

@app.post("/predict")
def predict_api(features: IrisFeatures):
    df = pd.DataFrame([features.dict()])
    prediction = model.predict(df)[0]
    return {"prediction": prediction, "metrics": metrics, "history": metrics_history}
