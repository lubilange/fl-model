import os
import threading
import torch

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from authexample.task import Net

# =========================
# CONFIG
# =========================
PORT = int(os.environ.get("PORT", 8080))
NUM_SERVER_ROUNDS = int(os.environ.get("NUM_SERVER_ROUNDS", 10))

# =========================
# GLOBAL MODEL
# =========================
global_model = Net()
model_lock = threading.Lock()

# =========================
# FLOWER SERVER
# =========================
flwr_app = ServerApp()


@flwr_app.main()
def main(grid: Grid, context: Context) -> None:
    print(" Flower Server started")

    with model_lock:
        initial_weights = ArrayRecord(global_model.state_dict())

    strategy = FedAvg()

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_weights,
        train_config=ConfigRecord({}),
        num_rounds=NUM_SERVER_ROUNDS,
        evaluate_fn=None  # 
    )

    final_state = result.arrays.to_torch_state_dict()

    with model_lock:
        global_model.load_state_dict(final_state)

    torch.save(final_state, "global_model.pt")

    print(" Training finished")


# =========================
# FLASK (ONLY MODEL DOWNLOAD)
# =========================
from flask import Flask, send_file
import io

app = Flask(__name__)


@app.route("/")
def home():
    return "FL server running"


@app.route("/get_model")
def get_model():
    with model_lock:
        buffer = io.BytesIO()
        torch.save(global_model.state_dict(), buffer)
        buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="global_model.pt")


# =========================
# RUN
# =========================
def run_flower():
    flwr_app.main()


if __name__ == "__main__":
    threading.Thread(target=run_flower, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)
