import os
import threading
import torch
from flask import Flask

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import DifferentialPrivacyServerSideFixedClipping, FedAvg
from authexample.task import Net

# --- PORT Render ---
PORT = int(os.environ.get("PORT", 8080))

# --- Config ---
FRACTION_EVALUATE = float(os.environ.get("FRACTION_EVALUATE", 0.5))
NUM_SERVER_ROUNDS = int(os.environ.get("NUM_SERVER_ROUNDS", 10))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.001))
NOISE_MULTIPLIER = float(os.environ.get("NOISE_MULTIPLIER", 0.3))
CLIPPING_NORM = float(os.environ.get("CLIPPING_NORM", 1.0))
NUM_SAMPLED_CLIENTS = int(os.environ.get("NUM_SAMPLED_CLIENTS", 2))

# --- Flower App ---
flwr_app = ServerApp()

@flwr_app.main()
def main(grid: Grid, context: Context) -> None:

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    base_strategy = FedAvg(fraction_evaluate=FRACTION_EVALUATE)

    dp_strategy = DifferentialPrivacyServerSideFixedClipping(
        base_strategy,
        noise_multiplier=NOISE_MULTIPLIER,
        clipping_norm=CLIPPING_NORM,
        num_sampled_clients=NUM_SAMPLED_CLIENTS,
    )

    result = dp_strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": LEARNING_RATE}),
        num_rounds=NUM_SERVER_ROUNDS,
        evaluate_fn=global_evaluate,
    )

    torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    return MetricRecord({"accuracy": 0.0, "loss": 0.0})


# --- Flask (pour Render) ---
app = Flask(__name__)

@app.route("/")
def home():
    return "Flower server is running 🚀"


# --- Lancer Flower en parallèle ---
def run_flower():
    flwr_app.main()  # lance Flower


if __name__ == "__main__":
    print(f"Starting services on port {PORT}...")

    # Thread Flower
    threading.Thread(target=run_flower).start()

    # Serveur HTTP (obligatoire pour Render)
    app.run(host="0.0.0.0", port=PORT)
