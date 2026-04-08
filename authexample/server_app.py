import os
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import DifferentialPrivacyServerSideFixedClipping, FedAvg
from authexample.task import Net

PORT = int(os.environ.get("PORT", 8080))

FRACTION_EVALUATE = float(os.environ.get("FRACTION_EVALUATE", 0.5))
NUM_SERVER_ROUNDS = int(os.environ.get("NUM_SERVER_ROUNDS", 10))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.001))
NOISE_MULTIPLIER = float(os.environ.get("NOISE_MULTIPLIER", 0.3))
CLIPPING_NORM = float(os.environ.get("CLIPPING_NORM", 1.0))
NUM_SAMPLED_CLIENTS = int(os.environ.get("NUM_SAMPLED_CLIENTS", 2))

app = ServerApp()

@app.main()
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
