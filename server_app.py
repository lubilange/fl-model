"""authexample: A Flower server app ready for Render deployment."""

import os
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import DifferentialPrivacyServerSideFixedClipping, FedAvg
from authexample.task import Net, load_centralized_dataset, test

# --- Récupérer le port dynamique pour Render ---
PORT = int(os.environ.get("PORT", 8080))

# --- Paramètres d'entraînement via variables d'environnement ---
FRACTION_EVALUATE = float(os.environ.get("FRACTION_EVALUATE", 0.5))
NUM_SERVER_ROUNDS = int(os.environ.get("NUM_SERVER_ROUNDS", 10))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.001))
NOISE_MULTIPLIER = float(os.environ.get("NOISE_MULTIPLIER", 0.3))
CLIPPING_NORM = float(os.environ.get("CLIPPING_NORM", 1.0))
NUM_SAMPLED_CLIENTS = int(os.environ.get("NUM_SAMPLED_CLIENTS", 2))

# --- Créer le serveur ---
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Charger le modèle global
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialiser FedAvg
    base_strategy = FedAvg(fraction_evaluate=FRACTION_EVALUATE)

    # Wrap FedAvg avec Differential Privacy
    dp_strategy = DifferentialPrivacyServerSideFixedClipping(
        base_strategy,
        noise_multiplier=NOISE_MULTIPLIER,
        clipping_norm=CLIPPING_NORM,
        num_sampled_clients=NUM_SAMPLED_CLIENTS,
    )

    # Démarrer l'entraînement fédéré
    result = dp_strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": LEARNING_RATE}),
        num_rounds=NUM_SERVER_ROUNDS,
        evaluate_fn=global_evaluate,
    )

    # Sauvegarder le modèle final
    print("\nSaving final model to disk...")
    torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
    print("Model saved as final_model.pt")

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Évaluer le modèle global sur le dataset centralisé."""
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Charger tout le dataset de test
    test_dataloader = load_centralized_dataset()

    # Evaluer le modèle global
    test_loss, test_acc = test(model, test_dataloader, device)

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})

# --- Point d'entrée pour Render ---
if __name__ == "__main__":
    print(f"Starting Flower Server on port {PORT}...")
    app.run(port=PORT)