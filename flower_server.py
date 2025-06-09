import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics
import numpy as np


class MyceliumStrategy(fl.server.strategy.FedAvg):
    """Custom Flower strategy for Mycelium Net"""

    def __init__(self, group_id: str, **kwargs):
        super().__init__(**kwargs)
        self.group_id = group_id
        self.round_performance = []

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """Aggregate evaluation results and track performance"""

        if not results:
            return None, {}

        # Calculate weighted average accuracy
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        avg_accuracy = sum(accuracies) / sum(examples) if sum(examples) > 0 else 0
        self.round_performance.append(avg_accuracy)

        print(f"Group {self.group_id} - Round {server_round}: {avg_accuracy:.3f} accuracy")

        return avg_accuracy, {"accuracy": avg_accuracy}


def start_flower_server(group_id: str, port: int = 8080):
    """Start Flower server for a specific group"""

    strategy = MyceliumStrategy(
        group_id=group_id,
        fraction_fit=1.0,  # Use all available clients
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )

    print(f"Starting Flower server for group {group_id} on port {port}")

    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )


if __name__ == "__main__":
    import sys

    group_id = sys.argv[1] if len(sys.argv) > 1 else "default_group"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080

    start_flower_server(group_id, port)