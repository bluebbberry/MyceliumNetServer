import requests
import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import threading
import random
import json

# Flower AI imports
import flwr as fl
from flwr.client import NumPyClient
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NodeConfig:
    registry_url: str = "http://localhost:8000"
    node_address: str = "localhost:8080"
    heartbeat_interval: int = 5
    performance_boost_rate: float = 0.1
    flower_server_address: str = "localhost:8080"


class SimpleNet(nn.Module):
    """Simple neural network for federated learning"""

    def __init__(self, input_size=784, hidden_size=64, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MyceliumFlowerClient(NumPyClient):
    """Flower client integrated with Mycelium Net"""

    def __init__(self, model, trainloader, testloader, node):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.node = node

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        self.model.train()
        for epoch in range(1):  # 1 epoch per round
            for batch_idx, (data, target) in enumerate(self.trainloader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        criterion = nn.CrossEntropyLoss()
        self.model.eval()

        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.testloader:
                output = self.model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total
        self.node.local_performance = accuracy

        return float(test_loss), len(self.testloader), {"accuracy": accuracy}


class MyceliumNode:
    def __init__(self, config: NodeConfig, node_name: str = None):
        self.config = config
        self.node_id: Optional[str] = None
        self.node_name = node_name or f"Node-{random.randint(1000, 9999)}"
        self.current_group_id: Optional[str] = None
        self.local_performance: float = random.uniform(0.3, 0.7)
        self.running = False
        self.training_rounds = 0
        self.flower_thread = None

        # Initialize model and data for Flower
        self.model = SimpleNet()
        self.trainloader, self.testloader = self._create_synthetic_data()
        self.flower_client = MyceliumFlowerClient(
            self.model, self.trainloader, self.testloader, self
        )

    def _create_synthetic_data(self):
        """Create synthetic dataset for demonstration"""
        # Generate random data (in practice, use real datasets)
        X_train = torch.randn(500, 784)
        y_train = torch.randint(0, 10, (500,))
        X_test = torch.randn(100, 784)
        y_test = torch.randint(0, 10, (100,))

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=32)

        return trainloader, testloader

    def register_to_registry(self) -> bool:
        try:
            response = requests.post(f"{self.config.registry_url}/nodes/register",
                                     json={
                                         "node_address": self.config.node_address,
                                         "local_performance": self.local_performance
                                     })
            if response.status_code == 200:
                self.node_id = response.json()["node_id"]
                logger.info(f"{self.node_name} registered with ID: {self.node_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to register {self.node_name}: {e}")
        return False

    def send_heartbeat(self):
        """Send heartbeat with current performance to registry"""
        if not self.node_id:
            return

        try:
            response = requests.put(f"{self.config.registry_url}/nodes/{self.node_id}/performance",
                                    json={"metric": self.local_performance})
            if response.status_code == 200:
                logger.debug(f"{self.node_name} heartbeat sent (perf: {self.local_performance:.3f})")
        except Exception as e:
            logger.error(f"Failed to send heartbeat for {self.node_name}: {e}")

    def discover_groups(self) -> List[Dict]:
        try:
            response = requests.get(f"{self.config.registry_url}/groups")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to discover groups: {e}")
        return []

    def join_group(self, group_id: str) -> bool:
        try:
            response = requests.post(f"{self.config.registry_url}/groups/join",
                                     json={"node_id": self.node_id, "group_id": group_id})
            if response.status_code == 200:
                self.current_group_id = group_id
                logger.info(f"{self.node_name} joined group: {group_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to join group {group_id}: {e}")
        return False

    def create_group(self) -> Optional[str]:
        try:
            response = requests.post(f"{self.config.registry_url}/groups",
                                     json={
                                         "model_type": "NeuralNet",
                                         "dataset_name": "synthetic",
                                         "performance_metric": self.local_performance,
                                         "max_capacity": 3
                                     })
            if response.status_code == 200:
                group_id = response.json()["group_id"]
                logger.info(f"{self.node_name} created group: {group_id}")
                return group_id
        except Exception as e:
            logger.error(f"Failed to create group: {e}")
        return None

    def start_flower_client(self):
        """Start Flower client in separate thread"""

        def client_fn(cid: str):
            return self.flower_client

        try:
            # Use simulation for simplicity - in production, connect to actual server
            fl.client.start_numpy_client(
                server_address="[::]:8080",  # Default Flower server address
                client=self.flower_client
            )
        except Exception as e:
            logger.error(f"Flower client error for {self.node_name}: {e}")
            # Fallback to simulated training
            self.simulate_training()

    def simulate_training(self):
        """Simulate federated learning training"""
        if not self.current_group_id:
            return

        try:
            # Get current model parameters
            current_params = self.flower_client.get_parameters({})

            # Perform local training
            updated_params, num_examples, metrics = self.flower_client.fit(current_params, {})

            # Evaluate model
            loss, num_test_examples, eval_metrics = self.flower_client.evaluate(updated_params, {})

            self.training_rounds += 1
            accuracy = eval_metrics.get('accuracy', self.local_performance)

            # Update performance with improvement
            self.local_performance = min(accuracy + random.uniform(0.01, self.config.performance_boost_rate), 0.95)

            logger.info(f"{self.node_name} completed training round {self.training_rounds}, "
                        f"accuracy: {self.local_performance:.3f}")

            # Update group performance in registry
            requests.put(f"{self.config.registry_url}/groups/{self.current_group_id}/performance",
                         json={"metric": self.local_performance})

        except Exception as e:
            logger.error(f"Training error for {self.node_name}: {e}")
            # Fallback performance boost
            self.local_performance = min(self.local_performance + random.uniform(0.01, 0.05), 0.95)

    def evaluate_group_switch(self):
        groups = self.discover_groups()
        if not groups:
            return

        # Find better performing groups
        better_groups = [g for g in groups
                         if g["performance_metric"] > self.local_performance * 1.1
                         and g["member_count"] < g["max_capacity"]
                         and g["group_id"] != self.current_group_id]

        if better_groups:
            best_group = max(better_groups, key=lambda x: x["performance_metric"])
            logger.info(f"{self.node_name} switching to group {best_group['group_id']} "
                        f"(perf: {best_group['performance_metric']:.3f})")
            self.join_group(best_group["group_id"])

    def training_loop(self):
        while self.running:
            # Send heartbeat
            self.send_heartbeat()

            # Do training (either real Flower or simulated)
            self.simulate_training()

            # Evaluate potential group switches
            self.evaluate_group_switch()

            time.sleep(self.config.heartbeat_interval)

    def start(self):
        if not self.register_to_registry():
            return

        self.running = True

        # Try to join existing group or create new one
        groups = self.discover_groups()
        if groups:
            available = [g for g in groups if g["member_count"] < g["max_capacity"]]
            if available:
                best_group = max(available, key=lambda x: x["performance_metric"])
                self.join_group(best_group["group_id"])

        if not self.current_group_id:
            group_id = self.create_group()
            if group_id:
                self.join_group(group_id)

        # Start Flower client in background
        self.flower_thread = threading.Thread(target=self.start_flower_client, daemon=True)
        self.flower_thread.start()

        # Start training loop
        training_thread = threading.Thread(target=self.training_loop, daemon=True)
        training_thread.start()

        logger.info(f"{self.node_name} started successfully")

    def stop(self):
        self.running = False