import asyncio
import requests
import json
import time
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
import threading
import random

# Flower imports
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
    heartbeat_interval: int = 30  # seconds
    group_evaluation_interval: int = 60  # seconds


class SimpleNet(nn.Module):
    """Simple neural network for demonstration"""

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MyceliumFlowerClient(NumPyClient):
    """Flower client that integrates with Mycelium Net"""

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
        for epoch in range(1):  # 1 epoch per round for speed
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
    """A node in the Mycelium Network"""

    def __init__(self, config: NodeConfig):
        self.config = config
        self.node_id: Optional[str] = None
        self.current_group_id: Optional[str] = None
        self.local_performance: float = 0.0
        self.model = SimpleNet()
        self.running = False

        # Generate synthetic training data for demo
        self.trainloader, self.testloader = self._create_demo_data()

    def _create_demo_data(self):
        """Create synthetic dataset for demonstration"""
        # Generate random data (normally you'd load real datasets)
        X_train = torch.randn(1000, 784)
        y_train = torch.randint(0, 10, (1000,))
        X_test = torch.randn(200, 784)
        y_test = torch.randint(0, 10, (200,))

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=32)

        return trainloader, testloader

    def register_to_registry(self) -> bool:
        """Register this node with the global registry"""
        try:
            response = requests.post(f"{self.config.registry_url}/nodes/register",
                                     json={
                                         "node_address": self.config.node_address,
                                         "local_performance": self.local_performance
                                     })
            if response.status_code == 200:
                self.node_id = response.json()["node_id"]
                logger.info(f"Node registered with ID: {self.node_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to register node: {e}")
        return False

    def discover_groups(self) -> List[Dict]:
        """Discover available learning groups"""
        try:
            response = requests.get(f"{self.config.registry_url}/groups")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to discover groups: {e}")
        return []

    def join_group(self, group_id: str) -> bool:
        """Join a specific learning group"""
        try:
            response = requests.post(f"{self.config.registry_url}/groups/join",
                                     json={
                                         "node_id": self.node_id,
                                         "group_id": group_id
                                     })
            if response.status_code == 200:
                self.current_group_id = group_id
                logger.info(f"Joined group: {group_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to join group {group_id}: {e}")
        return False

    def create_group(self, model_type: str = "SimpleNet", dataset_name: str = "synthetic") -> Optional[str]:
        """Create a new learning group"""
        try:
            response = requests.post(f"{self.config.registry_url}/groups",
                                     json={
                                         "model_type": model_type,
                                         "dataset_name": dataset_name,
                                         "performance_metric": self.local_performance,
                                         "max_capacity": 5
                                     })
            if response.status_code == 200:
                group_id = response.json()["group_id"]
                logger.info(f"Created group: {group_id}")
                return group_id
        except Exception as e:
            logger.error(f"Failed to create group: {e}")
        return None

    def evaluate_group_switch(self):
        """Evaluate whether to switch to a better performing group"""
        groups = self.discover_groups()
        if not groups:
            return

        # Find groups with better performance than current
        better_groups = [g for g in groups
                         if g["performance_metric"] > self.local_performance + 0.05  # 5% threshold
                         and g["member_count"] < g["max_capacity"]
                         and g["group_id"] != self.current_group_id]

        if better_groups:
            # Join the best performing group with available capacity
            best_group = max(better_groups, key=lambda x: x["performance_metric"])
            logger.info(f"Switching to better group {best_group['group_id']} "
                        f"(performance: {best_group['performance_metric']:.3f})")
            self.join_group(best_group["group_id"])

    def start_federated_training(self):
        """Start federated learning with Flower"""
        if not self.current_group_id:
            logger.warning("No group joined, cannot start federated training")
            return

        # Create Flower client
        client = MyceliumFlowerClient(self.model, self.trainloader, self.testloader, self)

        # Start Flower client (in a real scenario, this would connect to a Flower server)
        # For demo purposes, we'll simulate training rounds
        logger.info(f"Starting federated training for group {self.current_group_id}")

        # Simulate training rounds
        for round_num in range(3):
            logger.info(f"Training round {round_num + 1}")

            # Simulate getting global parameters
            global_params = client.get_parameters({})

            # Local training
            updated_params, num_examples, metrics = client.fit(global_params, {})

            # Local evaluation
            loss, num_test_examples, eval_metrics = client.evaluate(updated_params, {})

            logger.info(f"Round {round_num + 1} - Loss: {loss:.3f}, "
                        f"Accuracy: {eval_metrics.get('accuracy', 0):.3f}")

            # Update registry with new performance
            try:
                requests.put(f"{self.config.registry_url}/groups/{self.current_group_id}/performance",
                             json={"metric": eval_metrics.get('accuracy', 0)})
            except Exception as e:
                logger.error(f"Failed to update group performance: {e}")

            time.sleep(2)  # Simulate time between rounds

    def heartbeat_loop(self):
        """Send periodic heartbeats and evaluate group switches"""
        while self.running:
            try:
                # Send heartbeat (re-register)
                self.register_to_registry()

                # Evaluate potential group switches
                self.evaluate_group_switch()

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

            time.sleep(self.config.heartbeat_interval)

    def start(self):
        """Start the Mycelium node"""
        logger.info("Starting Mycelium node...")

        # Register with registry
        if not self.register_to_registry():
            logger.error("Failed to register node")
            return

        self.running = True

        # Start heartbeat in background
        heartbeat_thread = threading.Thread(target=self.heartbeat_loop, daemon=True)
        heartbeat_thread.start()

        # Try to join an existing group or create a new one
        groups = self.discover_groups()
        if groups:
            # Join the best available group
            available_groups = [g for g in groups if g["member_count"] < g["max_capacity"]]
            if available_groups:
                best_group = max(available_groups, key=lambda x: x["performance_metric"])
                self.join_group(best_group["group_id"])

        # If no group joined, create a new one
        if not self.current_group_id:
            group_id = self.create_group()
            if group_id:
                self.join_group(group_id)

        # Start federated training
        if self.current_group_id:
            self.start_federated_training()

        logger.info("Node started successfully")

    def stop(self):
        """Stop the node"""
        logger.info("Stopping Mycelium node...")
        self.running = False


if __name__ == "__main__":
    node = MyceliumNode(NodeConfig())
    try:
        node.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node.stop()