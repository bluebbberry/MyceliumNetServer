# Mycelium Net MVP - A Meta-Federated Learning Network
# Complete implementation with Flower AI integration

# ===========================================
# 1. GLOBAL LOOKUP TABLE SERVER (registry.py)
# ===========================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import sqlite3
import json
import time
from datetime import datetime
import uuid

app = FastAPI(title="Mycelium Net Registry", version="1.0.0")


# Database setup
def init_db():
    conn = sqlite3.connect('mycelium_registry.db')
    cursor = conn.cursor()

    # Groups table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_groups (
            group_id TEXT PRIMARY KEY,
            model_type TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            performance_metric REAL DEFAULT 0.0,
            member_count INTEGER DEFAULT 0,
            max_capacity INTEGER DEFAULT 10,
            join_policy TEXT DEFAULT 'open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Nodes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            node_address TEXT NOT NULL,
            current_group_id TEXT,
            local_performance REAL DEFAULT 0.0,
            last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (current_group_id) REFERENCES learning_groups (group_id)
        )
    ''')

    conn.commit()
    conn.close()


# Pydantic models
class LearningGroup(BaseModel):
    group_id: Optional[str] = None
    model_type: str
    dataset_name: str
    performance_metric: float = 0.0
    member_count: int = 0
    max_capacity: int = 10
    join_policy: str = "open"


class Node(BaseModel):
    node_id: Optional[str] = None
    node_address: str
    current_group_id: Optional[str] = None
    local_performance: float = 0.0


class JoinRequest(BaseModel):
    node_id: str
    group_id: str


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()


# Registry endpoints
@app.post("/groups", response_model=dict)
async def create_group(group: LearningGroup):
    conn = sqlite3.connect('mycelium_registry.db')
    cursor = conn.cursor()

    group_id = str(uuid.uuid4())
    cursor.execute('''
        INSERT INTO learning_groups 
        (group_id, model_type, dataset_name, performance_metric, max_capacity, join_policy)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (group_id, group.model_type, group.dataset_name,
          group.performance_metric, group.max_capacity, group.join_policy))

    conn.commit()
    conn.close()

    return {"group_id": group_id, "status": "created"}


@app.get("/groups", response_model=List[dict])
async def list_groups():
    conn = sqlite3.connect('mycelium_registry.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM learning_groups ORDER BY performance_metric DESC')
    groups = cursor.fetchall()
    conn.close()

    return [
        {
            "group_id": g[0], "model_type": g[1], "dataset_name": g[2],
            "performance_metric": g[3], "member_count": g[4], "max_capacity": g[5],
            "join_policy": g[6], "created_at": g[7], "last_updated": g[8]
        }
        for g in groups
    ]


@app.post("/nodes/register", response_model=dict)
async def register_node(node: Node):
    conn = sqlite3.connect('mycelium_registry.db')
    cursor = conn.cursor()

    node_id = str(uuid.uuid4()) if not node.node_id else node.node_id
    cursor.execute('''
        INSERT OR REPLACE INTO nodes 
        (node_id, node_address, local_performance, last_heartbeat)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    ''', (node_id, node.node_address, node.local_performance))

    conn.commit()
    conn.close()

    return {"node_id": node_id, "status": "registered"}


@app.post("/groups/join", response_model=dict)
async def join_group(request: JoinRequest):
    conn = sqlite3.connect('mycelium_registry.db')
    cursor = conn.cursor()

    # Check if group has capacity
    cursor.execute('SELECT member_count, max_capacity FROM learning_groups WHERE group_id = ?',
                   (request.group_id,))
    result = cursor.fetchone()

    if not result:
        conn.close()
        raise HTTPException(status_code=404, detail="Group not found")

    member_count, max_capacity = result
    if member_count >= max_capacity:
        conn.close()
        raise HTTPException(status_code=400, detail="Group at capacity")

    # Update node's group and increment group member count
    cursor.execute('UPDATE nodes SET current_group_id = ? WHERE node_id = ?',
                   (request.group_id, request.node_id))
    cursor.execute('UPDATE learning_groups SET member_count = member_count + 1 WHERE group_id = ?',
                   (request.group_id,))

    conn.commit()
    conn.close()

    return {"status": "joined", "group_id": request.group_id}


@app.put("/groups/{group_id}/performance")
async def update_group_performance(group_id: str, performance: dict):
    conn = sqlite3.connect('mycelium_registry.db')
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE learning_groups 
        SET performance_metric = ?, last_updated = CURRENT_TIMESTAMP 
        WHERE group_id = ?
    ''', (performance["metric"], group_id))

    conn.commit()
    conn.close()

    return {"status": "updated"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# ===========================================
# 2. MYCELIUM NET NODE (mycelium_node.py)
# ===========================================

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


# ===========================================
# 3. DEMO SCRIPT (demo.py)
# ===========================================

def run_demo():
    """Run a demonstration of the Mycelium Network"""
    import subprocess
    import time
    import sys

    print("🌐 Starting Mycelium Net Demo")
    print("=" * 50)

    # Start registry server
    print("1. Starting registry server...")
    registry_process = subprocess.Popen([
        sys.executable, "-c",
        """
from registry import app
import uvicorn
uvicorn.run(app, host='0.0.0.0', port=8000)
        """
    ])

    time.sleep(3)  # Wait for server to start

    # Start multiple nodes
    print("2. Starting Mycelium nodes...")

    nodes = []
    for i in range(3):
        node = MyceliumNode(NodeConfig(
            node_address=f"localhost:808{i}",
            heartbeat_interval=10,
            group_evaluation_interval=15
        ))
        nodes.append(node)

    # Start nodes in separate threads
    node_threads = []
    for i, node in enumerate(nodes):
        thread = threading.Thread(target=node.start, daemon=True)
        thread.start()
        node_threads.append(thread)
        time.sleep(2)  # Stagger node starts

    print("3. Nodes are running and training...")
    print("   - Check http://localhost:8000/groups for group status")
    print("   - Press Ctrl+C to stop demo")

    try:
        # Keep demo running
        while True:
            time.sleep(10)
            print(f"Demo running... {len(nodes)} nodes active")
    except KeyboardInterrupt:
        print("\n4. Stopping demo...")
        for node in nodes:
            node.stop()
        registry_process.terminate()
        print("Demo stopped.")


if __name__ == "__main__":
    # You can run different components
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "registry":
            # Run registry server
            exec(open("registry.py").read())
        elif sys.argv[1] == "node":
            # Run single node
            node = MyceliumNode(NodeConfig())
            node.start()
        elif sys.argv[1] == "demo":
            # Run full demo
            run_demo()
    else:
        print("Usage:")
        print("  python mycelium_net.py registry  # Start registry server")
        print("  python mycelium_net.py node     # Start single node")
        print("  python mycelium_net.py demo     # Run full demo")

# ===========================================
# 4. REQUIREMENTS (requirements.txt)
# ===========================================

"""
fastapi>=0.104.1
uvicorn>=0.24.0
pydantic>=2.5.0
requests>=2.31.0
flwr>=1.6.0
torch>=2.1.0
numpy>=1.24.0
sqlite3  # Built into Python
"""