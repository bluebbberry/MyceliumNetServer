import asyncio
import json
import pickle
import hashlib
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
import threading
import socket
import struct
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PeerInfo:
    """Information about a peer in the network"""
    peer_id: str
    host: str
    port: int
    last_seen: float
    model_hash: str = ""


@dataclass
class ModelUpdate:
    """Represents a model update (gradients) to be shared"""
    sender_id: str
    gradients: Dict[str, np.ndarray]
    timestamp: float
    compression_ratio: float = 1.0
    round_number: int = 0


class SimpleModel(nn.Module):
    """Simple neural network for demonstration"""

    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class GradientCompressor:
    """Handles gradient compression to reduce bandwidth"""

    @staticmethod
    def top_k_compression(gradients: Dict[str, torch.Tensor], k_ratio: float = 0.1) -> Tuple[
        Dict[str, np.ndarray], float]:
        """Compress gradients by keeping only top-k values"""
        compressed = {}
        total_params = 0
        kept_params = 0

        for name, grad in gradients.items():
            if grad is None:
                continue

            grad_flat = grad.flatten()
            total_params += len(grad_flat)

            # Keep top k% of gradients by magnitude
            k = max(1, int(len(grad_flat) * k_ratio))
            _, indices = torch.topk(torch.abs(grad_flat), k)

            # Create sparse representation
            sparse_grad = torch.zeros_like(grad_flat)
            sparse_grad[indices] = grad_flat[indices]

            compressed[name] = {
                'values': sparse_grad.numpy(),
                'shape': grad.shape
            }
            kept_params += k

        compression_ratio = kept_params / total_params if total_params > 0 else 1.0
        return compressed, compression_ratio

    @staticmethod
    def decompress_gradients(compressed: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Decompress gradients back to torch tensors"""
        decompressed = {}
        for name, data in compressed.items():
            if isinstance(data, dict) and 'values' in data:
                values = torch.from_numpy(data['values']).float()
                decompressed[name] = values.reshape(data['shape'])
            else:
                decompressed[name] = torch.from_numpy(data).float()
        return decompressed


class P2PNetwork:
    """Handles P2P networking for the federated learning system"""

    def __init__(self, host: str = 'localhost', port: int = 8000):
        self.host = host
        self.port = port
        self.peer_id = self._generate_peer_id()
        self.peers: Dict[str, PeerInfo] = {}
        self.server_socket = None
        self.running = False
        self.message_handlers = {}

    def _generate_peer_id(self) -> str:
        """Generate unique peer ID"""
        return hashlib.sha256(f"{self.host}:{self.port}:{time.time()}".encode()).hexdigest()[:16]

    async def start_server(self):
        """Start the P2P server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True

        logger.info(f"P2P server started on {self.host}:{self.port} (ID: {self.peer_id})")

        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                threading.Thread(target=self._handle_client, args=(client_socket,)).start()
            except Exception as e:
                if self.running:
                    logger.error(f"Server error: {e}")

    def _handle_client(self, client_socket):
        """Handle incoming client connections"""
        try:
            # Receive message length
            length_data = client_socket.recv(4)
            if not length_data:
                return

            message_length = struct.unpack('!I', length_data)[0]

            # Receive message
            message_data = b''
            while len(message_data) < message_length:
                chunk = client_socket.recv(min(4096, message_length - len(message_data)))
                if not chunk:
                    break
                message_data += chunk

            if len(message_data) == message_length:
                message = pickle.loads(message_data)
                self._process_message(message)

        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            client_socket.close()

    def _process_message(self, message: Dict):
        """Process received messages"""
        msg_type = message.get('type')
        if msg_type in self.message_handlers:
            self.message_handlers[msg_type](message)

    def register_handler(self, message_type: str, handler):
        """Register message handler"""
        self.message_handlers[message_type] = handler

    async def send_message(self, peer_info: PeerInfo, message: Dict) -> bool:
        """Send message to a peer"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)  # 10 second timeout
            sock.connect((peer_info.host, peer_info.port))

            # Serialize message
            message_data = pickle.dumps(message)
            message_length = len(message_data)

            # Send length first, then message
            sock.send(struct.pack('!I', message_length))
            sock.send(message_data)

            sock.close()
            return True

        except Exception as e:
            logger.error(f"Failed to send message to {peer_info.peer_id}: {e}")
            return False

    def add_peer(self, host: str, port: int, peer_id: str = None):
        """Add a peer to the network"""
        if not peer_id:
            peer_id = f"{host}:{port}"

        peer_info = PeerInfo(
            peer_id=peer_id,
            host=host,
            port=port,
            last_seen=time.time()
        )
        self.peers[peer_id] = peer_info
        logger.info(f"Added peer: {peer_id}")

    def get_active_peers(self, timeout: float = 300) -> List[PeerInfo]:
        """Get list of active peers (seen within timeout seconds)"""
        current_time = time.time()
        active_peers = []

        for peer in self.peers.values():
            if current_time - peer.last_seen < timeout:
                active_peers.append(peer)

        return active_peers

    def stop(self):
        """Stop the P2P network"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()


class FederatedLearningNode:
    """Main federated learning node that combines P2P networking with ML training"""

    def __init__(self, host: str = 'localhost', port: int = 8000, model_params: Dict = None):
        self.network = P2PNetwork(host, port)
        self.model = SimpleModel(**(model_params or {}))
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.compressor = GradientCompressor()

        # Federated learning state
        self.round_number = 0
        self.local_updates = []
        self.aggregated_gradients = {}
        self.training_data = None

        # Register message handlers
        self.network.register_handler('model_update', self._handle_model_update)
        self.network.register_handler('peer_discovery', self._handle_peer_discovery)
        self.network.register_handler('training_round', self._handle_training_round)

    def set_training_data(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32):
        """Set training data for the node"""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        self.training_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        logger.info(f"Training data set: {len(X)} samples")

    def _handle_model_update(self, message: Dict):
        """Handle incoming model updates from peers"""
        try:
            update = ModelUpdate(**message['data'])

            # Decompress gradients
            gradients = self.compressor.decompress_gradients(update.gradients)

            # Store update for aggregation
            self.local_updates.append({
                'sender': update.sender_id,
                'gradients': gradients,
                'timestamp': update.timestamp,
                'round': update.round_number
            })

            logger.info(f"Received model update from {update.sender_id} "
                        f"(compression: {update.compression_ratio:.2%})")

        except Exception as e:
            logger.error(f"Error handling model update: {e}")

    def _handle_peer_discovery(self, message: Dict):
        """Handle peer discovery messages"""
        peer_data = message['data']
        peer_id = peer_data['peer_id']

        if peer_id != self.network.peer_id:
            self.network.add_peer(
                host=peer_data['host'],
                port=peer_data['port'],
                peer_id=peer_id
            )

    def _handle_training_round(self, message: Dict):
        """Handle training round initiation"""
        round_data = message['data']
        self.round_number = max(self.round_number, round_data['round_number'])
        logger.info(f"Starting training round {self.round_number}")

        # Trigger local training
        asyncio.create_task(self.train_local_model())

    async def start(self):
        """Start the federated learning node"""
        await self.network.start_server()

    async def train_local_model(self, epochs: int = 1) -> Dict[str, torch.Tensor]:
        """Train the local model and return gradients"""
        if not self.training_data:
            logger.warning("No training data set")
            return {}

        self.model.train()
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.training_data):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

        # Calculate gradients (difference between final and initial parameters)
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Use actual gradients from last batch
                gradients[name] = param.grad.clone()
            else:
                # Calculate parameter difference as proxy for average gradient
                gradients[name] = param.data - initial_params[name]

        logger.info(f"Local training completed for round {self.round_number}")
        return gradients

    async def share_model_update(self, gradients: Dict[str, torch.Tensor]):
        """Share model update with peers"""
        if not gradients:
            return

        # Compress gradients
        compressed_gradients, compression_ratio = self.compressor.top_k_compression(gradients)

        # Create update message
        update = ModelUpdate(
            sender_id=self.network.peer_id,
            gradients=compressed_gradients,
            timestamp=time.time(),
            compression_ratio=compression_ratio,
            round_number=self.round_number
        )

        message = {
            'type': 'model_update',
            'data': asdict(update)
        }

        # Send to all active peers
        active_peers = self.network.get_active_peers()
        success_count = 0

        for peer in active_peers:
            if await self.network.send_message(peer, message):
                success_count += 1

        logger.info(f"Shared update with {success_count}/{len(active_peers)} peers "
                    f"(compression: {compression_ratio:.2%})")

    def aggregate_updates(self):
        """Aggregate received model updates"""
        if not self.local_updates:
            return

        # Simple averaging of gradients
        aggregated = defaultdict(list)

        for update in self.local_updates:
            for name, grad in update['gradients'].items():
                aggregated[name].append(grad)

        # Average gradients
        final_gradients = {}
        for name, grad_list in aggregated.items():
            if grad_list:
                stacked = torch.stack(grad_list)
                final_gradients[name] = torch.mean(stacked, dim=0)

        # Apply aggregated gradients
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in final_gradients:
                    param.data -= 0.01 * final_gradients[name]  # Simple gradient descent

        logger.info(f"Aggregated {len(self.local_updates)} updates")
        self.local_updates.clear()

    async def run_federated_round(self):
        """Run a complete federated learning round"""
        logger.info(f"Starting federated learning round {self.round_number}")

        # 1. Train local model
        gradients = await self.train_local_model()

        # 2. Share updates with peers
        await self.share_model_update(gradients)

        # 3. Wait for peer updates (in practice, you'd want more sophisticated synchronization)
        await asyncio.sleep(10)

        # 4. Aggregate received updates
        self.aggregate_updates()

        self.round_number += 1
        logger.info(f"Completed federated learning round {self.round_number - 1}")

    async def discover_peers(self, bootstrap_peers: List[Tuple[str, int]]):
        """Discover peers in the network"""
        discovery_message = {
            'type': 'peer_discovery',
            'data': {
                'peer_id': self.network.peer_id,
                'host': self.network.host,
                'port': self.network.port
            }
        }

        for host, port in bootstrap_peers:
            peer_info = PeerInfo(peer_id=f"{host}:{port}", host=host, port=port, last_seen=time.time())
            await self.network.send_message(peer_info, discovery_message)
            self.network.add_peer(host, port)

    def get_model_accuracy(self, test_data: DataLoader) -> float:
        """Evaluate model accuracy on test data"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_data:
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return correct / total if total > 0 else 0.0

    def stop(self):
        """Stop the federated learning node"""
        self.network.stop()


# Example usage and testing
async def create_test_node(port: int, training_data: Tuple[np.ndarray, np.ndarray]):
    """Create a test node with sample data"""
    node = FederatedLearningNode(port=port)

    # Set training data
    X, y = training_data
    node.set_training_data(X, y)

    # Start the node in a separate task
    server_task = asyncio.create_task(node.start())

    return node, server_task


async def run_demo():
    """Demonstrate the P2P federated learning system"""
    print("Starting P2P Federated Learning Demo...")

    # Create synthetic data for demonstration
    np.random.seed(42)

    # Node 1 data (focused on classes 0-4)
    X1 = np.random.randn(1000, 784)
    y1 = np.random.randint(0, 5, 1000)

    # Node 2 data (focused on classes 5-9)
    X2 = np.random.randn(1000, 784)
    y2 = np.random.randint(5, 10, 1000)

    # Create test data
    X_test = np.random.randn(200, 784)
    y_test = np.random.randint(0, 10, 200)
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=32)

    try:
        # Create nodes
        node1, task1 = await create_test_node(8001, (X1, y1))
        await asyncio.sleep(1)  # Let first node start

        node2, task2 = await create_test_node(8002, (X2, y2))
        await asyncio.sleep(1)  # Let second node start

        # Connect nodes
        await node2.discover_peers([('localhost', 8001)])
        await asyncio.sleep(2)

        print(f"Node 1 peers: {len(node1.network.get_active_peers())}")
        print(f"Node 2 peers: {len(node2.network.get_active_peers())}")

        # Run federated learning rounds
        for round_num in range(3):
            print(f"\n--- Round {round_num + 1} ---")

            # Run training rounds concurrently
            await asyncio.gather(
                node1.run_federated_round(),
                node2.run_federated_round()
            )

            # Evaluate models
            acc1 = node1.get_model_accuracy(test_loader)
            acc2 = node2.get_model_accuracy(test_loader)

            print(f"Node 1 accuracy: {acc1:.3f}")
            print(f"Node 2 accuracy: {acc2:.3f}")

        print("\nDemo completed successfully!")

    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        # Cleanup
        node1.stop()
        node2.stop()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_demo())
