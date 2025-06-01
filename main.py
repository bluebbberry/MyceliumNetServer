#!/usr/bin/env python3
"""
Simplified Gossip Learning for Music Recommendations
No complex JSON serialization - just basic Python types
"""

import json
import sys
import time
import threading
import socket
import random
import pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class Config:
    """Simple configuration class"""
    host: str = "localhost"
    port: int = 8000
    peers: List[Tuple[str, int]] = None
    data_file: str = "ratings.csv"
    gossip_rounds: int = 10
    gossip_interval: float = 2.0

    @classmethod
    def load(cls, filename="config.json"):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            return cls(
                host=data.get("host", "localhost"),
                port=data.get("port", 8000),
                peers=[tuple(p) for p in data.get("peers", [])],
                data_file=data.get("data_file", "ratings.csv"),
                gossip_rounds=data.get("gossip_rounds", 10),
                gossip_interval=data.get("gossip_interval", 2.0)
            )
        except Exception as e:
            print(f"Error loading config: {e}")
            return cls()


class SimpleModel:
    """Very simple collaborative filtering model"""

    def __init__(self):
        self.user_ratings = {}  # user_id -> {song_id: rating}
        self.user_averages = {}  # user_id -> average_rating
        self.song_averages = {}  # song_id -> average_rating
        self.global_average = 0.0
        self.is_trained = False

    def load_data(self, filename):
        """Load CSV data: user_id,song_id,rating"""
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()[1:]  # Skip header

            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    user_id = int(parts[0])
                    song_id = int(parts[1])
                    rating = float(parts[2])

                    if user_id not in self.user_ratings:
                        self.user_ratings[user_id] = {}
                    self.user_ratings[user_id][song_id] = rating

            self.train()
            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def train(self):
        """Train the model - compute averages"""
        if not self.user_ratings:
            return False

        # Compute user averages
        for user_id, ratings in self.user_ratings.items():
            self.user_averages[user_id] = sum(ratings.values()) / len(ratings)

        # Compute song averages
        song_ratings = {}
        for user_id, ratings in self.user_ratings.items():
            for song_id, rating in ratings.items():
                if song_id not in song_ratings:
                    song_ratings[song_id] = []
                song_ratings[song_id].append(rating)

        for song_id, ratings in song_ratings.items():
            self.song_averages[song_id] = sum(ratings) / len(ratings)

        # Global average
        all_ratings = []
        for ratings in self.user_ratings.values():
            all_ratings.extend(ratings.values())
        self.global_average = sum(all_ratings) / len(all_ratings)

        self.is_trained = True
        return True

    def predict(self, user_id, song_id):
        """Simple prediction: blend user avg, song avg, global avg"""
        if not self.is_trained:
            return None

        user_avg = self.user_averages.get(user_id, self.global_average)
        song_avg = self.song_averages.get(song_id, self.global_average)

        # Simple weighted average
        prediction = (user_avg + song_avg + self.global_average) / 3.0
        return max(1.0, min(5.0, prediction))  # Clamp to 1-5 range

    def get_parameters(self):
        """Get model parameters as simple Python dict"""
        return {
            'user_averages': dict(self.user_averages),
            'song_averages': dict(self.song_averages),
            'global_average': float(self.global_average)
        }

    def update_parameters(self, other_params):
        """Average our parameters with another node's parameters"""
        if not other_params:
            return False

        try:
            # Average user averages
            for user_id, avg in other_params.get('user_averages', {}).items():
                user_id = int(user_id)
                if user_id in self.user_averages:
                    self.user_averages[user_id] = (self.user_averages[user_id] + avg) / 2.0
                else:
                    self.user_averages[user_id] = avg

            # Average song averages
            for song_id, avg in other_params.get('song_averages', {}).items():
                song_id = int(song_id)
                if song_id in self.song_averages:
                    self.song_averages[song_id] = (self.song_averages[song_id] + avg) / 2.0
                else:
                    self.song_averages[song_id] = avg

            # Average global average
            other_global = other_params.get('global_average', self.global_average)
            self.global_average = (self.global_average + other_global) / 2.0

            return True
        except Exception as e:
            print(f"Error updating parameters: {e}")
            return False


class SimpleNetwork:
    """Simple TCP networking without complex serialization"""

    def __init__(self, host, port, peers):
        self.host = host
        self.port = port
        self.peers = peers
        self.running = False
        self.server_socket = None
        self.message_handler = None

    def set_handler(self, handler):
        self.message_handler = handler

    def start_server(self):
        """Start server in background thread"""

        def server_loop():
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind((self.host, self.port))
                self.server_socket.listen(5)
                self.running = True
                print(f"Server listening on {self.host}:{self.port}")

                while self.running:
                    try:
                        client, addr = self.server_socket.accept()
                        threading.Thread(target=self._handle_client, args=(client, addr), daemon=True).start()
                    except:
                        if self.running:
                            print("Server accept error")
                        break
            except Exception as e:
                print(f"Server error: {e}")

        threading.Thread(target=server_loop, daemon=True).start()
        time.sleep(0.5)  # Give server time to start

    def _handle_client(self, client, addr):
        """Handle incoming connection"""
        try:
            data = client.recv(4096)
            if data:
                # Use pickle for reliable serialization
                message = pickle.loads(data)

                if self.message_handler:
                    response = self.message_handler(message)
                    if response:
                        client.send(pickle.dumps(response))
        except Exception as e:
            print(f"Client handler error: {e}")
        finally:
            client.close()

    def send_message(self, peer_host, peer_port, message):
        """Send message to peer"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((peer_host, peer_port))

            # Use pickle instead of JSON
            sock.send(pickle.dumps(message))

            response_data = sock.recv(4096)
            if response_data:
                return pickle.loads(response_data)

        except Exception as e:
            print(f"Send error to {peer_host}:{peer_port} - {e}")
            return None
        finally:
            try:
                sock.close()
            except:
                pass
        return None

    def get_random_peer(self):
        """Get random peer (not self)"""
        available = [(h, p) for h, p in self.peers if not (h == self.host and p == self.port)]
        return random.choice(available) if available else None

    def stop(self):
        """Stop server"""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass


class SimpleGossipNode:
    """Simple gossip learning node"""

    def __init__(self, config):
        self.config = config
        self.model = SimpleModel()
        self.network = SimpleNetwork(config.host, config.port, config.peers)
        self.network.set_handler(self._handle_message)
        self.training = False
        self.round_count = 0

    def _handle_message(self, message):
        """Handle incoming gossip message"""
        if message.get('type') == 'gossip_request':
            # Send our parameters
            return {
                'type': 'gossip_response',
                'parameters': self.model.get_parameters()
            }
        return None

    def start(self):
        """Start the node"""
        print(f"🚀 Starting node {self.config.host}:{self.config.port}")

        # Start network
        self.network.start_server()

        # Load data
        if not self.model.load_data(self.config.data_file):
            print(f"❌ Failed to load {self.config.data_file}")
            return False

        print(f"✅ Loaded data and trained local model")
        return True

    def run_gossip(self):
        """Run gossip training"""
        if not self.model.is_trained:
            print("❌ Model not trained")
            return

        print(f"🔄 Starting gossip for {self.config.gossip_rounds} rounds...")
        self.training = True
        self.round_count = 0

        while self.training and self.round_count < self.config.gossip_rounds:
            peer = self.network.get_random_peer()
            if not peer:
                print("⚠️ No peers available")
                time.sleep(self.config.gossip_interval)
                continue

            print(f"Round {self.round_count + 1}: Contacting {peer[0]}:{peer[1]}")

            # Send gossip request
            message = {'type': 'gossip_request'}
            response = self.network.send_message(peer[0], peer[1], message)

            if response and response.get('parameters'):
                # Update our model
                if self.model.update_parameters(response['parameters']):
                    print(f"✅ Updated parameters from {peer[0]}:{peer[1]}")
                else:
                    print(f"❌ Failed to update parameters")
            else:
                print(f"⚠️ No response from {peer[0]}:{peer[1]}")

            self.round_count += 1

            if self.round_count < self.config.gossip_rounds:
                time.sleep(self.config.gossip_interval)

        self.training = False
        print(f"🎉 Gossip training completed!")

    def test_predictions(self):
        """Test some predictions"""
        print("\n🔮 Testing predictions:")
        test_cases = [(1, 101), (2, 102), (1, 103)]

        for user_id, song_id in test_cases:
            pred = self.model.predict(user_id, song_id)
            print(f"User {user_id}, Song {song_id}: {pred:.2f} ⭐")

    def stop(self):
        """Stop the node"""
        self.training = False
        self.network.stop()
        print("👋 Node stopped")


def main():
    """Main entry point"""
    print("🎵 Simple Gossip Music Recommendation System")
    print("=" * 50)

    # Load config
    config = Config.load()
    print(f"Node: {config.host}:{config.port}")
    print(f"Peers: {config.peers}")

    # Create and start node
    node = SimpleGossipNode(config)

    if not node.start():
        sys.exit(1)

    try:
        # Run gossip training
        node.run_gossip()

        # Test predictions
        node.test_predictions()

        # Keep running
        print("\nNode running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n⏹️ Stopping...")
        node.stop()


if __name__ == "__main__":
    main()