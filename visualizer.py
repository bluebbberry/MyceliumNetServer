import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import requests
import json
import time
import threading
from collections import defaultdict
import numpy as np
import sys


class NetworkVisualizer:
    def __init__(self, registry_url="http://localhost:8000"):
        self.registry_url = registry_url
        self.history = []
        self.running = False
        self.collection_errors = []

    def collect_data(self):
        """Collect network state data with better error handling"""
        print(f"Starting data collection from {self.registry_url}")

        while self.running:
            try:
                response = requests.get(f"{self.registry_url}/network/state", timeout=5)

                if response.status_code == 200:
                    state = response.json()
                    timestamp = time.time()

                    if "groups" in state and "nodes" in state:
                        self.history.append({
                            "timestamp": timestamp,
                            "groups": state["groups"],
                            "nodes": state["nodes"]
                        })
                        print(
                            f"✓ Collected data point {len(self.history)}: {len(state['groups'])} groups, {len(state['nodes'])} nodes")

                        if len(self.history) > 50:
                            self.history.pop(0)
                    else:
                        print(f"⚠ Invalid state structure: {state}")

                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    print(f"✗ Data collection failed: {error_msg}")
                    self.collection_errors.append(error_msg)

            except Exception as e:
                error_msg = f"Data collection error: {str(e)}"
                print(f"✗ {error_msg}")
                self.collection_errors.append(error_msg)

            time.sleep(2)

    def test_registry_connection(self):
        """Test if registry is accessible"""
        print(f"Testing connection to registry at {self.registry_url}...")

        try:
            response = requests.get(f"{self.registry_url}/network/state", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(
                    f"✓ Network state endpoint working: {len(data.get('groups', []))} groups, {len(data.get('nodes', []))} nodes")
                return True
            else:
                print(f"✗ Network state endpoint returned {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Cannot connect to network state endpoint: {e}")
            return False

    def create_network_graph(self, state):
        """Create networkx graph from state"""
        G = nx.Graph()

        # Add group nodes
        for group in state["groups"]:
            G.add_node(f"G_{group['id']}",
                       type="group",
                       performance=group.get("performance", 0.0),
                       members=group.get("members", []))

        # Add node nodes and edges
        for node in state["nodes"]:
            G.add_node(f"N_{node['id']}",
                       type="node",
                       performance=node.get("performance", 0.0))

            if node.get("group"):
                G.add_edge(f"N_{node['id']}", f"G_{node['group']}")

        return G

    def animate_network(self, save_gif=True):
        """Create animated visualization"""
        if not self.history:
            print("No data collected yet")
            return

        print(f"Creating animation with {len(self.history)} data points...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        def update(frame):
            if frame >= len(self.history):
                return

            ax1.clear()
            ax2.clear()

            state = self.history[frame]
            G = self.create_network_graph(state)

            # Network visualization
            pos = {}
            group_nodes = [n for n in G.nodes() if n.startswith('G_')]
            node_nodes = [n for n in G.nodes() if n.startswith('N_')]

            # Position groups in circle
            if group_nodes:
                for i, group in enumerate(group_nodes):
                    angle = 2 * np.pi * i / len(group_nodes)
                    pos[group] = (2 * np.cos(angle), 2 * np.sin(angle))

            # Position nodes around their groups
            for node in node_nodes:
                group_neighbors = [n for n in G.neighbors(node) if n.startswith('G_')]
                if group_neighbors:
                    group_pos = pos[group_neighbors[0]]
                    node_index = node_nodes.index(node)
                    offset_angle = 2 * np.pi * node_index / len(node_nodes)
                    offset_radius = 0.5
                    offset = (offset_radius * np.cos(offset_angle), offset_radius * np.sin(offset_angle))
                    pos[node] = (group_pos[0] + offset[0], group_pos[1] + offset[1])
                else:
                    pos[node] = (np.random.uniform(-3, 3), np.random.uniform(-3, 3))

            # Draw network
            if group_nodes:
                nx.draw_networkx_nodes(G, pos,
                                       nodelist=group_nodes,
                                       node_color='red',
                                       node_size=800,
                                       alpha=0.7,
                                       ax=ax1)

            if node_nodes:
                nx.draw_networkx_nodes(G, pos,
                                       nodelist=node_nodes,
                                       node_color='lightblue',
                                       node_size=300,
                                       ax=ax1)

            nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.5)

            # Labels
            labels = {}
            for node in G.nodes():
                if node.startswith('G_'):
                    perf = G.nodes[node]['performance']
                    labels[node] = f"G\n{perf:.2f}"
                else:
                    labels[node] = node.split('_')[1]

            nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax1)

            ax1.set_title(f"Mycelium Network - Frame {frame + 1}/{len(self.history)}")
            ax1.set_xlim(-4, 4)
            ax1.set_ylim(-4, 4)
            ax1.axis('off')

            # Performance over time
            if len(self.history) > 1:
                timestamps = [s["timestamp"] - self.history[0]["timestamp"] for s in self.history[:frame + 1]]

                # Plot group performance
                for group in state["groups"]:
                    group_id = group["id"]
                    perfs = []
                    times = []
                    for i, s in enumerate(self.history[:frame + 1]):
                        for g in s["groups"]:
                            if g["id"] == group_id:
                                perfs.append(g["performance"])
                                times.append(timestamps[i])
                                break

                    if perfs:
                        ax2.plot(times, perfs, 'o-', label=f"Group {group_id}", linewidth=2)

                # Plot node performance
                for node in state["nodes"]:
                    node_id = node["id"]
                    perfs = []
                    times = []
                    for i, s in enumerate(self.history[:frame + 1]):
                        for n in s["nodes"]:
                            if n["id"] == node_id:
                                perfs.append(n["performance"])
                                times.append(timestamps[i])
                                break

                    if perfs:
                        ax2.plot(times, perfs, '--', alpha=0.7, label=f"Node {node_id}", linewidth=1)

            ax2.set_xlabel("Time (seconds)")
            ax2.set_ylabel("Performance")
            ax2.set_title("Performance Over Time")
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)

        anim = animation.FuncAnimation(fig, update, frames=len(self.history),
                                       interval=500, repeat=True)

        if save_gif:
            print("Saving animation as mycelium_network.gif...")
            try:
                anim.save('mycelium_network.gif', writer='pillow', fps=2)
                print("✓ Animation saved as mycelium_network.gif!")
            except Exception as e:
                print(f"✗ Error saving GIF: {e}")

        plt.tight_layout()
        plt.show()

    def start_collection(self):
        """Start data collection in background"""
        if not self.test_registry_connection():
            print("⚠ Registry connection failed")

        self.running = True
        self.collection_thread = threading.Thread(target=self.collect_data, daemon=True)
        self.collection_thread.start()

    def stop_collection(self):
        """Stop data collection"""
        self.running = False


def start_flower_server():
    """Start Flower server"""
    import flwr as fl

    def weighted_average(metrics):
        """Aggregate function for evaluation metrics"""
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"accuracy": sum(accuracies) / sum(examples)}

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )


def run_visual_demo():
    """Run demo with real Flower federated learning"""
    import subprocess
    import time
    import os

    print("🎬 Starting Mycelium Net Visual Demo with Real Flower FL")
    print("=" * 55)

    processes = []

    try:
        # Start registry
        print("Starting registry...")
        registry_process = subprocess.Popen([
            sys.executable, "-c", """
import sys
sys.path.append('.')
from registry import app
import uvicorn
uvicorn.run(app, host='0.0.0.0', port=8000)
"""
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(registry_process)
        time.sleep(5)

        # Test registry
        print("Testing registry connection...")
        import requests
        try:
            response = requests.get("http://localhost:8000/health", timeout=10)
            print(f"✓ Registry health check: {response.status_code}")
        except Exception as e:
            print(f"✗ Registry connection failed: {e}")

        # Start Flower server
        print("Starting Flower server...")
        flower_server_process = subprocess.Popen([
            sys.executable, "-c", """
import flwr as fl

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2,
    min_evaluate_clients=2, 
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
)

fl.server.start_server(
    server_address="[::]:8080",
    config=fl.server.ServerConfig(num_rounds=15),
    strategy=strategy,
)
"""
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(flower_server_process)
        time.sleep(3)

        # Start visualizer
        print("Starting visualizer...")
        visualizer = NetworkVisualizer()
        visualizer.start_collection()
        time.sleep(2)

        # Start nodes with real Flower integration
        print("Starting nodes with Flower clients...")
        try:
            from mycelium_node import MyceliumNode, NodeConfig
            nodes = []
            for i in range(4):
                config = NodeConfig(
                    heartbeat_interval=3,
                    performance_boost_rate=0.08,
                )
                node = MyceliumNode(config, f"Node-{i + 1}")
                nodes.append(node)
                node.start()
                time.sleep(2)  # Stagger node starts

            print("Running federated learning for 90 seconds...")
            time.sleep(90)

            print("Stopping nodes...")
            for node in nodes:
                node.stop()

        except ImportError as e:
            print(f"Could not import mycelium_node: {e}")
            print("Running registry-only demo...")
            time.sleep(60)

        visualizer.stop_collection()
        print(f"Collected {len(visualizer.history)} data points")

        if len(visualizer.history) > 0:
            visualizer.animate_network(save_gif=True)
        else:
            print("No data collected")

    except KeyboardInterrupt:
        print("\nStopping demo...")
    finally:
        print("Cleaning up processes...")
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()


if __name__ == "__main__":
    run_visual_demo()