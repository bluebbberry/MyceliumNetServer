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
                # First, test if the registry is accessible
                response = requests.get(f"{self.registry_url}/network/state", timeout=5)

                if response.status_code == 200:
                    state = response.json()
                    timestamp = time.time()

                    # Validate the state structure
                    if "groups" in state and "nodes" in state:
                        self.history.append({
                            "timestamp": timestamp,
                            "groups": state["groups"],
                            "nodes": state["nodes"]
                        })
                        print(
                            f"✓ Collected data point {len(self.history)}: {len(state['groups'])} groups, {len(state['nodes'])} nodes")

                        # Keep only last 50 snapshots
                        if len(self.history) > 50:
                            self.history.pop(0)
                    else:
                        print(f"⚠ Invalid state structure: {state}")

                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    print(f"✗ Data collection failed: {error_msg}")
                    self.collection_errors.append(error_msg)

            except requests.exceptions.ConnectionError as e:
                error_msg = f"Connection error - Registry not accessible at {self.registry_url}"
                print(f"✗ {error_msg}")
                self.collection_errors.append(error_msg)

            except requests.exceptions.Timeout as e:
                error_msg = f"Timeout error - Registry not responding"
                print(f"✗ {error_msg}")
                self.collection_errors.append(error_msg)

            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                print(f"✗ Data collection error: {error_msg}")
                self.collection_errors.append(error_msg)

            time.sleep(2)

    def test_registry_connection(self):
        """Test if registry is accessible and has the required endpoint"""
        print(f"Testing connection to registry at {self.registry_url}...")

        try:
            # Test basic connectivity
            response = requests.get(f"{self.registry_url}/health", timeout=5)
            print(f"Health check: {response.status_code}")
        except:
            print("Health endpoint not available")

        try:
            # Test the network state endpoint
            response = requests.get(f"{self.registry_url}/network/state", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(
                    f"✓ Network state endpoint working: {len(data.get('groups', []))} groups, {len(data.get('nodes', []))} nodes")
                return True
            else:
                print(f"✗ Network state endpoint returned {response.status_code}: {response.text}")
                return False
        except Exception as e:
            print(f"✗ Cannot connect to network state endpoint: {e}")
            return False

    def create_network_graph(self, state):
        """Create networkx graph from state - FIXED VERSION"""
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

            # Only add edge if node has a group
            if node.get("group"):
                G.add_edge(f"N_{node['id']}", f"G_{node['group']}")

        return G

    def create_dummy_data(self):
        """Create some dummy data for testing visualization - FIXED VERSION"""
        print("Creating dummy data for testing...")

        # Clear any existing history
        self.history = []

        for i in range(10):
            timestamp = time.time() + i * 2

            # Create dummy state with exactly 4 nodes and 2 groups
            state = {
                "groups": [
                    {
                        "id": "group_0",
                        "performance": 0.5 + 0.1 * np.sin(i * 0.5),
                        "members": ["Node-1", "Node-2"],
                        "model_type": "NeuralNet",
                        "dataset_name": "synthetic"
                    },
                    {
                        "id": "group_1",
                        "performance": 0.6 + 0.1 * np.cos(i * 0.5),
                        "members": ["Node-3", "Node-4"],
                        "model_type": "NeuralNet",
                        "dataset_name": "synthetic"
                    }
                ],
                "nodes": [
                    {"id": "Node-1", "performance": 0.7 + 0.05 * np.sin(i * 0.3), "group": "group_0"},
                    {"id": "Node-2", "performance": 0.8 + 0.05 * np.cos(i * 0.3), "group": "group_0"},
                    {"id": "Node-3", "performance": 0.6 + 0.05 * np.sin(i * 0.4), "group": "group_1"},
                    {"id": "Node-4", "performance": 0.9 + 0.05 * np.cos(i * 0.4), "group": "group_1"}
                ]
            }

            self.history.append({
                "timestamp": timestamp,
                "groups": state["groups"],
                "nodes": state["nodes"]
            })

    def animate_network(self, save_gif=True):
        """Create animated visualization - FIXED VERSION"""
        if not self.history:
            print("No data collected yet")
            print("Collection errors:", self.collection_errors)

            # Offer to create dummy visualization
            create_dummy = input("Create dummy visualization for testing? (y/n): ").lower().strip()
            if create_dummy == 'y':
                self.create_dummy_data()
            else:
                return

        print(f"Creating animation with {len(self.history)} data points...")

        # Debug: Print first data point to check structure
        if self.history:
            print(f"First data point: {len(self.history[0]['groups'])} groups, {len(self.history[0]['nodes'])} nodes")
            for i, group in enumerate(self.history[0]['groups']):
                print(f"  Group {i}: {group['id']} with {len(group.get('members', []))} members")
            for i, node in enumerate(self.history[0]['nodes']):
                print(f"  Node {i}: {node['id']} in group {node.get('group', 'None')}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        def update(frame):
            if frame >= len(self.history):
                return

            ax1.clear()
            ax2.clear()

            state = self.history[frame]
            G = self.create_network_graph(state)

            # Debug: Print graph info
            if frame == 0:
                print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                print(f"Nodes: {list(G.nodes())}")

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
                    # Create small offset for nodes around group
                    node_index = node_nodes.index(node)
                    offset_angle = 2 * np.pi * node_index / len(node_nodes)
                    offset_radius = 0.5
                    offset = (offset_radius * np.cos(offset_angle), offset_radius * np.sin(offset_angle))
                    pos[node] = (group_pos[0] + offset[0], group_pos[1] + offset[1])
                else:
                    # Position unconnected nodes randomly
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
                print("Showing animation instead...")

        plt.tight_layout()
        plt.show()

    def start_collection(self):
        """Start data collection in background"""
        # Test connection first
        if not self.test_registry_connection():
            print("⚠ Registry connection failed - data collection may not work")

        self.running = True
        self.collection_thread = threading.Thread(target=self.collect_data, daemon=True)
        self.collection_thread.start()

    def stop_collection(self):
        """Stop data collection"""
        self.running = False
        if hasattr(self, 'collection_thread'):
            self.collection_thread.join(timeout=5)


def run_visual_demo():
    """Run demo with Flower integration and visualization"""
    import subprocess
    import time
    import os

    print("🎬 Starting Mycelium Net Visual Demo with Flower AI")
    print("=" * 50)

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
        time.sleep(5)  # Give registry more time to start

        # Test registry connection
        print("Testing registry connection...")
        import requests
        try:
            response = requests.get("http://localhost:8000/health", timeout=10)
            print(f"✓ Registry health check: {response.status_code}")
        except Exception as e:
            print(f"✗ Registry connection failed: {e}")

        # Start visualizer
        print("Starting visualizer...")
        visualizer = NetworkVisualizer()
        visualizer.start_collection()
        time.sleep(2)

        # Start nodes
        print("Starting nodes...")
        try:
            from mycelium_node import MyceliumNode, NodeConfig
            nodes = []
            for i in range(4):
                config = NodeConfig(
                    heartbeat_interval=3,  # Faster heartbeats for better visualization
                    performance_boost_rate=0.08,
                    flower_server_address=f"localhost:808{(i % 2) + 1}"
                )
                node = MyceliumNode(config, f"Node-{i + 1}")
                nodes.append(node)
                node.start()
                time.sleep(1)

            print("Collecting data for 60 seconds with training...")
            time.sleep(60)

            print("Stopping nodes and creating visualization...")
            for node in nodes:
                node.stop()

        except ImportError as e:
            print(f"Could not import mycelium_node: {e}")
            print("Running without nodes for 30 seconds...")
            time.sleep(30)

        visualizer.stop_collection()
        print(f"Collected {len(visualizer.history)} data points")

        if len(visualizer.history) > 0:
            visualizer.animate_network(save_gif=True)
        else:
            print("No data collected - trying dummy visualization...")
            visualizer.create_dummy_data()
            visualizer.animate_network(save_gif=True)

    except KeyboardInterrupt:
        print("\nStopping demo...")
    finally:
        # Cleanup
        print("Cleaning up processes...")
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()


if __name__ == "__main__":
    run_visual_demo()