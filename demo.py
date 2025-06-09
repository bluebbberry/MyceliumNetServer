import subprocess
import time
import sys
import threading
import requests
import json
from mycelium_node import MyceliumNode, NodeConfig


def start_flower_servers():
    """Start Flower servers for different groups"""
    servers = []

    # Start servers on different ports for different groups
    for i, port in enumerate([8081, 8082, 8083]):
        server_process = subprocess.Popen([
            sys.executable, "flower_server.py", f"group_{i}", str(port)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        servers.append(server_process)
        time.sleep(1)

    return servers


def run_basic_demo():
    """Run basic demo with Flower integration"""
    print("🌐 Starting Mycelium Net Demo with Flower AI")
    print("=" * 50)

    # Start registry
    print("1. Starting registry server...")
    registry_process = subprocess.Popen([
        sys.executable, "registry.py"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    time.sleep(3)

    # Start Flower servers
    print("2. Starting Flower servers...")
    flower_servers = start_flower_servers()
    time.sleep(3)

    # Create and start nodes
    print("3. Starting Mycelium nodes...")
    nodes = []
    for i in range(4):
        config = NodeConfig(
            heartbeat_interval=8,  # Slower for Flower integration
            performance_boost_rate=0.05,
            flower_server_address=f"localhost:808{(i % 3) + 1}"  # Distribute across servers
        )
        node = MyceliumNode(config, f"Node-{i + 1}")
        nodes.append(node)
        node.start()
        time.sleep(2)

    print("4. Nodes training with Flower federated learning...")
    print("   Press Ctrl+C to stop")

    try:
        for round_num in range(15):
            time.sleep(8)
            # Show network state
            try:
                response = requests.get("http://localhost:8000/network/state")
                if response.status_code == 200:
                    state = response.json()
                    print(f"\nRound {round_num + 1}:")
                    for group in state["groups"]:
                        members = [n["id"] for n in state["nodes"] if n["group"] == group["id"]]
                        print(f"  Group {group['id']}: {len(members)} members, perf: {group['performance']:.3f}")
            except:
                pass
    except KeyboardInterrupt:
        print("\nStopping demo...")
        for node in nodes:
            node.stop()
        registry_process.terminate()
        for server in flower_servers:
            server.terminate()


if __name__ == "__main__":
    run_basic_demo()