import subprocess
import time
import sys
import threading
from mycelium_node import MyceliumNode, NodeConfig


def run_demo():
    """Run a demonstration of the Mycelium Network"""
    print("🌐 Starting Mycelium Net Demo")
    print("=" * 50)

    # Start registry server
    print("1. Starting registry server...")
    registry_process = subprocess.Popen([
        sys.executable, "registry.py"
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
    print("   - Check http://localhost:8000/docs for API documentation")
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
    run_demo()