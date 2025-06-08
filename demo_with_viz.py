import subprocess
import time
import sys
import threading
from mycelium_node import MyceliumNode, NodeConfig
from visualization import MyceliumVisualizer


def run_demo_with_visualization():
    """Run a demonstration of the Mycelium Network with real-time visualization"""
    print("🌐 Starting Mycelium Net Demo with Visualization")
    print("=" * 60)

    # Start registry server
    print("1. Starting registry server...")
    registry_process = subprocess.Popen([
        sys.executable, "registry.py"
    ])

    time.sleep(3)  # Wait for server to start

    # Start multiple nodes
    print("2. Starting Mycelium nodes...")

    nodes = []
    for i in range(4):  # Start 4 nodes for better visualization
        node = MyceliumNode(NodeConfig(
            node_address=f"localhost:808{i}",
            heartbeat_interval=8,  # Faster updates for better visualization
            group_evaluation_interval=12
        ))
        nodes.append(node)

    # Start nodes in separate threads
    node_threads = []
    for i, node in enumerate(nodes):
        thread = threading.Thread(target=node.start, daemon=True)
        thread.start()
        node_threads.append(thread)
        time.sleep(2)  # Stagger node starts

    print("3. Starting visualization...")
    print("   - Real-time network topology")
    print("   - Performance tracking over time")
    print("   - Network statistics dashboard")
    print("   - Automatic GIF export")

    # Initialize visualizer
    visualizer = MyceliumVisualizer(update_interval=3)  # Update every 3 seconds

    print("\n4. Demo is running...")
    print("   - Registry API: http://localhost:8000/docs")
    print("   - Group status: http://localhost:8000/groups")
    print("   - Visualization window will open shortly...")
    print("   - Press Ctrl+C to stop and save GIF")

    try:
        # Start visualization (non-blocking)
        viz_thread = visualizer.start_visualization(duration=300)  # 5 minutes max

        # Keep demo running
        start_time = time.time()
        while time.time() - start_time < 300:  # Run for 5 minutes max
            time.sleep(5)
            active_nodes = len([n for n in nodes if n.running])
            elapsed = int(time.time() - start_time)
            print(f"⏱️  Demo running... {active_nodes} nodes active, {elapsed}s elapsed")

    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")

    finally:
        print("\n5. Shutting down...")

        # Stop nodes
        for node in nodes:
            node.stop()

        # Stop visualization and save GIF
        print("📹 Saving visualization as GIF...")
        visualizer.save_gif("mycelium_net_demo.gif", fps=0.5)  # Slow GIF for better viewing
        visualizer.stop()

        # Stop registry
        registry_process.terminate()

        print("✅ Demo completed successfully!")
        print("   - Check 'output/mycelium_net_demo.gif' for the visualization")


def run_visualization_only():
    """Run only the visualization (assumes registry is already running)"""
    print("📊 Starting Mycelium Net Visualization Only")
    print("=" * 50)
    print("   Assumes registry server is running on http://localhost:8000")

    try:
        visualizer = MyceliumVisualizer(update_interval=2)
        viz_thread = visualizer.start_visualization(duration=180)  # 3 minutes

        print("📈 Visualization active for 3 minutes...")
        print("   - Close plot window or Ctrl+C to stop early")

        time.sleep(180)

    except KeyboardInterrupt:
        print("\n⏹️  Visualization stopped")

    finally:
        visualizer.save_gif("mycelium_viz_only.gif", fps=0.5)
        visualizer.stop()
        print("✅ GIF saved as 'output/mycelium_viz_only.gif'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mycelium Net Demo with Visualization")
    parser.add_argument("--viz-only", action="store_true",
                        help="Run visualization only (registry must be running)")
    parser.add_argument("--duration", type=int, default=300,
                        help="Demo duration in seconds (default: 300)")

    args = parser.parse_args()

    if args.viz_only:
        run_visualization_only()
    else:
        run_demo_with_visualization()