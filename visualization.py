import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
import requests
import time
import threading
from datetime import datetime
from collections import defaultdict, deque
import os


class MyceliumVisualizer:
    """Real-time visualization of Mycelium Network dynamics"""

    def __init__(self, registry_url="http://localhost:8000", update_interval=2):
        self.registry_url = registry_url
        self.update_interval = update_interval
        self.running = False

        # Data storage
        self.timestamps = deque(maxlen=100)
        self.group_performance = defaultdict(lambda: deque(maxlen=100))
        self.group_membership = defaultdict(lambda: deque(maxlen=100))
        self.node_positions = {}
        self.group_colors = {}
        self.color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

        # Setup plots
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('🌐 Mycelium Net - Meta-Federated Learning Visualization', fontsize=16, fontweight='bold')

        # Create subplots
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        self.ax_network = self.fig.add_subplot(gs[0, :2])  # Network topology
        self.ax_performance = self.fig.add_subplot(gs[1, :2])  # Performance over time
        self.ax_stats = self.fig.add_subplot(gs[:, 2])  # Statistics

        # Initialize plots
        self._setup_plots()

        # Animation setup
        self.frames = []
        self.frame_count = 0

    def _setup_plots(self):
        """Initialize plot settings"""
        # Network topology plot
        self.ax_network.set_xlim(-10, 10)
        self.ax_network.set_ylim(-10, 10)
        self.ax_network.set_aspect('equal')
        self.ax_network.set_title('Network Topology & Groups', fontweight='bold')
        self.ax_network.grid(True, alpha=0.3)
        self.ax_network.set_facecolor('#f8f9fa')

        # Performance plot
        self.ax_performance.set_title('Group Performance Over Time', fontweight='bold')
        self.ax_performance.set_xlabel('Time')
        self.ax_performance.set_ylabel('Accuracy')
        self.ax_performance.grid(True, alpha=0.3)
        self.ax_performance.set_facecolor('#f8f9fa')

        # Statistics plot
        self.ax_stats.set_title('Network Statistics', fontweight='bold')
        self.ax_stats.axis('off')
        self.ax_stats.set_facecolor('#f8f9fa')

    def fetch_data(self):
        """Fetch current data from registry"""
        try:
            # Get groups data
            groups_response = requests.get(f"{self.registry_url}/groups", timeout=5)
            if groups_response.status_code == 200:
                groups = groups_response.json()

                current_time = datetime.now()
                self.timestamps.append(current_time)

                # Update group data
                for group in groups:
                    group_id = group['group_id'][:8]  # Short ID for display
                    performance = group['performance_metric']
                    member_count = group['member_count']

                    self.group_performance[group_id].append(performance)
                    self.group_membership[group_id].append(member_count)

                    # Assign color if new group
                    if group_id not in self.group_colors:
                        color_idx = len(self.group_colors) % len(self.color_palette)
                        self.group_colors[group_id] = self.color_palette[color_idx]

                return groups
        except Exception as e:
            print(f"Error fetching data: {e}")
            return []

        return []

    def _generate_node_positions(self, groups):
        """Generate positions for nodes in network topology"""
        self.node_positions.clear()

        # Group centers in a circle
        group_centers = {}
        n_groups = len(groups)
        if n_groups == 0:
            return

        for i, group in enumerate(groups):
            group_id = group['group_id'][:8]
            angle = 2 * np.pi * i / n_groups
            center_x = 6 * np.cos(angle)
            center_y = 6 * np.sin(angle)
            group_centers[group_id] = (center_x, center_y)

        # Position nodes around group centers
        for group in groups:
            group_id = group['group_id'][:8]
            member_count = group['member_count']
            center_x, center_y = group_centers[group_id]

            # Position nodes in a small circle around group center
            for j in range(member_count):
                if member_count == 1:
                    node_x, node_y = center_x, center_y
                else:
                    angle = 2 * np.pi * j / member_count
                    node_x = center_x + 1.5 * np.cos(angle)
                    node_y = center_y + 1.5 * np.sin(angle)

                node_id = f"{group_id}_node_{j}"
                self.node_positions[node_id] = (node_x, node_y, group_id)

    def update_plots(self):
        """Update all plots with current data"""
        groups = self.fetch_data()
        if not groups:
            return

        # Clear previous plots
        self.ax_network.clear()
        self.ax_performance.clear()
        self.ax_stats.clear()

        self._setup_plots()

        # Generate node positions
        self._generate_node_positions(groups)

        # Plot network topology
        self._plot_network_topology(groups)

        # Plot performance over time
        self._plot_performance_timeline()

        # Plot statistics
        self._plot_statistics(groups)

        # Store frame for GIF
        if self.running:
            self.frames.append(self._capture_frame())
            self.frame_count += 1

    def _plot_network_topology(self, groups):
        """Plot the network topology"""
        # Draw group boundaries
        for group in groups:
            group_id = group['group_id'][:8]
            if group_id in self.group_colors:
                # Find group center
                group_nodes = [pos for node_id, pos in self.node_positions.items() if pos[2] == group_id]
                if group_nodes:
                    center_x = np.mean([pos[0] for pos in group_nodes])
                    center_y = np.mean([pos[1] for pos in group_nodes])

                    # Draw group circle
                    circle = Circle((center_x, center_y), 2.5,
                                    color=self.group_colors[group_id],
                                    alpha=0.2, linewidth=2, fill=True)
                    self.ax_network.add_patch(circle)

                    # Group label
                    self.ax_network.text(center_x, center_y - 3, f'Group {group_id}',
                                         ha='center', va='center', fontweight='bold',
                                         bbox=dict(boxstyle="round,pad=0.3",
                                                   facecolor=self.group_colors[group_id],
                                                   alpha=0.7))

        # Draw nodes
        for node_id, (x, y, group_id) in self.node_positions.items():
            color = self.group_colors.get(group_id, '#666666')

            # Node circle
            self.ax_network.scatter(x, y, s=200, c=color, alpha=0.8,
                                    edgecolors='white', linewidth=2, zorder=5)

            # Node connections to group center
            group_nodes = [pos for nid, pos in self.node_positions.items() if pos[2] == group_id]
            if len(group_nodes) > 1:
                center_x = np.mean([pos[0] for pos in group_nodes])
                center_y = np.mean([pos[1] for pos in group_nodes])
                self.ax_network.plot([x, center_x], [y, center_y],
                                     color=color, alpha=0.4, linewidth=1, zorder=1)

        # Legend
        legend_elements = []
        for group_id, color in self.group_colors.items():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                              markerfacecolor=color, markersize=10,
                                              label=f'Group {group_id}'))

        if legend_elements:
            self.ax_network.legend(handles=legend_elements, loc='upper right',
                                   bbox_to_anchor=(1, 1))

    def _plot_performance_timeline(self):
        """Plot performance over time"""
        if not self.timestamps:
            return

        times = list(self.timestamps)

        for group_id, performances in self.group_performance.items():
            if len(performances) > 1:
                y_data = list(performances)[-len(times):]
                color = self.group_colors.get(group_id, '#666666')

                self.ax_performance.plot(times[-len(y_data):], y_data,
                                         color=color, linewidth=2,
                                         marker='o', markersize=4,
                                         label=f'Group {group_id}')

        if self.group_performance:
            self.ax_performance.legend()
            self.ax_performance.set_ylim(0, 1)

        # Format x-axis
        if len(times) > 1:
            self.ax_performance.tick_params(axis='x', rotation=45)

    def _plot_statistics(self, groups):
        """Plot network statistics"""
        if not groups:
            return

        # Calculate statistics
        total_nodes = sum(g['member_count'] for g in groups)
        total_groups = len(groups)
        avg_performance = np.mean([g['performance_metric'] for g in groups]) if groups else 0
        best_performance = max([g['performance_metric'] for g in groups]) if groups else 0

        # Create statistics text
        stats_text = f"""
📊 NETWORK STATISTICS

🔗 Total Groups: {total_groups}
👥 Total Nodes: {total_nodes}
📈 Avg Performance: {avg_performance:.3f}
🏆 Best Performance: {best_performance:.3f}

📋 GROUP DETAILS:
"""

        for i, group in enumerate(sorted(groups, key=lambda x: x['performance_metric'], reverse=True)):
            group_id = group['group_id'][:8]
            stats_text += f"\n🔸 {group_id}: {group['member_count']} nodes, {group['performance_metric']:.3f} acc"

        # Add timestamp
        stats_text += f"\n\n🕒 Updated: {datetime.now().strftime('%H:%M:%S')}"

        self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

    def _capture_frame(self):
        """Capture current frame for GIF"""
        self.fig.canvas.draw()
        frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return frame

    def start_visualization(self, duration=60):
        """Start real-time visualization"""
        print("🎬 Starting Mycelium Net visualization...")
        self.running = True

        def update_loop():
            start_time = time.time()
            while self.running and (time.time() - start_time) < duration:
                self.update_plots()
                plt.pause(self.update_interval)

        # Start update loop in background
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()

        # Show plot
        plt.show(block=False)

        return update_thread

    def save_gif(self, filename="mycelium_net_demo.gif", fps=1):
        """Save collected frames as GIF"""
        if not self.frames:
            print("No frames captured for GIF")
            return

        print(f"🎞️  Saving {len(self.frames)} frames to {filename}...")

        try:
            import imageio

            # Create output directory
            os.makedirs('output', exist_ok=True)
            filepath = os.path.join('output', filename)

            # Save as GIF
            imageio.mimsave(filepath, self.frames, fps=fps, loop=0)
            print(f"✅ GIF saved successfully: {filepath}")

        except ImportError:
            print("❌ imageio not installed. Install with: pip install imageio")
        except Exception as e:
            print(f"❌ Error saving GIF: {e}")

    def stop(self):
        """Stop visualization"""
        self.running = False
        plt.close('all')


# Standalone visualization runner
def run_visualization(duration=60, save_gif=True):
    """Run standalone visualization"""
    visualizer = MyceliumVisualizer()

    try:
        # Start visualization
        update_thread = visualizer.start_visualization(duration)

        print(f"📈 Visualization running for {duration} seconds...")
        print("   - Close the plot window or press Ctrl+C to stop early")

        # Wait for completion
        time.sleep(duration)

        # Save GIF
        if save_gif:
            visualizer.save_gif()

    except KeyboardInterrupt:
        print("\n⏹️  Visualization stopped by user")
        if save_gif and visualizer.frames:
            visualizer.save_gif()

    finally:
        visualizer.stop()


if __name__ == "__main__":
    run_visualization(duration=120, save_gif=True)