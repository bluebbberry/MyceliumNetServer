# Mycelium Net - Meta-Federated Learning Network

A prototype implementation of a "network of ML networks" - an internet-like protocol for federated learning where nodes can discover, join, and migrate between different learning groups based on performance metrics.

![image of mycelium net](docs/image.png)

## Overview

Mycelium Net implements a novel meta-federated learning approach where:
- **Nodes** autonomously discover and join optimal learning groups
- **Registry** maintains a global lookup table of groups and nodes
- **Dynamic switching** enables nodes to migrate to better-performing groups
- **Flower AI integration** provides robust federated learning capabilities

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Node 1        │    │   Node 2        │    │   Node 3        │
│                 │    │                 │    │                 │
│ Local Model     │    │ Local Model     │    │ Local Model     │
│ Performance     │    │ Performance     │    │ Performance     │
│ Group Discovery │    │ Group Discovery │    │ Group Discovery │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │    Registry Server        │
                    │                           │
                    │ • Group Management        │
                    │ • Node Registration       │
                    │ • Performance Tracking    │
                    │ • Discovery API           │
                    └───────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone or download the files:**
   ```bash
   # Ensure you have these files:
   # registry.py, mycelium_node.py, demo.py, requirements.txt
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Demo

**Option 1: Full Demo with Visualization (Recommended)**
```bash
python demo_with_viz.py
```
This starts:
- Registry server on `http://localhost:8000`
- 4 nodes that automatically discover/create groups
- Real-time federated training simulation
- **Live visualization with network topology, performance charts, and statistics**
- **Automatic GIF export** of the entire demo session

**Option 2: Basic Demo (No Visualization)**
```bash
python demo.py
```
Basic demo without visualization.

**Option 2: Manual Setup**

Terminal 1 - Start Registry:
```bash
python registry.py
```

Terminal 2 - Start Node:
```bash
python mycelium_node.py
```

Repeat Terminal 2 for additional nodes.

### Monitoring

- **API Documentation:** http://localhost:8000/docs
- **Group Status:** http://localhost:8000/groups
- **Node Registration:** http://localhost:8000/nodes/register

## Visualization Features

### 🎬 Real-time Animated Dashboard
- **Network Topology**: Visual representation of nodes and groups
- **Performance Tracking**: Live charts showing group performance over time  
- **Statistics Panel**: Real-time network metrics and group details
- **Automatic GIF Export**: Creates animated GIF of the entire demo session

### 📊 What You'll See
1. **Group Formation**: Nodes clustering into performance-based groups
2. **Dynamic Switching**: Nodes migrating to better-performing groups
3. **Performance Evolution**: Real-time accuracy improvements across groups
4. **Network Growth**: New groups forming as nodes join the network

### 🎞️ GIF Output
After running the demo, check `output/mycelium_net_demo.gif` for a complete animated visualization of your session.

### Visualization Controls
```bash
# Full demo with visualization  
python demo_with_viz.py

# Visualization only (registry must be running)
python demo_with_viz.py --viz-only

# Custom duration (default: 5 minutes)
python demo_with_viz.py --duration 180
```

## Key Features
Nodes automatically discover existing learning groups and evaluate performance metrics to make joining decisions.

### 🚀 Adaptive Group Switching
Nodes continuously monitor group performance and migrate to better-performing groups when beneficial.

### 🌐 Decentralized Architecture
No single point of failure - nodes can create new groups independently.

### 🤖 Flower AI Integration
Built on top of Flower's robust federated learning framework for production-ready ML training.

### 📊 Performance Tracking
Real-time monitoring of group and node performance metrics with automatic registry updates.

## Configuration

### Node Configuration
```python
config = NodeConfig(
    registry_url="http://localhost:8000",
    node_address="localhost:8080",
    heartbeat_interval=30,  # seconds
    group_evaluation_interval=60  # seconds
)
```

### Group Parameters
- **Max Capacity:** 5 nodes per group (configurable)
- **Performance Threshold:** 5% improvement required for group switching
- **Join Policy:** Open (can be extended to invite-only)

## API Endpoints

### Registry Server (Port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST   | `/groups` | Create new learning group |
| GET    | `/groups` | List all groups with metrics |
| POST   | `/nodes/register` | Register/update node |
| POST   | `/groups/join` | Join specific group |
| PUT    | `/groups/{id}/performance` | Update group performance |

## Demo Output

```
🌐 Starting Mycelium Net Demo
==================================================
1. Starting registry server...
2. Starting Mycelium nodes...
3. Nodes are running and training...
   - Check http://localhost:8000/groups for group status
   - Check http://localhost:8000/docs for API documentation
   - Press Ctrl+C to stop demo

Demo running... 3 nodes active
```

## Extending the System

### Custom Models
Replace `SimpleNet` in `mycelium_node.py` with your own PyTorch models.

### Real Datasets
Replace the synthetic data generation in `_create_demo_data()` with actual dataset loading.

### Advanced Aggregation
Implement custom aggregation strategies in the Flower client.

### Security
Add authentication, encryption, and secure communication protocols.

## Troubleshooting

**Port conflicts:** Change ports in configuration if 8000/8080 are in use.

**Dependencies:** Ensure PyTorch is properly installed for your system.

**Firewall:** Check that ports are accessible if running across networks.

**Database:** SQLite file is created automatically - delete `mycelium_registry.db` to reset.

## License

Open source - feel free to modify and extend for your use cases.

## Next Steps

- Implement Byzantine fault tolerance
- Add support for heterogeneous models
- Integrate with real-world datasets (CIFAR-10, ImageNet)
- Add web dashboard for monitoring
- Implement advanced group selection algorithms
