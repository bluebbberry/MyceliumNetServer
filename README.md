# 🌐 Mycelium Net - Meta-Federated Learning Network

A prototype implementation of a "network of ML networks" - an internet-like protocol for federated learning where nodes can discover, join, and migrate between different learning groups based on performance metrics.

![image of mycelium net](docs/image.png)

## 🧠 Concept

Traditional federated learning connects nodes in fixed groups. Mycelium Net creates a **meta-layer** where:
- **Learning Groups** train models collaboratively (using Flower AI)
- **Global Registry** acts as DNS for ML groups, tracking performance metrics
- **Smart Migration** allows nodes to switch to better-performing groups
- **Open Discovery** enables nodes to find optimal learning partnerships

Think of it as **BGP routing for machine learning** - nodes can discover and join the most effective learning communities.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Learning      │    │   Learning      │    │   Learning      │
│   Group A       │    │   Group B       │    │   Group C       │
│  ┌───┐ ┌───┐    │    │  ┌───┐ ┌───┐    │    │  ┌───┐ ┌───┐    │
│  │N1 │ │N2 │    │    │  │N3 │ │N4 │    │    │  │N5 │ │N6 │    │
│  └───┘ └───┘    │    │  └───┘ └───┘    │    │  └───┘ └───┘    │
└─────┬───────────┘    └─────┬───────────┘    └─────┬───────────┘
      │                      │                      │
      └──────────────────────┼──────────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Global Registry │
                    │ - Group Discovery│
                    │ - Performance   │
                    │ - Node Migration│
                    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install fastapi uvicorn pydantic requests flwr torch numpy
```

### 1. Start the Global Registry
```bash
python mycelium_net.py registry
```
The registry will start on `http://localhost:8000`

### 2. Start Individual Nodes
```bash
# Terminal 2
python mycelium_net.py node

# Terminal 3  
python mycelium_net.py node

# Terminal 4
python mycelium_net.py node
```

### 3. Run Full Demo
```bash
python mycelium_net.py demo
```

## 📊 Monitor the Network

- **Registry API**: `http://localhost:8000/docs`
- **List Groups**: `GET http://localhost:8000/groups`
- **Node Status**: `GET http://localhost:8000/nodes`

## 🔧 Key Components

### 1. Global Registry (`registry.py`)
- **FastAPI** server with SQLite database
- Tracks learning groups and their performance metrics
- Provides discovery API for nodes
- Handles group joining/leaving

### 2. Mycelium Node (`mycelium_node.py`)
- **Flower AI integration** for federated learning
- **Smart group discovery** and migration logic
- **Performance tracking** and reporting
- **Heartbeat system** for registry updates

### 3. Demo System
- Spawns multiple nodes automatically
- Shows group formation and migration
- Demonstrates performance-based switching

## 🧪 What Happens in the Demo

1. **Bootstrapping**: First node creates a learning group
2. **Discovery**: Subsequent nodes discover existing groups
3. **Joining**: Nodes join groups with available capacity
4. **Training**: Federated learning rounds begin within groups
5. **Migration**: Nodes evaluate and switch to better-performing groups
6. **Evolution**: The network self-optimizes over time

## 📈 Research Direction & Benefits

### Novel Aspects
- **Multi-tier Federation**: Groups of federated groups
- **Dynamic Topology**: Nodes migrate based on performance
- **Open Discovery**: Global registry enables network effects
- **Emergent Optimization**: Better algorithms naturally attract nodes

### Potential Benefits
- **Faster Convergence**: Nodes gravitate toward effective learning clusters
- **Fault Tolerance**: Failed groups can be abandoned and reformed
- **Specialization**: Groups can develop expertise in specific data distributions
- **Resource Efficiency**: Underperforming groups naturally dissolve

### Related Research Areas
- **Multi-Agent Federated Learning**
- **Hierarchical Federated Learning** 
- **Decentralized Learning Networks**
- **Swarm Intelligence for ML**

## ⚙️ Configuration

### Node Configuration
```python
config = NodeConfig(
    registry_url="http://localhost:8000",
    node_address="localhost:8080", 
    heartbeat_interval=30,  # seconds
    group_evaluation_interval=60  # seconds
)
```

### Group Settings
- `max_capacity`: Maximum nodes per group (default: 10)
- `join_policy`: "open" or "invite-only"
- `performance_threshold`: Minimum improvement to trigger migration

## 🔮 Future Extensions

- **Blockchain Registry**: Decentralized group discovery using IPFS/libp2p
- **Incentive Mechanisms**: Token rewards for high-performing groups
- **Privacy Preservation**: Zero-knowledge proofs for performance metrics
- **Specialized Protocols**: Different algorithms for different domains
- **Cross-Chain Learning**: Groups spanning different blockchain networks

## 🐛 Limitations & Notes

- **MVP Implementation**: Uses synthetic data and simplified FL rounds
- **Centralized Registry**: Single point of failure (future: decentralize)
- **No Authentication**: Production would need secure node identity
- **Simple Migration**: Could be more sophisticated (e.g., gradual migration)

## 🤝 Contributing

This is a research prototype exploring novel federated learning architectures. Contributions welcome for:

- Decentralized registry implementations
- Advanced migration strategies  
- Real dataset integration
- Performance benchmarking
- Security and privacy enhancements

## 📚 References

- [Flower: A Friendly Federated Learning Framework](https://flower.dev/)
- [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
- [Hierarchical Federated Learning](https://arxiv.org/abs/2004.10450)

---

**🌟 Star this project if you find the concept interesting!**

The future of AI might be networks of AI networks - let's build it together.