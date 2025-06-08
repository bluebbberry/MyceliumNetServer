# MyceliumNetServer

SPORE is a decentralized protocol where nodes register publicly and join collaborative learning groups to train shared machine learning models. Groups publish performance metrics, allowing nodes to evaluate and switch participation over time. This forms a dynamic, self-organizing ecosystem of federated learners — a global, adaptive "internet for machine learning" or in the following called the "mycelium net", because of it's similarity to the interaction and growth of fungi.

A decentralized federated learning system inspired by P2P file sharing networks like Napster and collaborative computing apps like SETI@home.
This app enables distributed machine learning across multiple nodes without requiring a central server, while optimizing bandwidth usage through gradient compression techniques.
Users can train a model for a given machine learning problem with their local data and receive gradient updates from other users of the app to improve their own model.

## 🚀 Features

- **Decentralized Architecture**: No central server required - all nodes communicate peer-to-peer
- **Bandwidth Optimization**: Top-k gradient compression reduces network traffic by up to 90%
- **Asynchronous Training**: Nodes can train and share updates independently
- **Automatic Peer Discovery**: Nodes automatically discover and connect to other peers
- **Gradient Aggregation**: Averaging of model updates across the network

![ChatGPT Image May 31, 2025, 08_41_03 AM](docs/image.png)

## How to Use:

### 1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the demo:

```
python main.py
```
