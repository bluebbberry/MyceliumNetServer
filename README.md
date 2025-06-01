# OpenFL

P2P + federated machine learning + nodes are able to switch learning groups (similar to p2pfl plus switching of nodes)

![ChatGPT Image May 31, 2025, 08_41_03 AM](https://github.com/user-attachments/assets/57aeda8a-d2e9-4d36-a9c1-704a96c06211)

# GOSSIP Music Recommendation System

A distributed music recommendation system using GOSSIP protocol for federated learning.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Files Needed
- `ratings.csv` - Your music rating data
- `songs.csv` - Your songs data
- `config.json` - Node configuration
- All Python files in the same directory

## Quick Start

### Single Node Test
```bash
python main.py
```
That's it! The program will:
1. Load config.json
2. Load ratings.csv
3. Load songs.csv  
4. Train locally
5. Start GOSSIP training
6. Save the model
7. Show recommendations

### Multi-Node Test (3 Terminals)

**Step 1**: Copy the config files
```bash
cp config1.json config.json  # Terminal 1
cp config2.json config.json  # Terminal 2  
cp config3.json config.json  # Terminal 3
```

**Step 2**: Start all nodes simultaneously
```bash
# Terminal 1
python main.py

# Terminal 2 (different directory or copy config2.json to config.json)
python main.py

# Terminal 3 (different directory or copy config3.json to config.json)  
python main.py
```

**That's it!** Each node will automatically:
- Start on its configured port
- Load the same rating data
- Begin GOSSIP training with other nodes
- Exchange model parameters
- Show final recommendations
