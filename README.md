# OpenFL

P2P + federated machine learning + nodes are able to switch learning groups (similar to p2pfl plus switching of nodes)
# GOSSIP Music Recommendation System

A distributed music recommendation system using GOSSIP protocol for federated learning.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configuration
Edit `config.json` to set your network configuration:
- `host`: Your node's IP address (default: localhost)
- `port`: Your node's port (default: 8000)  
- `peers`: List of other nodes to connect to

For testing on one machine, use different ports:
```json
{
  "host": "localhost",
  "port": 8000,
  "peers": [
    ["localhost", 8001],
    ["localhost", 8002]
  ]
}
```

# GOSSIP Music Recommendation System

A distributed music recommendation system using GOSSIP protocol for federated learning.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Files Needed
- `ratings.csv` - Your music rating data
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
3. Train locally
4. Start GOSSIP training
5. Save the model
6. Show recommendations

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

## Configuration Options

Edit `config.json` to customize:

```json
{
  "host": "localhost",           // Node IP address
  "port": 8000,                  // Node port
  "peers": [["localhost", 8001]], // Other nodes to connect to
  "data_file": "ratings.csv",    // Training data file
  "model_file": "model.pkl",     // Where to save model
  "gossip_rounds": 30,           // Number of training rounds
  "gossip_interval": 5,          // Seconds between rounds
  "auto_start_training": true,   // Start training automatically
  "auto_predict_after_training": true, // Show predictions when done
  "test_user_id": 1,            // User to test recommendations for
  "num_recommendations": 5       // Number of recommendations to show
}
```

## What You'll See

When you run the program, you'll see:
```
==================================================
🎵 GOSSIP Music Recommendation System  
==================================================
Loaded config: Node localhost:8000
Starting network on localhost:8000...
Loading training data from ratings.csv...
✅ Data loaded and local model trained!

🔄 Starting GOSSIP training for 30 rounds...
Connecting to peers and exchanging model parameters...
Training started! Press Ctrl+C to stop early.
==================================================
GOSSIP round 1: contacting localhost:8001
GOSSIP round 1 completed successfully
...
✅ GOSSIP training completed!

💾 Saving model to model.pkl...
✅ Model saved successfully!

🔮 Testing predictions...
🎵 Top 5 recommendations for user 1:
----------------------------------------
1. Song 105:  4.23 ⭐
2. Song 103:  4.18 ⭐  
3. Song 110:  3.95 ⭐
4. Song 108:  3.87 ⭐
5. Song 106:  3.72 ⭐
```

## Data Format

Your `ratings.csv` should look like:
```csv
user_id,song_id,rating
1,101,4.5
1,102,3.0
2,101,5.0
```

## Easy Testing Setup

For the simplest test, create 3 directories:
```bash
mkdir node1 node2 node3

# Copy all files to each directory
cp *.py *.csv requirements.txt node1/
cp *.py *.csv requirements.txt node2/  
cp *.py *.csv requirements.txt node3/

# Copy different configs
cp config1.json node1/config.json
cp config2.json node2/config.json
cp config3.json node3/config.json

# Run in 3 terminals
cd node1 && python main.py  # Terminal 1
cd node2 && python main.py  # Terminal 2
cd node3 && python main.py  # Terminal 3
```

## Data Format

The CSV file should have columns: `user_id`, `song_id`, `rating`

Example:
```csv
user_id,song_id,rating
1,101,4.5
1,102,3.0
2,101,5.0
```

## How It Works

1. **Local Training**: Each node trains a matrix factorization model on its local data
2. **GOSSIP Protocol**: Nodes randomly contact peers and exchange model parameters
3. **Parameter Averaging**: Received parameters are averaged with local parameters
4. **Convergence**: After many rounds, all nodes converge to similar models
5. **Recommendations**: The final model can recommend songs for any user

## Architecture

- `network.py`: TCP client/server for peer communication
- `train.py`: Matrix factorization model using scikit-learn
- `gossip.py`: GOSSIP protocol implementation
- `main.py`: Command-line interface
- `ratings.csv`: Sample training data
- `config.json`: Network configuration

## Limitations (MVP)

- No encryption or authentication
- Simple averaging (no weighted aggregation)
- Fixed network topology
- No fault tolerance
- No privacy protection

## Troubleshooting

**Connection refused**: Make sure all peer nodes are running and ports are correct in config.json

**No peers available**: Check that peer list in config.json includes other running nodes

**Model not fitted**: Make sure to load data before training

**Port already in use**: Change port numbers in config.json for each node