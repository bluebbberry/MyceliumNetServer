from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import sqlite3
import json
import time
from datetime import datetime
import uuid

app = FastAPI(title="Mycelium Net Registry", version="1.0.0")


# Database setup
def init_db():
    conn = sqlite3.connect('mycelium_registry.db')
    cursor = conn.cursor()

    # Groups table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_groups (
            group_id TEXT PRIMARY KEY,
            model_type TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            performance_metric REAL DEFAULT 0.0,
            member_count INTEGER DEFAULT 0,
            max_capacity INTEGER DEFAULT 10,
            join_policy TEXT DEFAULT 'open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Nodes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            node_address TEXT NOT NULL,
            current_group_id TEXT,
            local_performance REAL DEFAULT 0.0,
            last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (current_group_id) REFERENCES learning_groups (group_id)
        )
    ''')

    conn.commit()
    conn.close()


# Pydantic models
class LearningGroup(BaseModel):
    group_id: Optional[str] = None
    model_type: str
    dataset_name: str
    performance_metric: float = 0.0
    member_count: int = 0
    max_capacity: int = 10
    join_policy: str = "open"


class Node(BaseModel):
    node_id: Optional[str] = None
    node_address: str
    current_group_id: Optional[str] = None
    local_performance: float = 0.0


class JoinRequest(BaseModel):
    node_id: str
    group_id: str


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# NEW: Network state endpoint for visualization
@app.get("/network/state")
async def get_network_state():
    """Get current network state for visualization"""
    conn = sqlite3.connect('mycelium_registry.db')
    cursor = conn.cursor()

    # Get all groups
    cursor.execute('SELECT * FROM learning_groups')
    groups_data = cursor.fetchall()

    # Get all nodes
    cursor.execute('SELECT * FROM nodes')
    nodes_data = cursor.fetchall()

    conn.close()

    # Format for visualizer
    groups = []
    for g in groups_data:
        groups.append({
            "id": g[0],  # group_id
            "performance": float(g[3]),  # performance_metric
            "members": [],  # Will be populated below
            "model_type": g[1],
            "dataset_name": g[2],
            "member_count": g[4],
            "max_capacity": g[5]
        })

    nodes = []
    for n in nodes_data:
        nodes.append({
            "id": n[0],  # node_id
            "performance": float(n[3]),  # local_performance
            "group": n[2],  # current_group_id
            "address": n[1],  # node_address
            "last_heartbeat": n[4]
        })

    # Add member lists to groups
    for group in groups:
        group["members"] = [n["id"] for n in nodes if n["group"] == group["id"]]

    return {
        "groups": groups,
        "nodes": nodes,
        "timestamp": datetime.now().isoformat()
    }


# Registry endpoints
@app.post("/groups", response_model=dict)
async def create_group(group: LearningGroup):
    conn = sqlite3.connect('mycelium_registry.db')
    cursor = conn.cursor()

    group_id = str(uuid.uuid4())
    cursor.execute('''
        INSERT INTO learning_groups 
        (group_id, model_type, dataset_name, performance_metric, max_capacity, join_policy)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (group_id, group.model_type, group.dataset_name,
          group.performance_metric, group.max_capacity, group.join_policy))

    conn.commit()
    conn.close()

    return {"group_id": group_id, "status": "created"}


@app.get("/groups", response_model=List[dict])
async def list_groups():
    conn = sqlite3.connect('mycelium_registry.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM learning_groups ORDER BY performance_metric DESC')
    groups = cursor.fetchall()
    conn.close()

    return [
        {
            "group_id": g[0], "model_type": g[1], "dataset_name": g[2],
            "performance_metric": g[3], "member_count": g[4], "max_capacity": g[5],
            "join_policy": g[6], "created_at": g[7], "last_updated": g[8]
        }
        for g in groups
    ]


@app.post("/nodes/register", response_model=dict)
async def register_node(node: Node):
    conn = sqlite3.connect('mycelium_registry.db')
    cursor = conn.cursor()

    node_id = str(uuid.uuid4()) if not node.node_id else node.node_id
    cursor.execute('''
        INSERT OR REPLACE INTO nodes 
        (node_id, node_address, local_performance, last_heartbeat)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    ''', (node_id, node.node_address, node.local_performance))

    conn.commit()
    conn.close()

    return {"node_id": node_id, "status": "registered"}


@app.get("/nodes", response_model=List[dict])
async def list_nodes():
    """Get all registered nodes"""
    conn = sqlite3.connect('mycelium_registry.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM nodes ORDER BY last_heartbeat DESC')
    nodes = cursor.fetchall()
    conn.close()

    return [
        {
            "node_id": n[0], "node_address": n[1], "current_group_id": n[2],
            "local_performance": n[3], "last_heartbeat": n[4]
        }
        for n in nodes
    ]


@app.post("/groups/join", response_model=dict)
async def join_group(request: JoinRequest):
    conn = sqlite3.connect('mycelium_registry.db')
    cursor = conn.cursor()

    # Check if group has capacity
    cursor.execute('SELECT member_count, max_capacity FROM learning_groups WHERE group_id = ?',
                   (request.group_id,))
    result = cursor.fetchone()

    if not result:
        conn.close()
        raise HTTPException(status_code=404, detail="Group not found")

    member_count, max_capacity = result
    if member_count >= max_capacity:
        conn.close()
        raise HTTPException(status_code=400, detail="Group at capacity")

    # Check if node was in another group and decrement that group's count
    cursor.execute('SELECT current_group_id FROM nodes WHERE node_id = ?', (request.node_id,))
    old_group = cursor.fetchone()
    if old_group and old_group[0]:
        cursor.execute('UPDATE learning_groups SET member_count = member_count - 1 WHERE group_id = ?',
                       (old_group[0],))

    # Update node's group and increment new group member count
    cursor.execute('UPDATE nodes SET current_group_id = ? WHERE node_id = ?',
                   (request.group_id, request.node_id))
    cursor.execute('UPDATE learning_groups SET member_count = member_count + 1 WHERE group_id = ?',
                   (request.group_id,))

    conn.commit()
    conn.close()

    return {"status": "joined", "group_id": request.group_id}


@app.put("/groups/{group_id}/performance")
async def update_group_performance(group_id: str, performance: dict):
    conn = sqlite3.connect('mycelium_registry.db')
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE learning_groups 
        SET performance_metric = ?, last_updated = CURRENT_TIMESTAMP 
        WHERE group_id = ?
    ''', (performance["metric"], group_id))

    conn.commit()
    conn.close()

    return {"status": "updated"}


@app.put("/nodes/{node_id}/performance")
async def update_node_performance(node_id: str, performance: dict):
    """Update node performance"""
    conn = sqlite3.connect('mycelium_registry.db')
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE nodes 
        SET local_performance = ?, last_heartbeat = CURRENT_TIMESTAMP 
        WHERE node_id = ?
    ''', (performance["metric"], node_id))

    conn.commit()
    conn.close()

    return {"status": "updated"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)