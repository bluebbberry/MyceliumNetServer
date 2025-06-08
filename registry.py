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

    # Update node's group and increment group member count
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)