import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
MAX_NAME_LENGTH = 20
ALLOWED_TEAMS = {"red", "blue", "spectator"}
ALLOWED_ACTIONS = {"attack", "rumor", "defend", "qe"}

# --- Game State ---

@dataclass
class GameState:
    vix: float = 20.0
    red_score: int = 0
    blue_score: int = 0
    players: Dict[str, dict] = field(default_factory=dict)
    log: List[dict] = field(default_factory=list)

    def to_dict(self):
        return {
            "vix": self.vix,
            "redScore": self.red_score,
            "blueScore": self.blue_score,
            "players": self.players,
            "log": self.log
        }

STATE = GameState()
CLIENTS = set()

async def register(websocket):
    CLIENTS.add(websocket)
    logging.info(f"Client connected. Total clients: {len(CLIENTS)}")

async def unregister(websocket, player_id=None):
    if websocket in CLIENTS:
        CLIENTS.remove(websocket)
    if player_id and player_id in STATE.players:
        del STATE.players[player_id]
        broadcast_message('PLAYER_LEFT', {'id': player_id})
    logging.info(f"Client disconnected. Total clients: {len(CLIENTS)}")

def broadcast(message):
    if not CLIENTS:
        return

    payload = json.dumps(message)
    asyncio.create_task(broadcast_task(payload))

async def broadcast_task(payload):
    # Copy set to avoid size change during iteration
    for ws in list(CLIENTS):
        try:
            await ws.send(payload)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logging.error(f"Error broadcasting: {e}")

def broadcast_message(msg_type, payload):
    broadcast({"type": msg_type, "payload": payload})

def log_event(text, team=None):
    entry = {"text": text, "time": time.time(), "team": team}
    STATE.log.insert(0, entry)
    if len(STATE.log) > 20:
        STATE.log.pop()
    broadcast_message('LOG', entry)

async def game_loop():
    """Simulates market dynamics."""
    logging.info("Starting simulation loop...")
    while True:
        # Natural drift / Mean reversion
        noise = (random.random() - 0.5) * 0.1
        STATE.vix += noise

        # Mean reversion to 20
        if STATE.vix > 20:
            STATE.vix -= 0.01
        elif STATE.vix < 20:
            STATE.vix += 0.01

        STATE.vix = max(5.0, min(100.0, STATE.vix))

        broadcast_message('UPDATE', STATE.to_dict())
        await asyncio.sleep(0.1) # 10 ticks per second

async def handler(websocket):
    player_id = None
    await register(websocket)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                payload = data.get("payload", {})

                if msg_type == "JOIN":
                    player_id = str(id(websocket)) # Simple ID

                    # Validate Name
                    raw_name = str(payload.get("name", "Anonymous")).strip()
                    # Truncate to MAX_NAME_LENGTH
                    name = raw_name[:MAX_NAME_LENGTH]
                    if not name:
                        name = "Anonymous"

                    # Validate Team
                    raw_team = str(payload.get("team", "spectator")).lower()
                    team = raw_team if raw_team in ALLOWED_TEAMS else "spectator"

                    STATE.players[player_id] = {"name": name, "team": team}
                    log_event(f"{name} joined Team {team.upper()}", team)
                    # Send welcome
                    await websocket.send(json.dumps({"type": "WELCOME", "payload": {"id": player_id, "state": STATE.to_dict()}}))

                elif msg_type == "ACTION":
                    if not player_id: continue
                    player = STATE.players.get(player_id)
                    if not player: continue

                    action = str(payload.get("action", "")).lower()
                    if action not in ALLOWED_ACTIONS:
                         continue # Invalid action

                    team = player["team"]
                    log_event(f"{player['name']} used {action.upper()}", team)

                    if team == "red":
                        if action == "attack":
                            STATE.vix += 2.0
                            STATE.red_score += 100
                        elif action == "rumor": # Fallback/Alternative action
                             STATE.vix += 1.0
                             STATE.red_score += 50
                    elif team == "blue":
                        if action == "defend":
                            STATE.vix -= 1.5
                            STATE.blue_score += 50
                        elif action == "qe": # Fallback/Alternative action
                            STATE.vix -= 2.0
                            STATE.blue_score += 100

                    # Clamp VIX
                    STATE.vix = max(5.0, min(100.0, STATE.vix))
            except json.JSONDecodeError:
                logging.warning("Received invalid JSON")
            except Exception as e:
                logging.error(f"Error handling message: {e}")

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        await unregister(websocket, player_id)

async def main():
    # Start the server
    # Note: Using localhost to restrict to local access for safety in this environment
    port = 8765
    server = await websockets.serve(handler, "localhost", port)
    logging.info(f"WAR ROOM SERVER STARTED ON ws://localhost:{port}")

    # Start the simulation loop
    asyncio.create_task(game_loop())

    await asyncio.Future() # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Server stopped.")
