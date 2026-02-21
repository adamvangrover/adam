/**
 * MARKET MAYHEM: WAR ROOM SERVER
 * -----------------------------------------------------------------------------
 * A simple WebSocket server to synchronize the Global VIX state.
 * Requires: npm install ws
 */

const WebSocket = require('ws');

// Configuration
const PORT = 8080;
const wss = new WebSocket.Server({ port: PORT });

console.log(`WAR ROOM SERVER INITIALIZED ON PORT ${PORT}`);

// Game State
let gameState = {
    vix: 20.0,
    redScore: 0,
    blueScore: 0,
    players: {}, // id -> { name, team }
    log: []
};

// Simulation Loop (10 ticks per second)
setInterval(() => {
    // Natural Decay/Drift
    // If VIX is high, it tends to come down (mean reversion) unless pushed
    // If VIX is low, it tends to stay low

    // Random noise
    let noise = (Math.random() - 0.5) * 0.1;
    gameState.vix += noise;

    // Clamp
    gameState.vix = Math.max(5, Math.min(100, gameState.vix));

    // Broadcast
    broadcast(JSON.stringify({ type: 'UPDATE', payload: gameState }));
}, 100);

wss.on('connection', (ws) => {
    const id = Math.random().toString(36).substring(7);
    ws.id = id;

    console.log(`Client connected: ${id}`);

    ws.send(JSON.stringify({ type: 'WELCOME', payload: { id, state: gameState } }));

    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);
            handleMessage(ws, data);
        } catch (e) {
            console.error("Invalid message:", message);
        }
    });

    ws.on('close', () => {
        console.log(`Client disconnected: ${id}`);
        delete gameState.players[id];
        broadcast(JSON.stringify({ type: 'PLAYER_LEFT', payload: { id } }));
    });
});

function handleMessage(ws, data) {
    switch (data.type) {
        case 'JOIN':
            const { name, team } = data.payload;
            gameState.players[ws.id] = { name, team };
            log(`${name} joined Team ${team.toUpperCase()}`);
            break;

        case 'ACTION':
            const player = gameState.players[ws.id];
            if (!player) return;

            const { action } = data.payload;
            log(`${player.name} used ${action.toUpperCase()}`);

            // Apply Effects
            if (player.team === 'red') {
                if (action === 'attack') {
                    gameState.vix += 2.0; // Panic
                    gameState.redScore += 100; // Profit from chaos
                }
            } else if (player.team === 'blue') {
                if (action === 'defend') {
                    gameState.vix -= 1.5; // Calm
                    gameState.blueScore += 50; // Stability points
                }
            }
            break;
    }
}

function broadcast(msg) {
    wss.clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(msg);
        }
    });
}

function log(text) {
    const entry = { text, time: Date.now() };
    gameState.log.unshift(entry);
    if (gameState.log.length > 20) gameState.log.pop();
    broadcast(JSON.stringify({ type: 'LOG', payload: entry }));
}
