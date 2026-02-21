/**
 * MARKET MAYHEM: MULTIPLAYER CLIENT
 * -----------------------------------------------------------------------------
 * Connects to the War Room Server and renders the battlefield.
 */

const SERVER_URL = 'ws://localhost:8765';
let socket;
let isConnected = false;
let myTeam = null;
let myName = "Operator";
let lastVix = 20.0;
let simulatedVix = 20.0;

document.getElementById('btnJoin').addEventListener('click', () => {
    const nameInput = document.getElementById('playerName').value;
    if (!nameInput) {
        alert("Please enter a callsign.");
        return;
    }
    if (!selectedTeam) {
        alert("Please select a team.");
        return;
    }

    myName = nameInput;
    myTeam = selectedTeam;

    initGame();
});

function initGame() {
    // Hide Login, Show Game
    document.getElementById('login-panel').style.display = 'none';
    document.getElementById('game-controls').style.display = 'block';
    document.getElementById('game-board').style.display = 'block';
    document.getElementById('lobby-message').style.display = 'none';

    // Show correct buttons based on team
    if (myTeam === 'blue') {
        document.getElementById('btnAction1').innerText = "RATE HIKE (STABILIZE)";
        document.getElementById('btnAction1').className = "blue-team";
        document.getElementById('btnAction2').style.display = 'inline-block';
        document.getElementById('btnAction2').innerText = "QE INJECTION";
    } else {
        document.getElementById('btnAction1').innerText = "SHORT LADDER ATTACK";
        document.getElementById('btnAction1').className = "primary"; // Red
        document.getElementById('btnAction2').style.display = 'inline-block';
        document.getElementById('btnAction2').innerText = "SPREAD FUD RUMOR";
        document.getElementById('btnAction2').className = "primary"; // Red
    }

    connectToServer();
}

function connectToServer() {
    document.getElementById('statusText').innerText = "CONNECTING...";

    try {
        socket = new WebSocket(SERVER_URL);

        socket.onopen = () => {
            isConnected = true;
            document.getElementById('statusText').innerText = "ONLINE (SECURE UPLINK)";
            document.getElementById('statusText').style.color = "#00ff9d";

            // Join
            socket.send(JSON.stringify({
                type: 'JOIN',
                payload: { name: myName, team: myTeam }
            }));
        };

        socket.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            handleServerMessage(msg);
        };

        socket.onclose = () => {
            isConnected = false;
            document.getElementById('statusText').innerText = "OFFLINE (SIMULATION MODE)";
            document.getElementById('statusText').style.color = "#ff9d00";
            startSimulationMode();
        };

        socket.onerror = (error) => {
            console.warn("Server connection failed, falling back to local simulation.", error);
            // Don't alert, just fallback
        };

    } catch (e) {
        console.warn("WebSocket not supported or failed.", e);
        startSimulationMode();
    }
}

function handleServerMessage(msg) {
    if (msg.type === 'UPDATE') {
        const state = msg.payload;
        updateUI(state);
    } else if (msg.type === 'LOG') {
        addLog(msg.payload.text, msg.payload.team); // Assuming log payload has team info? Server log function didn't send team, fixing client logic assumption.
        // Actually server log just sends text.
        addLog(msg.payload.text);
    }
}

function sendAction(actionType) {
    if (isConnected) {
        socket.send(JSON.stringify({
            type: 'ACTION',
            payload: { action: actionType }
        }));
    } else {
        // Local Simulation Effect
        simulateAction(actionType);
    }
}

function updateUI(state) {
    // VIX
    const vix = state.vix;
    lastVix = vix;
    document.getElementById('vix-val').innerText = vix.toFixed(2);

    // Move Line (Map 0-100 to 100%-0% top)
    const topPct = 100 - vix;
    document.getElementById('vix-line').style.top = topPct + "%";

    // Scores
    document.getElementById('score-red').innerText = state.redScore || 0;
    document.getElementById('score-blue').innerText = state.blueScore || 0;
}

function addLog(text) {
    const feed = document.getElementById('log-feed');
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerText = `> ${text}`;
    feed.prepend(entry);
    if (feed.children.length > 20) feed.lastChild.remove();
}

// --- LOCAL SIMULATION (FALLBACK) ---

function startSimulationMode() {
    if (window.simInterval) return;
    window.simInterval = setInterval(() => {
        // Random Walk
        simulatedVix += (Math.random() - 0.5) * 2;
        simulatedVix = Math.max(5, Math.min(100, simulatedVix));

        updateUI({
            vix: simulatedVix,
            redScore: Math.floor(Math.random() * 1000), // Mock score
            blueScore: Math.floor(Math.random() * 1000)
        });
    }, 500);
}

function simulateAction(action) {
    addLog(`(LOCAL) You used ${action.toUpperCase()}`);
    if (myTeam === 'red') {
        simulatedVix += 5;
    } else {
        simulatedVix -= 5;
    }
}

// Hook up buttons
// Note: In HTML onclick="sendAction('attack')" works if functions are global.
// But modules are better. For this prototype, I'll expose them.
window.sendAction = sendAction;
