// Market Mayhem Viewer - Integrated Runtime
// Merged functionality: System 2 Audits + Main Interactive Features
// Priority: Live Runtime > Client Fallback > Mock Simulation

console.log("Market Mayhem Viewer: Initializing Neural Link...");

// ==========================================
// 1. UTILITIES & UI COMPONENTS
// ==========================================

/**
 * Displays a "Cyberpunk" style toast notification.
 * Merged styling for high visibility.
 */
function showToast(message) {
    let container = document.getElementById('toast-container');
    
    // Auto-create container if missing (Robustness)
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.style.position = 'fixed';
        container.style.top = '20px';
        container.style.right = '20px';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = 'cyber-toast';
    toast.innerText = '> ' + message;
    
    // Fallback inline styles if CSS classes aren't loaded
    if (!document.querySelector('style')) {
        toast.style.background = 'rgba(0, 0, 0, 0.9)';
        toast.style.border = '1px solid #00f3ff';
        toast.style.color = '#00f3ff';
        toast.style.padding = '12px 24px';
        toast.style.marginBottom = '10px';
        toast.style.fontFamily = "'JetBrains Mono', 'Courier New', monospace";
        toast.style.boxShadow = '0 0 15px rgba(0, 243, 255, 0.3)';
        toast.style.fontSize = '0.85rem';
    }

    container.appendChild(toast);

    // Animation logic
    requestAnimationFrame(() => {
        toast.style.opacity = '1';
        toast.style.transform = 'translateX(0)';
    });

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(20px)';
        setTimeout(() => toast.remove(), 500);
    }, 3000);
}

/**
 * Syntax highlights JSON string for the Source Viewer.
 * From Main branch.
 */
function syntaxHighlight(json) {
    if (typeof json !== 'string') {
         json = JSON.stringify(json, undefined, 2);
    }
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        var cls = 'color: #f08d49;'; // default number/bool
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'color: #00f3ff; font-weight: bold;'; // key
            } else {
                cls = 'color: #aaffaa;'; // string
            }
        } else if (/true|false/.test(match)) {
            cls = 'color: #ff0055;'; // boolean
        } else if (/null/.test(match)) {
            cls = 'color: #888;'; // null
        }
        return '<span style="' + cls + '">' + match + '</span>';
    });
}

// ==========================================
// 2. DATA LAYER (Live -> Client -> Mock)
// ==========================================

const State = {
    data: null,
    sourceType: 'UNKNOWN', // 'LIVE', 'STATIC', 'MOCK'
    isRedactionMode: false
};

async function initializeData() {
    // 1. Try Live Runtime (API)
    try {
        const response = await fetch('/api/market_mayhem/latest', { method: 'HEAD' });
        if (response.ok) {
            const jsonResp = await fetch('/api/market_mayhem/latest');
            State.data = await jsonResp.json();
            State.sourceType = 'LIVE RUNTIME';
            showToast("CONNECTED: LIVE DATA FEED ACTIVE");
            updateRawDataViewer(State.data);
            return;
        }
    } catch (e) {
        console.log("Live fetch failed, falling back to static...");
    }

    // 2. Try Client-Side Static (Hidden Input)
    const hiddenInput = document.getElementById('hidden-raw-data');
    if (hiddenInput && hiddenInput.value) {
        try {
            State.data = JSON.parse(hiddenInput.value);
            State.sourceType = 'STATIC ARCHIVE';
            showToast("LOADED: LOCAL ARCHIVE DATA");
            return;
        } catch (e) {
            console.error("Static JSON parse error");
        }
    }

    // 3. Fallback to Mock/Simulation
    State.data = generateMockData();
    State.sourceType = 'SIMULATION';
    showToast("WARNING: RUNNING IN SIMULATION MODE");
    // Inject mock data into viewer for consistency
    if (hiddenInput) hiddenInput.value = JSON.stringify(State.data, null, 2);
}

function generateMockData() {
    return {
        "meta": { "timestamp": new Date().toISOString(), "simulation": true },
        "market_sentiment": "EXTREME FEAR",
        "sp500_prediction": "3850.00",
        "notable_movements": ["NVDA -12%", "JPM +2%"]
    };
}

// ==========================================
// 3. INTERACTIVE FEATURES
// ==========================================

/**
 * Toggles visual theme. Merges 'tier' concept with 'dark-mode'.
 */
function toggleTheme() {
    document.body.classList.toggle('dark-mode');
    document.body.classList.toggle('tier-high'); // Maintain feature branch class compatibility
    
    const isDark = document.body.classList.contains('dark-mode');
    showToast(`VISUAL INTERFACE: ${isDark ? 'HIGH CONTRAST (DARK)' : 'STANDARD (LIGHT)'}`);
}

/**
 * Source Data Viewer Logic
 */
function toggleSource() {
    const modal = document.getElementById('raw-source-modal');
    const content = document.getElementById('raw-source-content');
    const rawDataArea = document.getElementById('hidden-raw-data');

    if (!modal || !content) return;

    if (modal.style.display === 'flex' || modal.style.display === 'block') {
        modal.style.display = 'none';
    } else {
        // Ensure we display the data currently in State if the text area is empty
        const rawText = rawDataArea ? rawDataArea.value : JSON.stringify(State.data, null, 2);
        
        try {
            const jsonObj = JSON.parse(rawText || "{}");
            content.innerHTML = '<pre style="white-space: pre-wrap;">' + syntaxHighlight(JSON.stringify(jsonObj, null, 2)) + '</pre>';
        } catch (e) {
            content.innerHTML = '<pre>' + (rawText || "NO DATA") + '</pre>';
        }
        
        modal.style.display = 'flex'; // Use flex for centering if CSS supports it
    }
}

function updateRawDataViewer(jsonData) {
    const rawDataArea = document.getElementById('hidden-raw-data');
    if (rawDataArea) {
        rawDataArea.value = JSON.stringify(jsonData, null, 2);
    }
}

// ==========================================
// 4. VERIFICATION & SECURITY (System 2)
// ==========================================

/**
 * Expansive verification logic.
 * Combines provenance checks from 'main' with the overlay structure.
 */
function runVerification() {
    const overlay = document.getElementById('system2-overlay');
    const content = document.getElementById('sys2-content');
    const status = document.getElementById('verification-status');
    const stamp = document.getElementById('top-secret-stamp');
    
    // Attempt to find real hash, fallback to generated one
    const provHash = document.querySelector('.provenance-log .log-entry span[style*="terminal-green"]')?.innerText 
                     || "HASH-" + Math.random().toString(16).substring(2, 10).toUpperCase();

    if (!overlay || !content) {
        showToast("ERROR: VERIFICATION MODULE NOT FOUND");
        return;
    }

    overlay.style.display = 'flex';
    content.innerHTML = '';
    if (stamp) stamp.style.display = 'none'; // Reset stamp

    // Expanded step sequence
    const steps = [
        { text: `INITIALIZING SYSTEM 2 PROTOCOL [SOURCE: ${State.sourceType}]...`, delay: 500 },
        { text: "ESTABLISHING SECURE HANDSHAKE...", delay: 1000 },
        { text: "DECRYPTING LAYERS [AES-256]...", delay: 1400 },
        { text: "VALIDATING DOM STRUCTURE INTEGRITY...", delay: 1800 },
        { text: `VERIFYING CRYPTOGRAPHIC SIGNATURE: ${provHash}`, delay: 2400 },
        { text: "ANALYZING BIAS VECTORS...", delay: 2800 },
        { text: "CROSS-REFERENCING KNOWLEDGE GRAPH...", delay: 3200 },
        { text: "AUDIT COMPLETE. INTEGRITY VERIFIED.", delay: 3800 }
    ];

    // Background Matrix-style hex rain effect
    const hexInterval = setInterval(() => {
        if (overlay.style.display === 'none') {
            clearInterval(hexInterval);
            return;
        }
        const hex = Math.random().toString(16).substring(2, 14).toUpperCase();
        const div = document.createElement('div');
        div.className = 'sys-bg-text'; // Ensure CSS handles opacity/position
        div.style.color = '#003300';
        div.style.fontSize = '0.6em';
        div.style.fontFamily = 'monospace';
        div.innerText = hex + ' :: ' + hex;
        // Optional: append to a background container if it existed, otherwise skip
    }, 100);

    // Execute steps
    steps.forEach((step, index) => {
        setTimeout(() => {
            const div = document.createElement('div');
            div.className = 'sys2-line';
            div.innerText = '> ' + step.text;
            div.style.color = index === steps.length - 1 ? '#00ff00' : '#00f3ff';
            div.style.fontFamily = "'JetBrains Mono', monospace";
            div.style.margin = '4px 0';
            
            content.appendChild(div);
            content.scrollTop = content.scrollHeight;

            if (index === steps.length - 1) {
                clearInterval(hexInterval);
                setTimeout(() => {
                    overlay.style.display = 'none';
                    showToast('DOCUMENT VERIFIED // ACCESS GRANTED');
                    
                    if (status) {
                        status.innerText = "VERIFIED // AUTHENTICATED";
                        status.style.color = "#aaffaa";
                        status.style.textShadow = "0 0 10px #00ff00";
                    }
                    if (stamp) {
                        stamp.style.display = 'block';
                        stamp.style.animation = 'stamp-bounce 0.4s ease-out';
                    }
                }, 1200);
            }
        }, step.delay);
    });
}

// ==========================================
// 5. REDACTION & UTILS
// ==========================================

function toggleRedactionMode() {
    State.isRedactionMode = !State.isRedactionMode;
    if (State.isRedactionMode) {
        document.body.classList.add('redaction-mode');
        showToast('REDACTION PROTOCOL: ACTIVE. CLICK TEXT TO REDACT.');
    } else {
        document.body.classList.remove('redaction-mode');
        showToast('REDACTION PROTOCOL: DISABLED.');
    }
}

function handleRedactionClick(e) {
    if (!State.isRedactionMode) {
        // Allow peeking under redaction even if mode is off
        if (e.target.classList.contains('redacted')) {
            e.target.classList.toggle('redaction-revealed');
        }
        return;
    }

    const validTags = ['P', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'LI', 'SPAN', 'STRONG', 'EM', 'TD', 'TH'];
    // Recursive check for parent compatibility if clicking deep elements
    let target = e.target;
    while (target && target !== this) {
        if (validTags.includes(target.tagName)) {
            target.classList.toggle('redacted');
            break;
        }
        target = target.parentNode;
    }
}

function shareReport() {
    const url = window.location.href;
    navigator.clipboard.writeText(url).then(() => {
        showToast('SECURE LINK COPIED TO CLIPBOARD');
    }).catch(() => showToast('COPY FAILED'));
}

function downloadReport() {
    const title = document.title || 'market_mayhem_report';
    const content = document.getElementById('paper-sheet')?.innerText || document.body.innerText;
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = title.replace(/[^a-z0-9]/gi, '_').toLowerCase() + '.txt';
    a.click();
    URL.revokeObjectURL(url);
    showToast('DOWNLOADING DECRYPTED ASSET...');
}

function printReport() {
    window.print();
}

// ==========================================
// 6. INITIALIZATION
// ==========================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize Data Source
    initializeData();

    // Attach Event Listeners
    const sheet = document.getElementById('paper-sheet');
    if (sheet) {
        sheet.addEventListener('click', handleRedactionClick);
    }
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'b') { // Ctrl+B for Verification
            runVerification();
        }
        if (e.ctrlKey && e.key === 'm') { // Ctrl+M for Redaction
            toggleRedactionMode();
        }
    });
});