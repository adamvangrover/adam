// Market Mayhem Viewer Interaction Logic

function showToast(msg) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = 'cyber-toast';
    toast.innerText = '> ' + msg;
    container.appendChild(toast);

    // Trigger reflow for animation
    void toast.offsetWidth;
    toast.style.opacity = '1';

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function toggleTheme() {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    showToast('THEME: ' + (isDark ? 'DARK_MODE' : 'LIGHT_MODE'));
}

function toggleSource() {
    const modal = document.getElementById('raw-source-modal');
    const content = document.getElementById('raw-source-content');
    const rawDataArea = document.getElementById('hidden-raw-data');

    if (!modal || !content || !rawDataArea) {
        console.error("Source viewer elements missing");
        return;
    }

    if (modal.style.display === 'flex') {
        modal.style.display = 'none';
    } else {
        content.innerText = rawDataArea.value || "NO SOURCE DATA AVAILABLE.";
        modal.style.display = 'flex';
    }
}

// --- Verification Logic ---

function runVerification() {
    const overlay = document.getElementById('system2-overlay');
    const content = document.getElementById('sys2-content');
    const status = document.getElementById('verification-status');
    const provHash = document.querySelector('.provenance-log .log-entry span[style*="terminal-green"]')?.innerText || "UNKNOWN";
    const stamp = document.getElementById('top-secret-stamp');

    if (!overlay || !content) return;

    overlay.style.display = 'flex';
    content.innerHTML = '';

    const lines = [
        "INITIATING SYSTEM 2 PROTOCOL...",
        "VERIFYING CRYPTOGRAPHIC SIGNATURE...",
        "HASH: " + provHash,
        "CHECKING CHAIN OF CUSTODY...",
        "SCANNING FOR ANOMALIES...",
        "ANALYZING SENTIMENT VECTORS...",
        "CROSS-REFERENCING KNOWLEDGE GRAPH...",
        "VERIFICATION COMPLETE."
    ];

    let i = 0;

    function addRandomHex() {
        const hex = Math.random().toString(16).substring(2, 14).toUpperCase();
        const div = document.createElement('div');
        div.className = 'system2-line';
        div.style.color = '#005500';
        div.style.fontSize = '0.7em';
        div.innerText = hex + ' ' + hex + ' ' + hex;
        content.appendChild(div);
        content.scrollTop = content.scrollHeight;
    }

    function addLine() {
        if (i < lines.length) {
            const div = document.createElement('div');
            div.className = 'system2-line';
            div.innerText = '> ' + lines[i];
            div.style.opacity = 1;
            content.appendChild(div);
            content.scrollTop = content.scrollHeight;

            // Burst of hex codes between lines
            let hexCount = 0;
            const hexInterval = setInterval(() => {
                addRandomHex();
                hexCount++;
                if(hexCount > 5) clearInterval(hexInterval);
            }, 50);

            i++;
            setTimeout(addLine, 600);
        } else {
            setTimeout(() => {
                overlay.style.display = 'none';
                showToast('DOCUMENT VERIFIED');
                if(status) {
                    status.innerText = "VERIFIED // AUTHENTICATED";
                    status.style.color = "#aaffaa";
                    status.style.textShadow = "0 0 10px #00ff00";
                }
                if (stamp) {
                    stamp.style.display = 'block';
                    // Add a slight rubber stamp animation class if we had one
                }
            }, 1000);
        }
    }
    addLine();
}

// --- Redaction Logic ---

let isRedactionMode = false;

function toggleRedactionMode() {
    isRedactionMode = !isRedactionMode;
    if (isRedactionMode) {
        document.body.classList.add('redaction-mode');
        showToast('REDACTION MODE: ACTIVE (CLICK TEXT TO REDACT)');
    } else {
        document.body.classList.remove('redaction-mode');
        showToast('REDACTION MODE: DISABLED');
    }
}

function handleRedactionClick(e) {
    if (!isRedactionMode) {
        // If clicked on an existing redacted element, toggle reveal
        if (e.target.classList.contains('redacted')) {
            e.target.classList.toggle('redaction-revealed');
        }
        return;
    }

    // Only redact text containers
    const validTags = ['P', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'LI', 'SPAN', 'STRONG', 'EM'];
    if (validTags.includes(e.target.tagName)) {
        e.target.classList.toggle('redacted');
    }
}

// --- Print Logic ---

function printReport() {
    window.print();
}

// --- Initialization ---

document.addEventListener('DOMContentLoaded', () => {
    // Attach click listener for redaction
    const sheet = document.getElementById('paper-sheet');
    if (sheet) {
        sheet.addEventListener('click', handleRedactionClick);
    }
});
