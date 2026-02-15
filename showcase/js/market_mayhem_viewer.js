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

function runVerification() {
    const overlay = document.getElementById('system2-overlay');
    const content = document.getElementById('sys2-content');
    const status = document.getElementById('verification-status');
    const provHash = document.querySelector('.provenance-log .log-entry span[style*="terminal-green"]')?.innerText || "UNKNOWN";

    if (!overlay || !content) return;

    overlay.style.display = 'flex';
    content.innerHTML = '';

    const lines = [
        "INITIATING SYSTEM 2 PROTOCOL...",
        "VERIFYING CRYPTOGRAPHIC SIGNATURE...",
        "HASH: " + provHash,
        "CHECKING CHAIN OF CUSTODY...",
        "ANALYZING SENTIMENT VECTORS...",
        "CROSS-REFERENCING KNOWLEDGE GRAPH...",
        "VERIFICATION COMPLETE."
    ];

    let i = 0;
    function addLine() {
        if (i < lines.length) {
            const div = document.createElement('div');
            div.className = 'system2-line';
            div.innerText = '> ' + lines[i];
            div.style.opacity = 1;
            content.appendChild(div);
            i++;
            setTimeout(addLine, 400);
        } else {
            setTimeout(() => {
                overlay.style.display = 'none';
                showToast('DOCUMENT VERIFIED');
                if(status) {
                    status.innerText = "VERIFIED // AUTHENTICATED";
                    status.style.color = "#aaffaa";
                    status.style.textShadow = "0 0 10px #00ff00";
                }
            }, 1000);
        }
    }
    addLine();
}

// Initialize Chart if data exists
document.addEventListener('DOMContentLoaded', () => {
    // Check for metrics data in a global variable or data attribute
    // The generator injects `const metricsData = ...` in a separate script block usually.
    // If we move that logic here, we need to read it from DOM.
    // Ideally, we keep the data injection inline, and the chart rendering here or inline.
    // For now, let's leave the chart logic inline in the generator as it depends on `metrics_json` insertion.
});
