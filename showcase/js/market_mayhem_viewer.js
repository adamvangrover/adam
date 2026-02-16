// Market Mayhem Viewer Interaction Logic

let isRedactionMode = false;

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

function toggleRedactMode() {
    isRedactionMode = !isRedactionMode;
    document.body.classList.toggle('redaction-mode', isRedactionMode);

    if (isRedactionMode) {
        showToast('REDACTION MODE: ENABLED. CLICK TEXT TO REDACT.');
    } else {
        showToast('REDACTION MODE: DISABLED.');
    }
}

function printDoc() {
    window.print();
}

function shareDoc() {
    const url = window.location.href;
    navigator.clipboard.writeText(url).then(() => {
        showToast('LINK COPIED TO CLIPBOARD');
    }).catch(err => {
        console.error('Failed to copy: ', err);
        showToast('ERROR COPYING LINK');
    });
}

function formatJSON(json) {
    if (typeof json !== 'string') {
        json = JSON.stringify(json, undefined, 2);
    }
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        var cls = 'json-number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'json-key';
            } else {
                cls = 'json-string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'json-boolean';
        } else if (/null/.test(match)) {
            cls = 'json-null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
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
        try {
            const rawText = rawDataArea.value || "{}";
            const jsonData = JSON.parse(rawText);
            content.innerHTML = formatJSON(jsonData);
        } catch (e) {
            content.innerText = rawDataArea.value || "NO SOURCE DATA AVAILABLE.";
        }
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

// Global Event Listeners
document.addEventListener('click', function(e) {
    if (!isRedactionMode) return;

    // Check if clicked element is redactable (p, span, li, h1-h6) inside paper-sheet
    if (e.target.closest('.paper-sheet')) {
        const tag = e.target.tagName.toLowerCase();
        if (['p', 'span', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div'].includes(tag)) {
            // Don't redact the entire sheet or wrappers, specific text blocks only
            if (!e.target.classList.contains('paper-sheet') && !e.target.classList.contains('newsletter-wrapper')) {
                e.target.classList.toggle('redacted');
            }
        }
    }
});

// Initialize Chart if data exists
document.addEventListener('DOMContentLoaded', () => {
    // Note: The main logic for chart initialization is often injected directly by the generator script
    // into the page to pass JSON data. However, if we move towards a cleaner separation,
    // we would look for a data attribute or global variable here.

    // Example: If we had <div id="chart-data" data-metrics='...'>
    /*
    const chartDataEl = document.getElementById('chart-data');
    if (chartDataEl) {
        const metrics = JSON.parse(chartDataEl.dataset.metrics);
        // ... init chart ...
    }
    */

   // Current implementation relies on the generator script injecting the Chart.js init code block.
   // This file handles interactions (redaction, verification, etc.)
});
