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

function syntaxHighlight(json) {
    if (typeof json != 'string') {
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
        const rawText = rawDataArea.value || "{}";
        try {
            // Re-parse to ensure formatting
            const jsonObj = JSON.parse(rawText);
            const formatted = JSON.stringify(jsonObj, null, 2);
            content.innerHTML = syntaxHighlight(formatted);
        } catch (e) {
            content.innerText = rawText || "NO SOURCE DATA AVAILABLE.";
        }
        modal.style.display = 'flex';
    }
}

function shareReport() {
    const url = window.location.href;
    navigator.clipboard.writeText(url).then(() => {
        showToast('LINK COPIED TO CLIPBOARD');
    }).catch(err => {
        console.error('Failed to copy: ', err);
        showToast('COPY FAILED');
    });
}

function downloadReport() {
    const title = document.title || 'report';
    const content = document.getElementById('paper-sheet').innerText;
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = title.replace(/[^a-z0-9]/gi, '_').toLowerCase() + '.txt';
    a.click();
    URL.revokeObjectURL(url);
    showToast('DOWNLOAD STARTED');
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

    const steps = [
        { text: "INITIATING SYSTEM 2 PROTOCOL...", delay: 500 },
        { text: "ESTABLISHING SECURE HANDSHAKE...", delay: 800 },
        { text: "DECRYPTING LAYERS [AES-256]...", delay: 1200 },
        { text: "VERIFYING CRYPTOGRAPHIC SIGNATURE...", delay: 1500 },
        { text: "HASH: " + provHash, delay: 1800 },
        { text: "CHECKING CHAIN OF CUSTODY...", delay: 2200 },
        { text: "SCANNING FOR ANOMALIES...", delay: 2600 },
        { text: "ANALYZING SENTIMENT VECTORS...", delay: 3000 },
        { text: "CROSS-REFERENCING KNOWLEDGE GRAPH...", delay: 3500 },
        { text: "ACCESS GRANTED.", delay: 4000 }
    ];

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

    steps.forEach((step, index) => {
        setTimeout(() => {
            const div = document.createElement('div');
            div.className = 'system2-line';
            div.innerText = '> ' + step.text;
            div.style.opacity = 1;
            content.appendChild(div);
            content.scrollTop = content.scrollHeight;

             let hexCount = 0;
            const hexInterval = setInterval(() => {
                addRandomHex();
                hexCount++;
                if(hexCount > 3) clearInterval(hexInterval);
            }, 50);

            if (index === steps.length - 1) {
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
                    }
                }, 1000);
            }
        }, step.delay);
    });
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
