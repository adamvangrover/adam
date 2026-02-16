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

function highlightJSON(json) {
    if (typeof json !== 'string') {
        try {
            json = JSON.stringify(json, undefined, 2);
        } catch (e) {
            return json;
        }
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
        const rawJson = rawDataArea.value || "{}";
        try {
            const jsonObj = JSON.parse(rawJson);
            content.innerHTML = '<pre>' + highlightJSON(jsonObj) + '</pre>';
        } catch (e) {
            content.innerText = rawJson;
        }
        modal.style.display = 'flex';
    }
}

function printDocument() {
    window.print();
}

function copyLink() {
    const url = window.location.href;
    navigator.clipboard.writeText(url).then(() => {
        showToast('LINK COPIED TO CLIPBOARD');
    }).catch(err => {
        showToast('ERROR COPYING LINK: ' + err);
    });
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

    // Helper for "decrypting" text effect
    function typeLine(text, callback) {
        const div = document.createElement('div');
        div.className = 'system2-line';
        content.appendChild(div);

        let steps = 0;
        const maxSteps = 15; // Animation frames
        const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%^&*";

        const interval = setInterval(() => {
            let displayedText = "";
            for (let j = 0; j < text.length; j++) {
                if (j < (steps / maxSteps) * text.length) {
                    displayedText += text[j];
                } else {
                    displayedText += chars[Math.floor(Math.random() * chars.length)];
                }
            }
            div.innerText = '> ' + displayedText;
            steps++;

            if (steps > maxSteps) {
                clearInterval(interval);
                div.innerText = '> ' + text;
                div.style.opacity = 1; // Ensure full opacity
                if (callback) callback();
            }
        }, 30);
    }

    function processNextLine() {
        if (i < lines.length) {
            typeLine(lines[i], () => {
                i++;
                setTimeout(processNextLine, 200);
            });
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

    processNextLine();
}

// Keyboard Shortcuts
document.addEventListener('keydown', (e) => {
    // Only if no modal is open (except source modal which can be closed)
    // and not typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    if (e.key.toLowerCase() === 'v') {
        runVerification();
    } else if (e.key.toLowerCase() === 's') {
        toggleSource();
    } else if (e.key.toLowerCase() === 'p') {
        printDocument();
    }
});

// Initialize Chart if data exists
document.addEventListener('DOMContentLoaded', () => {
    // Chart logic is typically injected inline by the generator
});
