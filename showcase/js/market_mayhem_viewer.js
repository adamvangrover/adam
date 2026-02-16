// Market Mayhem Viewer - Interactive Logic
// Handles Virtual Toolbar, System 2 Verification, and Raw Source Data

console.log("Market Mayhem Viewer Loaded.");

// 1. Raw Source Data Viewer
function toggleSource() {
    const modal = document.getElementById('raw-source-modal');
    const content = document.getElementById('raw-source-content');
    const hiddenData = document.getElementById('hidden-raw-data');

    if (modal.style.display === 'block') {
        modal.style.display = 'none';
    } else {
        if (hiddenData && content.innerHTML === '') {
            try {
                const json = JSON.parse(hiddenData.value);
                content.innerHTML = '<pre>' + JSON.stringify(json, null, 2) + '</pre>';
            } catch (e) {
                content.innerHTML = '<pre>' + hiddenData.value + '</pre>';
            }
        }
        modal.style.display = 'block';
    }
}

// 2. Toast Notification System
function showToast(message) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'cyber-toast';
    toast.innerText = message;

    // Basic styling for toast if CSS doesn't cover it fully
    toast.style.background = 'rgba(0, 0, 0, 0.8)';
    toast.style.border = '1px solid #00f3ff';
    toast.style.color = '#00f3ff';
    toast.style.padding = '10px 20px';
    toast.style.marginBottom = '10px';
    toast.style.fontFamily = "'JetBrains Mono', monospace";
    toast.style.fontSize = '0.8rem';
    toast.style.boxShadow = '0 0 10px rgba(0, 243, 255, 0.2)';

    container.appendChild(toast);

    // Remove after 3 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 500);
    }, 3000);
}

// 3. Theme Toggler (Simulated)
function toggleTheme() {
    document.body.classList.toggle('tier-high');
    document.body.classList.toggle('tier-low');
    const mode = document.body.classList.contains('tier-high') ? 'HIGH TIER' : 'LOW TIER';
    showToast(`VIEW MODE: ${mode}`);
}

// 4. System 2 Verification Simulation
function runVerification() {
    const overlay = document.getElementById('system2-overlay');
    const content = document.getElementById('sys2-content');

    overlay.style.display = 'flex';
    content.innerHTML = '';

    const steps = [
        "INITIATING SYSTEM 2 PROTOCOL...",
        "VERIFYING CRYPTOGRAPHIC SIGNATURE...",
        "CHECKING PROVENANCE HASH...",
        "ANALYZING BIAS VECTORS...",
        "VALIDATING DATA SOURCES...",
        "VERIFICATION COMPLETE: INTEGRITY 100%"
    ];

    let i = 0;
    function typeStep() {
        if (i < steps.length) {
            const div = document.createElement('div');
            div.className = 'sys2-line';
            div.innerText = "> " + steps[i];
            div.style.color = i === steps.length - 1 ? '#00ff00' : '#00f3ff';
            div.style.fontFamily = "'JetBrains Mono', monospace";
            div.style.marginBottom = '5px';
            content.appendChild(div);
            i++;
            setTimeout(typeStep, 600);
        } else {
            setTimeout(() => {
                overlay.style.display = 'none';
                showToast("SYSTEM VERIFIED: AUTHORIZED");
            }, 1500);
        }
    }

    typeStep();
}
