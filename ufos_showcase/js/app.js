// State
let currentView = 'dashboard';

// Navigation
function showView(viewId, el) {
    document.querySelectorAll('.view').forEach(view => view.classList.remove('active'));
    document.querySelectorAll('.nav-links li').forEach(link => link.classList.remove('active'));

    document.getElementById(`view-${viewId}`).classList.add('active');
    if(el) el.classList.add('active');
}

// Command Palette
function toggleCommandPalette() {
    const el = document.getElementById('command-palette');
    el.classList.toggle('hidden');
    if (!el.classList.contains('hidden')) {
        document.getElementById('palette-input').focus();
    }
}

document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        toggleCommandPalette();
    }
    if (e.key === 'Escape') {
        document.getElementById('command-palette').classList.add('hidden');
    }
});

// Mock Search
document.getElementById('palette-input').addEventListener('input', (e) => {
    const val = e.target.value.toLowerCase();
    const results = document.getElementById('palette-results');
    results.innerHTML = '';

    const mockData = [
        { type: 'Exec', label: '> Analyze AAPL Credit Risk' },
        { type: 'Exec', label: '> Run Backtest: Momentum_V4' },
        { type: 'Nav', label: 'Go to Unified Ledger' },
        { type: 'Nav', label: 'Go to Terminal' },
        { type: 'Search', label: 'Search for "High Yield Bonds"' },
        { type: 'Search', label: 'Search for "FOMC Minutes"' }
    ];

    mockData.filter(item => item.label.toLowerCase().includes(val)).forEach(item => {
        const li = document.createElement('li');
        li.innerHTML = `<span style="color:var(--primary-color)">${item.type}</span> ${item.label}`;
        li.onclick = () => {
            alert(`Executing: ${item.label}`);
            toggleCommandPalette();
        };
        results.appendChild(li);
    });

    if (val === '') {
        mockData.forEach(item => {
             const li = document.createElement('li');
            li.innerHTML = `<span style="color:var(--primary-color)">${item.type}</span> ${item.label}`;
             li.onclick = () => {
                alert(`Executing: ${item.label}`);
                toggleCommandPalette();
            };
            results.appendChild(li);
        });
    }
});

// Initialize Palette with all items
document.getElementById('palette-input').dispatchEvent(new Event('input'));


// Terminal Logic
document.getElementById('terminal-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        const val = e.target.value;
        const output = document.getElementById('terminal-output');

        // Echo
        const line = document.createElement('div');
        line.className = 'line';
        line.innerText = `$ ${val}`;
        output.insertBefore(line, e.target.parentElement);

        // Response (Mock)
        const res = document.createElement('div');
        res.className = 'line';

        if (val === 'help') {
             res.innerText = 'Available commands: run_backtest, analyze_credit, connect_core, whoami';
        } else if (val === 'whoami') {
             res.innerText = 'admin (EACI-v2.0 System Administrator)';
        } else if (val.startsWith('analyze_credit')) {
             res.innerText = '[Orchestrator] Starting Credit Risk Assessment Agent... Done. Report generated.';
        } else {
             res.innerText = `Executing command: ${val}... Done.`;
        }
        output.insertBefore(res, e.target.parentElement);

        e.target.value = '';
        output.scrollTop = output.scrollHeight;
    }
});
