// Repository Management Logic

let REPO_DATA = [];
let CURRENT_VIEW = 'grid'; // 'grid' or 'list'
let CURRENT_FILTER = 'ALL';

document.addEventListener('DOMContentLoaded', () => {
    initRepository();
});

async function initRepository() {
    try {
        const response = await fetch('data/market_mayhem_index.json');
        if (!response.ok) throw new Error("Failed to load index");
        REPO_DATA = await response.json();

        // Add fake ID if missing for easier management
        REPO_DATA.forEach((item, index) => {
            if (!item.id) item.id = `doc_${index}`;
        });

        renderSidebar();
        renderMainView();
        setupDragDrop();
        updateStatus(`System Online. Loaded ${REPO_DATA.length} artifacts.`);
    } catch (e) {
        console.error(e);
        updateStatus("ERROR: CONNECTION FAILED", true);
    }
}

function updateStatus(msg, error = false) {
    const el = document.getElementById('status-text');
    if (el) {
        el.innerText = `> ${msg}`;
        el.style.color = error ? '#cc0000' : '#666';
    }
}

function renderSidebar() {
    const counts = {};
    REPO_DATA.forEach(item => {
        const type = item.type || 'UNKNOWN';
        counts[type] = (counts[type] || 0) + 1;
    });

    const list = document.getElementById('filter-list');
    if (!list) return;

    let html = `
        <div class="filter-item ${CURRENT_FILTER === 'ALL' ? 'active' : ''}" onclick="setFilter('ALL')">
            <i class="fas fa-layer-group"></i> ALL FILES
            <span class="count-badge">${REPO_DATA.length}</span>
        </div>
    `;

    for (const [type, count] of Object.entries(counts)) {
        html += `
            <div class="filter-item ${CURRENT_FILTER === type ? 'active' : ''}" onclick="setFilter('${type}')">
                <i class="fas fa-folder"></i> ${type.replace('_', ' ')}
                <span class="count-badge">${count}</span>
            </div>
        `;
    }

    list.innerHTML = html;
}

function setFilter(filter) {
    CURRENT_FILTER = filter;
    renderSidebar(); // Update active state
    renderMainView();
}

function setView(view) {
    CURRENT_VIEW = view;
    renderMainView();
}

function renderMainView() {
    const container = document.getElementById('repo-content');
    if (!container) return;

    const filtered = REPO_DATA.filter(item => CURRENT_FILTER === 'ALL' || item.type === CURRENT_FILTER);

    // Sort by date desc
    filtered.sort((a, b) => new Date(b.date) - new Date(a.date));

    let html = '';

    if (CURRENT_VIEW === 'grid') {
        html = '<div class="file-grid">';
        html += `
            <div class="drop-zone" id="drop-zone">
                <i class="fas fa-cloud-upload-alt"></i>
                <div>UPLOAD ARTIFACT</div>
                <div style="font-size:0.7rem; color:#444;">Drag & Drop or Click</div>
                <input type="file" id="file-input" style="display:none;" multiple>
            </div>
        `;

        filtered.forEach((item, idx) => {
            html += `
                <div class="file-card animate-entry type-${item.type}" style="animation-delay: ${Math.min(idx * 0.05, 1)}s" ondblclick="openEditor('${item.id}')">
                    <div class="file-actions">
                        <div class="action-btn" onclick="downloadFile('${item.filename}')" title="Download"><i class="fas fa-download"></i></div>
                        <div class="action-btn" onclick="openEditor('${item.id}')" title="Edit"><i class="fas fa-edit"></i></div>
                        <div class="action-btn delete" onclick="deleteFile('${item.id}')" title="Delete"><i class="fas fa-trash"></i></div>
                    </div>
                    <div class="file-icon"><i class="fas ${getIconForType(item.type)}"></i></div>
                    <div class="file-name" title="${item.title}">${item.title}</div>
                    <div class="file-meta">${item.date}</div>
                    <div class="file-meta" style="color:#444;">${item.type}</div>
                </div>
            `;
        });
        html += '</div>';
    } else {
        html = '<div class="file-list">';
        // Headers
        html += `
            <div class="file-row" style="background:rgba(0,0,0,0.5); color:#888; border-bottom:1px solid #444;">
                <div></div>
                <div>TITLE</div>
                <div>DATE</div>
                <div>TYPE</div>
                <div>ACTIONS</div>
            </div>
        `;
        filtered.forEach((item, idx) => {
            html += `
                <div class="file-row animate-entry" style="animation-delay: ${Math.min(idx * 0.02, 1)}s">
                    <div style="color:#888;"><i class="fas ${getIconForType(item.type)}"></i></div>
                    <div style="font-weight:bold;">${item.title}</div>
                    <div class="mono" style="font-size:0.8rem; color:#666;">${item.date}</div>
                    <div><span class="cyber-badge" style="font-size:0.6rem;">${item.type}</span></div>
                    <div style="display:flex; gap:5px;">
                        <div class="action-btn" onclick="openEditor('${item.id}')"><i class="fas fa-edit"></i></div>
                        <div class="action-btn" onclick="downloadFile('${item.filename}')"><i class="fas fa-download"></i></div>
                    </div>
                </div>
            `;
        });
        html += '</div>';
    }

    container.innerHTML = html;
    setupDragDrop(); // Re-bind events
}

function getIconForType(type) {
    if (type === 'NEWSLETTER') return 'fa-newspaper';
    if (type === 'MARKET_PULSE') return 'fa-heartbeat';
    if (type === 'DEEP_DIVE') return 'fa-microscope';
    if (type === 'STRATEGY') return 'fa-chess';
    return 'fa-file-alt';
}

/* --- ACTIONS --- */

function downloadFile(filename) {
    const link = document.createElement('a');
    link.href = filename;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    updateStatus(`Downloaded ${filename}`);
}

function deleteFile(id) {
    if (!confirm('CONFIRM DELETION? This simulated action removes the item from view.')) return;
    REPO_DATA = REPO_DATA.filter(i => i.id !== id);
    renderSidebar();
    renderMainView();
    updateStatus(`Deleted artifact ${id}`);
}

/* --- UPLOAD SIMULATION --- */

function setupDragDrop() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    if (!dropZone) return;

    dropZone.onclick = () => fileInput.click();

    fileInput.onchange = (e) => handleFiles(e.target.files);

    dropZone.ondragover = (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    };
    dropZone.ondragleave = () => dropZone.classList.remove('dragover');
    dropZone.ondrop = (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    };
}

function handleFiles(files) {
    if (files.length === 0) return;

    updateStatus(`Uploading ${files.length} items...`);

    // Simulate upload delay
    setTimeout(() => {
        Array.from(files).forEach(file => {
            const newItem = {
                id: `new_${Date.now()}_${Math.random()}`,
                title: file.name.replace('.pdf', '').replace('.html', ''),
                date: new Date().toISOString().split('T')[0],
                type: 'UPLOADED',
                summary: 'User uploaded artifact.',
                filename: '#'
            };
            REPO_DATA.unshift(newItem);
        });
        renderSidebar();
        renderMainView();
        updateStatus("Upload Complete. Malware Scan Negative.");
    }, 1500);
}

/* --- EDITOR SIMULATION --- */

let currentEditId = null;

function openEditor(id) {
    const item = REPO_DATA.find(i => i.id === id);
    if (!item) return;

    currentEditId = id;
    const modal = document.getElementById('editor-modal');
    const title = document.getElementById('editor-title');
    const content = document.getElementById('editor-content');

    title.innerText = `EDITING: ${item.filename || 'UNTITLED'}`;

    // Simulate content
    if (item.full_body) {
        content.value = item.full_body;
    } else {
        content.value = JSON.stringify(item, null, 2);
    }

    modal.classList.add('active');
}

function closeEditor() {
    document.getElementById('editor-modal').classList.remove('active');
    currentEditId = null;
}

function saveEditor() {
    if (!currentEditId) return;
    const content = document.getElementById('editor-content').value;
    const item = REPO_DATA.find(i => i.id === currentEditId);

    if (item) {
        item.full_body = content; // In memory update
        updateStatus(`Saved changes to ${item.title}`);
        closeEditor();
    }
}

// Global search hook
function applySearch(term) {
    // This could be hooked up to the nav bar search
    term = term.toLowerCase();
    const filtered = REPO_DATA.filter(item => item.title.toLowerCase().includes(term));
    // ... logic to render specific subset ...
    // For now, simple filter
}
