document.addEventListener('DOMContentLoaded', () => {
    // --- State ---
    let artifacts = [];
    let currentView = 'grid'; // 'grid' or 'list'
    let draggedItem = null;

    // --- DOM Elements ---
    const gridContainer = document.getElementById('gridContainer');
    const listContainer = document.getElementById('listContainer');
    const searchInput = document.getElementById('searchInput');
    const typeFilter = document.getElementById('typeFilter');
    const viewToggleBtn = document.getElementById('viewToggleBtn');
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('btnUpload');

    // Modal Elements
    const editorModal = document.getElementById('editorModal');
    const modalTitle = document.getElementById('modalTitle');
    const codeEditor = document.getElementById('codeEditor');
    const closeModalBtn = document.getElementById('closeModalBtn');
    const saveFileBtn = document.getElementById('saveFileBtn');

    // --- Initialization ---
    fetchData();

    // --- Event Listeners ---
    searchInput.addEventListener('input', renderArtifacts);
    typeFilter.addEventListener('change', renderArtifacts);

    viewToggleBtn.addEventListener('click', () => {
        currentView = currentView === 'grid' ? 'list' : 'grid';
        viewToggleBtn.innerHTML = currentView === 'grid' ? '<i class="fas fa-list"></i> LIST VIEW' : '<i class="fas fa-th"></i> GRID VIEW';
        renderArtifacts();
    });

    // Upload Simulation
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileUpload);

    // Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFileUpload({ target: { files: e.dataTransfer.files } });
    });

    // Modal
    closeModalBtn.addEventListener('click', closeEditor);
    saveFileBtn.addEventListener('click', () => {
        showToast('FILE SAVED TO SECURE STORAGE', 'success');
        closeEditor();
    });

    // --- Functions ---

    async function fetchData() {
        try {
            const response = await fetch('data/market_mayhem_index.json');
            if (!response.ok) throw new Error("Failed to load index");
            artifacts = await response.json();

            // Add ID for handling deletion/editing
            artifacts = artifacts.map((item, index) => ({...item, id: index}));

            renderArtifacts();
            updateMetrics();
        } catch (error) {
            console.error(error);
            gridContainer.innerHTML = `<div class="mono text-red">ERROR LOADING DATA REPOSITORY. CHECK CONSOLE.</div>`;
        }
    }

    function renderArtifacts() {
        const query = searchInput.value.toLowerCase();
        const type = typeFilter.value;

        const filtered = artifacts.filter(item => {
            const matchesSearch = (item.title && item.title.toLowerCase().includes(query)) ||
                                  (item.summary && item.summary.toLowerCase().includes(query)) ||
                                  (item.filename && item.filename.toLowerCase().includes(query));
            const matchesType = type === 'ALL' || item.type === type;
            return matchesSearch && matchesType;
        });

        // Clear containers
        gridContainer.innerHTML = '';
        listContainer.innerHTML = '';

        if (filtered.length === 0) {
            gridContainer.innerHTML = '<div class="mono text-secondary text-center col-span-full">NO ARTIFACTS FOUND</div>';
            listContainer.innerHTML = '<div class="mono text-secondary text-center">NO ARTIFACTS FOUND</div>';
            return;
        }

        if (currentView === 'grid') {
            gridContainer.classList.remove('hidden');
            listContainer.classList.add('hidden');
            filtered.forEach(item => {
                const card = createCard(item);
                gridContainer.appendChild(card);
            });
        } else {
            gridContainer.classList.add('hidden');
            listContainer.classList.remove('hidden');
            filtered.forEach(item => {
                const row = createListRow(item);
                listContainer.appendChild(row);
            });
        }

        document.getElementById('metricCount').textContent = filtered.length;
    }

    function createCard(item) {
        const el = document.createElement('div');
        el.className = 'file-card';

        let icon = 'fa-file-alt';
        if (item.type === 'DAILY_BRIEFING') icon = 'fa-newspaper';
        if (item.type === 'MARKET_PULSE') icon = 'fa-heartbeat';
        if (item.type === 'STRATEGY') icon = 'fa-chess';
        if (item.type === 'DEEP_DIVE') icon = 'fa-search';

        // Truncate title
        const displayTitle = item.title.length > 30 ? item.title.substring(0, 30) + '...' : item.title;

        el.innerHTML = `
            <div class="flex justify-between items-start">
                <i class="fas ${icon} file-icon"></i>
                <span class="file-type-badge">${item.type}</span>
            </div>
            <div class="file-name" title="${item.title}">${displayTitle}</div>
            <div class="file-meta">
                <span>${item.date}</span>
                <span>${(Math.random() * 50 + 10).toFixed(1)} KB</span>
            </div>
            <div class="mt-2 flex gap-2 justify-end opacity-0 group-hover:opacity-100 transition-opacity">
                <button class="cyber-btn" style="padding: 2px 5px; font-size: 0.7rem;" onclick="editArtifact(${item.id})"><i class="fas fa-edit"></i></button>
                <button class="cyber-btn" style="padding: 2px 5px; font-size: 0.7rem; border-color: var(--accent-red); color: var(--accent-red);" onclick="deleteArtifact(${item.id})"><i class="fas fa-trash"></i></button>
            </div>
        `;

        // Add hover effect via JS for the buttons since they are dynamically added
        el.addEventListener('mouseenter', () => {
            const btns = el.querySelector('div:last-child');
            btns.classList.remove('opacity-0');
        });
        el.addEventListener('mouseleave', () => {
            const btns = el.querySelector('div:last-child');
            btns.classList.add('opacity-0');
        });

        // Click to edit
        el.addEventListener('click', (e) => {
            if (!e.target.closest('button')) {
                editArtifact(item.id);
            }
        });

        return el;
    }

    function createListRow(item) {
        const el = document.createElement('div');
        el.className = 'list-item';

        let icon = 'fa-file-alt';
        if (item.type === 'DAILY_BRIEFING') icon = 'fa-newspaper';

        el.innerHTML = `
            <div class="text-center"><i class="fas ${icon} text-secondary"></i></div>
            <div class="mono font-bold truncate" title="${item.title}">${item.title}</div>
            <div class="mono text-xs text-secondary">${item.type}</div>
            <div class="mono text-xs text-secondary">${item.date}</div>
            <div class="flex gap-2 justify-end">
                <button class="text-cyan hover:text-white" onclick="editArtifact(${item.id})"><i class="fas fa-edit"></i></button>
                <button class="text-red hover:text-white" onclick="deleteArtifact(${item.id})"><i class="fas fa-trash"></i></button>
            </div>
        `;
        return el;
    }

    function handleFileUpload(e) {
        const files = e.target.files;
        if (files.length > 0) {
            const file = files[0];
            showToast(`UPLOADING: ${file.name}`, 'info');

            setTimeout(() => {
                const newItem = {
                    id: artifacts.length,
                    title: file.name.replace('.html', '').replace('_', ' '),
                    date: new Date().toISOString().split('T')[0],
                    type: 'UPLOAD',
                    summary: 'Uploaded artifact.',
                    full_body: 'Content uploaded manually.',
                    filename: file.name
                };
                artifacts.unshift(newItem);
                renderArtifacts();
                updateMetrics();
                showToast('UPLOAD COMPLETE', 'success');
            }, 1500);
        }
    }

    window.editArtifact = function(id) {
        const item = artifacts.find(a => a.id === id);
        if (!item) return;

        modalTitle.textContent = `EDITING: ${item.filename}`;

        // Pretty print content if JSON, or just show text
        let content = item.full_body || item.summary;
        // Strip HTML for the editor view just to simulate code
        codeEditor.value = content;

        editorModal.style.display = 'flex';
    };

    window.deleteArtifact = function(id) {
        if (confirm('CONFIRM DELETION? THIS ACTION CANNOT BE UNDONE.')) {
            artifacts = artifacts.filter(a => a.id !== id);
            renderArtifacts();
            updateMetrics();
            showToast('ARTIFACT DELETED', 'success');
        }
    };

    function closeEditor() {
        editorModal.style.display = 'none';
    }

    function updateMetrics() {
        const total = artifacts.length;
        const sentimentAvg = Math.round(artifacts.reduce((acc, curr) => acc + (curr.sentiment_score || 50), 0) / total);

        document.getElementById('metricTotal').textContent = total;
        document.getElementById('metricSentiment').textContent = sentimentAvg + '%';

        // Just for show
        document.getElementById('metricStorage').textContent = (total * 0.05).toFixed(2) + ' GB';
    }

    function showToast(msg, type = 'info') {
        const toast = document.createElement('div');
        toast.className = 'fixed bottom-5 right-5 p-4 rounded shadow-lg z-50 mono text-sm';

        if (type === 'success') {
            toast.style.background = 'rgba(16, 185, 129, 0.9)';
            toast.style.color = 'white';
            toast.innerHTML = `<i class="fas fa-check-circle mr-2"></i> ${msg}`;
        } else {
            toast.style.background = 'rgba(0, 243, 255, 0.9)';
            toast.style.color = 'black';
            toast.innerHTML = `<i class="fas fa-info-circle mr-2"></i> ${msg}`;
        }

        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 3000);
    }
});
