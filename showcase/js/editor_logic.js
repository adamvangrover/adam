/**
 * EDITOR STUDIO LOGIC
 * Handles the creation and editing of intelligence artifacts.
 */

const EditorManager = {
    init() {
        console.log("[EditorManager] Initializing Creator Studio...");
        this.bindEvents();
        this.setupToolbar();
        this.loadDraft();
    },

    bindEvents() {
        // Save Button
        document.getElementById('btnSave')?.addEventListener('click', () => this.saveDocument());

        // Clear/New Button
        document.getElementById('btnNew')?.addEventListener('click', () => {
            if(confirm("Discard current draft?")) {
                this.clearForm();
            }
        });

        // Auto-save draft on input
        document.getElementById('editorContent')?.addEventListener('input', () => this.saveDraftLocally());
    },

    setupToolbar() {
        const buttons = document.querySelectorAll('.toolbar-btn');
        buttons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const command = btn.dataset.command;
                if (command === 'createLink') {
                    const url = prompt("Enter URL:", "https://");
                    if (url) document.execCommand(command, false, url);
                } else if (command === 'formatBlock') {
                    document.execCommand(command, false, btn.dataset.value);
                } else {
                    document.execCommand(command, false, null);
                }

                // Keep focus in editor
                document.getElementById('editorContent').focus();
            });
        });
    },

    saveDocument() {
        const title = document.getElementById('inputTitle').value;
        const type = document.getElementById('inputType').value;
        const sentiment = document.getElementById('inputSentiment').value;
        const content = document.getElementById('editorContent').innerHTML;
        const summary = document.getElementById('inputSummary').value;

        if (!title) {
            alert("Error: Title is required.");
            return;
        }

        const artifact = {
            title,
            type,
            sentiment_score: parseInt(sentiment),
            summary,
            full_body: content,
            date: new Date().toISOString().split('T')[0],
            is_draft: false,
            provenance_hash: this.generateHash()
        };

        console.log("[EditorManager] Saving Artifact:", artifact);

        // Simulate Network Request
        const saveBtn = document.getElementById('btnSave');
        const originalText = saveBtn.innerHTML;
        saveBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> ENCRYPTING...';

        setTimeout(() => {
            // Save to mock storage (localStorage for demo)
            const library = JSON.parse(localStorage.getItem('adam_mock_library') || '[]');
            library.unshift(artifact);
            localStorage.setItem('adam_mock_library', JSON.stringify(library));

            saveBtn.innerHTML = '<i class="fas fa-check"></i> SAVED';
            saveBtn.classList.add('text-green-400', 'border-green-400');

            setTimeout(() => {
                saveBtn.innerHTML = originalText;
                saveBtn.classList.remove('text-green-400', 'border-green-400');
                alert(`Artifact "${title}" successfully committed to the repository.`);
            }, 1500);
        }, 1000);
    },

    saveDraftLocally() {
        // Debounced local storage save
        if (this.saveTimeout) clearTimeout(this.saveTimeout);
        this.saveTimeout = setTimeout(() => {
            const draft = {
                title: document.getElementById('inputTitle').value,
                content: document.getElementById('editorContent').innerHTML
            };
            localStorage.setItem('adam_editor_draft', JSON.stringify(draft));
            console.log("[EditorManager] Draft auto-saved.");
        }, 1000);
    },

    loadDraft() {
        const draftStr = localStorage.getItem('adam_editor_draft');
        if (draftStr) {
            try {
                const draft = JSON.parse(draftStr);
                if (draft.title) document.getElementById('inputTitle').value = draft.title;
                if (draft.content) document.getElementById('editorContent').innerHTML = draft.content;
            } catch (e) {
                console.error("Failed to load draft", e);
            }
        }
    },

    clearForm() {
        document.getElementById('inputTitle').value = '';
        document.getElementById('inputSummary').value = '';
        document.getElementById('editorContent').innerHTML = '';
        localStorage.removeItem('adam_editor_draft');
    },

    generateHash() {
        return Array.from({length: 64}, () => Math.floor(Math.random() * 16).toString(16)).join('');
    }
};

document.addEventListener('DOMContentLoaded', () => {
    EditorManager.init();
});
