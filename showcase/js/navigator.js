document.addEventListener('DOMContentLoaded', async () => {
    const fileTreeContainer = document.getElementById('file-tree');
    const codeViewer = document.getElementById('code-viewer');
    const filePathLabel = document.getElementById('file-path');

    await window.dataManager.init();
    const files = window.dataManager.getFiles();

    // Build Tree Structure
    const tree = {};
    files.forEach(file => {
        const parts = file.path.split('/');
        let current = tree;
        parts.forEach((part, i) => {
            if (!current[part]) {
                current[part] = { _name: part, _path: file.path, _children: {} };
            }
            if (i === parts.length - 1) {
                current[part]._type = 'file';
            } else {
                current[part]._type = 'dir';
                current = current[part]._children;
            }
        });
    });

    function renderTree(node, container) {
        const ul = document.createElement('ul');
        ul.className = 'pl-4 border-l border-slate-700 ml-1';
        if (container === fileTreeContainer) ul.className = ''; // Root level

        const entries = Object.entries(node).sort((a, b) => {
            // Dirs first, then files
            const aIsDir = a[1]._type === 'dir';
            const bIsDir = b[1]._type === 'dir';
            if (aIsDir && !bIsDir) return -1;
            if (!aIsDir && bIsDir) return 1;
            return a[0].localeCompare(b[0]);
        });

        entries.forEach(([key, val]) => {
            const li = document.createElement('li');
            li.className = 'cursor-pointer select-none';

            const div = document.createElement('div');
            div.className = 'hover:text-blue-300 flex items-center py-0.5';

            const icon = document.createElement('span');
            icon.className = 'mr-2 text-xs text-gray-500';
            icon.textContent = val._type === 'dir' ? '📁' : '📄';

            const span = document.createElement('span');
            span.textContent = key;

            div.appendChild(icon);
            div.appendChild(span);
            li.appendChild(div);

            if (val._type === 'dir') {
                const childrenContainer = document.createElement('div');
                childrenContainer.style.display = 'none';
                renderTree(val._children, childrenContainer);
                li.appendChild(childrenContainer);

                div.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const isHidden = childrenContainer.style.display === 'none';
                    childrenContainer.style.display = isHidden ? 'block' : 'none';
                    icon.textContent = isHidden ? '📂' : '📁';
                });
            } else {
                icon.textContent = '📄';
                div.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    // Highlight selected
                    document.querySelectorAll('.text-blue-400').forEach(el => el.classList.remove('text-blue-400'));
                    span.classList.add('text-blue-400');

                    loadFile(val._path);
                });
            }
            ul.appendChild(li);
        });
        container.appendChild(ul);
    }

    async function loadFile(path) {
        filePathLabel.textContent = path;
        codeViewer.textContent = "Loading...";
        const content = await window.dataManager.getFileContent(path);
        codeViewer.textContent = content;

        // Determine language
        let lang = 'plaintext';
        if (path.endsWith('.py')) lang = 'python';
        else if (path.endsWith('.js')) lang = 'javascript';
        else if (path.endsWith('.html')) lang = 'html';
        else if (path.endsWith('.css')) lang = 'css';
        else if (path.endsWith('.json')) lang = 'json';
        else if (path.endsWith('.md')) lang = 'markdown';
        else if (path.endsWith('.sh')) lang = 'bash';

        codeViewer.className = `language-${lang}`;
        Prism.highlightElement(codeViewer);
    }

    fileTreeContainer.innerHTML = '';
    renderTree(tree, fileTreeContainer);
});
