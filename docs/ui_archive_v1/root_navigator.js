document.addEventListener('DOMContentLoaded', () => {
    const fileTreeContainer = document.getElementById('file-tree');
    const contentViewer = document.getElementById('content-viewer');

    // These should be dynamic in a real application
    const GITHUB_USER = 'adamvangrover';
    const GITHUB_REPO = 'adam';
    const GITHUB_BRANCH = 'main';

    async function fetchRepoTree() {
        // Using the recursive tree API to get all files at once
        const url = `https://api.github.com/repos/${GITHUB_USER}/${GITHUB_REPO}/git/trees/${GITHUB_BRANCH}?recursive=1`;
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`GitHub API request failed: ${response.status}`);
            }
            const data = await response.json();
            return data.tree;
        } catch (error) {
            console.error('Error fetching repository tree:', error);
            fileTreeContainer.innerHTML = 'Error loading repository tree.';
            return [];
        }
    }

    function buildFileTree(files) {
        const fileMap = {};
        const root = [];

        files.forEach(file => {
            const pathParts = file.path.split('/');
            let currentLevel = root;
            let currentMap = fileMap;

            pathParts.forEach((part, index) => {
                let node = currentMap[part];
                if (!node) {
                    node = {
                        name: part,
                        path: file.path,
                        type: file.type,
                        children: file.type === 'tree' ? [] : undefined,
                        childrenMap: file.type === 'tree' ? {} : undefined,
                    };
                    currentMap[part] = node;
                    currentLevel.push(node);
                }

                if (node.type === 'tree') {
                    currentLevel = node.children;
                    currentMap = node.childrenMap;
                }
            });
        });

        // The above logic is a bit flawed for nested paths. Let's simplify.
        // We'll create a nested object structure representing the file tree.
        const tree = {};
        files.forEach(file => {
            file.path.split('/').reduce((acc, part, index, arr) => {
                if (!acc[part]) {
                    acc[part] = {
                        _path: file.path,
                        _type: file.type,
                    };
                    if(file.type === 'tree') {
                        acc[part]._children = {};
                    }
                }
                return index === arr.length -1 ? acc[part] : (acc[part]._children || acc[part]);
            }, tree);
        });

        return tree;
    }

    function renderTree(tree, container) {
        const ul = document.createElement('ul');
        for (const key in tree) {
            const node = tree[key];
            const li = document.createElement('li');
            li.textContent = key;
            if (node._type === 'tree') {
                li.classList.add('directory');
                const childrenContainer = document.createElement('div');
                childrenContainer.style.display = 'none'; // Initially collapsed
                renderTree(node._children, childrenContainer);
                li.appendChild(childrenContainer);
                li.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const isVisible = childrenContainer.style.display === 'block';
                    childrenContainer.style.display = isVisible ? 'none' : 'block';
                });
            } else {
                li.classList.add('file');
                li.addEventListener('click', (e) => {
                    e.stopPropagation();
                    loadFileContent(node._path);
                });
            }
            ul.appendChild(li);
        }
        container.appendChild(ul);
    }

    async function loadFileContent(path) {
        contentViewer.innerHTML = `Loading ${path}...`;
        const url = `https://raw.githubusercontent.com/${GITHUB_USER}/${GITHUB_REPO}/${GITHUB_BRANCH}/${path}`;
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`Failed to fetch file: ${response.status}`);
            }
            const content = await response.text();
            renderFileContent(path, content);
        } catch (error) {
            console.error('Error loading file content:', error);
            contentViewer.innerHTML = `Error loading file: ${path}`;
        }
    }

    function renderFileContent(path, content) {
        if (path.endsWith('.md')) {
            contentViewer.innerHTML = marked.parse(content);
        } else if (path.endsWith('.html')) {
            // Use an iframe to render HTML to avoid style conflicts and for security
            const iframe = document.createElement('iframe');
            iframe.srcdoc = content;
            iframe.style.width = '100%';
            iframe.style.height = '100%';
            iframe.style.border = 'none';
            contentViewer.innerHTML = '';
            contentViewer.appendChild(iframe);
        } else {
            const pre = document.createElement('pre');
            const code = document.createElement('code');
            code.textContent = content;
            pre.appendChild(code);
            contentViewer.innerHTML = '';
            contentViewer.appendChild(pre);
        }
    }

    async function init() {
        const files = await fetchRepoTree();
        const fileTree = buildFileTree(files);
        fileTreeContainer.innerHTML = ''; // Clear loading message
        renderTree(fileTree, fileTreeContainer);
    }

    init();
});
