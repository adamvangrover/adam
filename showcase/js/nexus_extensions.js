// Nexus Extensions - Adds Provenance and Templates to Office Nexus

(function() {
    const EXTENSION_CONFIG = {
        apps: [
            {
                name: 'ProvenanceExplorer',
                icon: 'https://img.icons8.com/color/48/000000/hierarchy.png',
                title: 'Data Lineage',
                width: 900,
                height: 600
            },
            {
                name: 'TemplateStudio',
                icon: 'https://img.icons8.com/color/48/000000/template.png',
                title: 'Template Studio',
                width: 800,
                height: 500
            }
        ]
    };

    function initExtensions() {
        console.log("Nexus Extensions: Initializing...");

        if (!window.officeOS) {
            console.warn("Nexus Extensions: OfficeOS not found. Retrying in 500ms...");
            setTimeout(initExtensions, 500);
            return;
        }

        patchAppRegistry();
        injectIcons();
        console.log("Nexus Extensions: Loaded.");
    }

    function patchAppRegistry() {
        const originalLaunch = window.officeOS.appRegistry.launch.bind(window.officeOS.appRegistry);

        window.officeOS.appRegistry.launch = function(appName, args) {
            if (appName === 'ProvenanceExplorer') {
                launchProvenanceExplorer(args);
            } else if (appName === 'TemplateStudio') {
                launchTemplateStudio(args);
            } else {
                originalLaunch(appName, args);
            }
        };
    }

    function injectIcons() {
        // Desktop
        const desktop = document.getElementById('desktop');
        if (desktop) {
            EXTENSION_CONFIG.apps.forEach(app => {
                const el = document.createElement('div');
                el.className = 'desktop-icon';
                el.innerHTML = `<img src="${app.icon}"><span>${app.title}</span>`;
                el.addEventListener('dblclick', () => window.officeOS.appRegistry.launch(app.name));
                el.addEventListener('click', (e) => {
                    e.stopPropagation();
                    document.querySelectorAll('.desktop-icon').forEach(i => i.classList.remove('selected'));
                    el.classList.add('selected');
                });
                desktop.appendChild(el);
            });
        }

        // Start Menu
        const startMenuGrid = document.querySelector('.start-menu-grid');
        if (startMenuGrid) {
            EXTENSION_CONFIG.apps.forEach(app => {
                const el = document.createElement('div');
                el.className = 'start-menu-item';
                el.innerHTML = `<img src="${app.icon}"><span>${app.title}</span>`;
                el.addEventListener('click', () => {
                    window.officeOS.appRegistry.launch(app.name);
                    window.officeOS.toggleStartMenu();
                });
                startMenuGrid.appendChild(el);
            });
        }
    }

    // --- App: Provenance Explorer ---

    function launchProvenanceExplorer(args) {
        const winId = window.officeOS.windowManager.createWindow({
            title: 'Data Lineage Explorer',
            icon: EXTENSION_CONFIG.apps[0].icon,
            width: EXTENSION_CONFIG.apps[0].width,
            height: EXTENSION_CONFIG.apps[0].height,
            app: 'ProvenanceExplorer'
        });

        const container = document.createElement('div');
        container.style.display = 'flex';
        container.style.height = '100%';
        container.style.backgroundColor = '#f0f0f0';

        // Sidebar
        const sidebar = document.createElement('div');
        sidebar.style.width = '250px';
        sidebar.style.backgroundColor = '#fff';
        sidebar.style.borderRight = '1px solid #ddd';
        sidebar.style.overflowY = 'auto';
        sidebar.innerHTML = '<div style="padding:10px; font-weight:bold; border-bottom:1px solid #eee;">Artifacts</div><ul id="prov-list-' + winId + '" style="list-style:none; padding:0; margin:0;">Loading...</ul>';

        // Main View
        const main = document.createElement('div');
        main.id = 'prov-main-' + winId;
        main.style.flexGrow = '1';
        main.style.padding = '20px';
        main.style.overflow = 'auto';
        main.innerHTML = '<div style="color:#888; text-align:center; margin-top:100px;">Select an artifact to view lineage</div>';

        container.appendChild(sidebar);
        container.appendChild(main);
        window.officeOS.windowManager.setWindowContent(winId, container);

        // Populate List
        window.provenanceSystem.onReady(() => {
            const list = document.getElementById('prov-list-' + winId);
            list.innerHTML = '';
            const artifacts = window.provenanceSystem.getAllArtifacts();

            artifacts.forEach(art => {
                const li = document.createElement('li');
                li.style.padding = '8px 10px';
                li.style.cursor = 'pointer';
                li.style.borderBottom = '1px solid #f5f5f5';
                li.innerText = art.name;
                li.addEventListener('mouseenter', () => li.style.backgroundColor = '#f0f8ff');
                li.addEventListener('mouseleave', () => li.style.backgroundColor = 'transparent');
                li.addEventListener('click', () => renderLineageGraph(winId, art.id));
                list.appendChild(li);
            });
        });
    }

    function renderLineageGraph(winId, artifactId) {
        const main = document.getElementById('prov-main-' + winId);
        const lineage = window.provenanceSystem.getArtifactLineage(artifactId);

        if (!lineage) {
            main.innerHTML = 'Error: Lineage not found.';
            return;
        }

        // Simple Graph Visualization
        // Group nodes by type
        const sources = lineage.nodes.filter(n => n.type === 'Source');
        const agents = lineage.nodes.filter(n => n.type === 'Agent');
        const artifacts = lineage.nodes.filter(n => n.type === 'Artifact');

        let html = `<h3>Lineage: ${artifacts[0]?.label || artifactId}</h3>`;
        html += `<div style="display:flex; justify-content:space-around; align-items:flex-start; margin-top:20px;">`;

        // Column 1: Sources
        html += `<div style="flex:1; text-align:center;"><h4>Sources</h4>`;
        sources.forEach(n => html += renderNode(n, '#e8f5e9', '#2e7d32'));
        if(sources.length === 0) html += `<div style="color:#ccc;">(None)</div>`;
        html += `</div>`;

        // Column 2: Agents
        html += `<div style="flex:1; text-align:center;"><h4>Agents</h4>`;
        agents.forEach(n => html += renderNode(n, '#e3f2fd', '#1565c0'));
        html += `</div>`;

        // Column 3: Artifacts
        html += `<div style="flex:1; text-align:center;"><h4>Artifact</h4>`;
        artifacts.forEach(n => html += renderNode(n, '#fff3e0', '#ef6c00'));
        html += `</div>`;

        html += `</div>`;

        // Edges List (Visualizing edges as list for simplicity)
        html += `<div style="margin-top:30px; border-top:1px solid #ddd; padding-top:10px;">
            <h4>Process Flow</h4>
            <ul style="list-style:none; padding:0;">`;

        lineage.edges.forEach(e => {
            const sourceNode = lineage.nodes.find(n => n.id === e.source);
            const targetNode = lineage.nodes.find(n => n.id === e.target);
            html += `<li style="margin-bottom:5px;">
                <span style="font-weight:bold;">${sourceNode?.label || e.source}</span>
                --[ ${e.label} ]-->
                <span style="font-weight:bold;">${targetNode?.label || e.target}</span>
            </li>`;
        });
        html += `</ul></div>`;

        main.innerHTML = html;
    }

    function renderNode(node, bg, border) {
        return `
            <div style="
                background: ${bg};
                border: 1px solid ${border};
                border-radius: 5px;
                padding: 10px;
                margin: 10px auto;
                width: 80%;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <strong>${node.label}</strong><br>
                <span style="font-size:10px; color:#666;">${node.type}</span>
            </div>
        `;
    }

    // --- App: Template Studio ---

    function launchTemplateStudio(args) {
        const winId = window.officeOS.windowManager.createWindow({
            title: 'Template Studio',
            icon: EXTENSION_CONFIG.apps[1].icon,
            width: EXTENSION_CONFIG.apps[1].width,
            height: EXTENSION_CONFIG.apps[1].height,
            app: 'TemplateStudio'
        });

        const container = document.createElement('div');
        container.style.padding = '20px';
        container.innerHTML = `
            <h3>Create New Artifact</h3>
            <div style="margin-bottom:20px;">
                <button class="cyber-btn" onclick="window.loadTemplateForm('${winId}', 'credit_memo')">Credit Memo</button>
                <button class="cyber-btn" onclick="window.loadTemplateForm('${winId}', 'equity_report')">Equity Report</button>
            </div>
            <div id="template-form-${winId}" style="border:1px solid #ddd; padding:15px; background:#fff; min-height:300px;">
                Select a template above.
            </div>
        `;
        window.officeOS.windowManager.setWindowContent(winId, container);

        // Expose helper to window for onclick (hacky but effective for this context)
        window.loadTemplateForm = async (wId, type) => {
            const formContainer = document.getElementById(`template-form-${wId}`);
            formContainer.innerHTML = 'Loading template...';

            const template = await window.provenanceSystem.loadTemplate(type);
            if (!template) {
                formContainer.innerHTML = 'Error loading template.';
                return;
            }

            let html = `<h4>New ${type.replace('_', ' ').toUpperCase()}</h4>`;
            html += `<form id="form-${wId}">`;

            // Simple recursive form builder
            for (const [key, value] of Object.entries(template)) {
                if (typeof value === 'string' || typeof value === 'number') {
                    html += `
                        <div style="margin-bottom:10px;">
                            <label style="display:block; font-size:12px; color:#666;">${key}</label>
                            <input type="text" name="${key}" value="${value}" style="width:100%; padding:5px; border:1px solid #ccc;">
                        </div>
                    `;
                } else if (Array.isArray(value)) {
                     html += `
                        <div style="margin-bottom:10px;">
                            <label style="display:block; font-size:12px; color:#666;">${key} (List)</label>
                            <textarea name="${key}" style="width:100%; height:60px; padding:5px; border:1px solid #ccc;">${value.join('\n')}</textarea>
                        </div>
                    `;
                }
            }

            html += `<button type="button" class="cyber-btn" style="margin-top:10px; background:#0078d7; color:white;" onclick="alert('Artifact Created! (Mock)')">Create Artifact</button>`;
            html += `</form>`;

            formContainer.innerHTML = html;
        };
    }

    // Start
    initExtensions();

})();
