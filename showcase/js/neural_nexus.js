document.addEventListener('DOMContentLoaded', () => {
    const Graph = ForceGraph3D()
      (document.getElementById('graph-container'));

    const elLoader = document.getElementById('loader');
    const elNodeInfo = document.getElementById('node-info');
    const elStatsNodes = document.getElementById('stat-nodes');
    const elStatsLinks = document.getElementById('stat-links');

    let graphData = { nodes: [], links: [] };

    // --- Init ---
    async function init() {
        try {
            // Fetch Metadata (Agents) & Reports (Intelligence)
            const [repoMeta, reportIndex] = await Promise.all([
                fetch('data/repo_metadata.json').then(r => r.json()).catch(() => ({})),
                fetch('data/market_mayhem_index.json').then(r => r.json()).catch(() => [])
            ]);

            processData(repoMeta, reportIndex);
            renderGraph();

            setTimeout(() => {
                elLoader.style.opacity = '0';
                setTimeout(() => elLoader.style.display = 'none', 500);
            }, 1000);

        } catch (e) {
            console.error("Nexus Init Error:", e);
            document.querySelector('.loader-text').textContent = "SYSTEM FAILURE: " + e.message;
            document.querySelector('.loader-text').style.color = "red";
        }
    }

    // --- Data Processing ---
    function processData(meta, reports) {
        const nodes = [];
        const links = [];
        const seen = new Set();

        // 1. Central Hub
        nodes.push({ id: 'ADAM_CORE', group: 'CORE', val: 50, color: '#ffffff', desc: 'Central Nervous System' });
        seen.add('ADAM_CORE');

        // 2. Agents (from repo_metadata)
        if (meta.agents) {
            Object.keys(meta.agents).forEach(agentName => {
                const id = `AGENT:${agentName}`;
                if (!seen.has(id)) {
                    nodes.push({
                        id: id,
                        group: 'AGENT',
                        val: 20,
                        color: '#06b6d4', // Cyan
                        desc: meta.agents[agentName].docstring || 'Autonomous Agent'
                    });
                    seen.add(id);
                    links.push({ source: 'ADAM_CORE', target: id, color: 'rgba(6, 182, 212, 0.2)' });
                }
            });
        }

        // 3. Reports (from market_mayhem_index)
        reports.forEach((rep, idx) => {
            if (idx > 200) return; // Cap for performance if list is huge

            const id = `REPORT:${rep.filename}`;
            if (!seen.has(id)) {
                // Color based on sentiment
                const sent = rep.sentiment_score || 50;
                let color = '#f59e0b'; // Amber (Neutral)
                if (sent > 60) color = '#10b981'; // Green
                if (sent < 40) color = '#ef4444'; // Red

                nodes.push({
                    id: id,
                    group: 'REPORT',
                    val: 10,
                    color: color,
                    desc: rep.title,
                    link: rep.filename
                });
                seen.add(id);

                // Link to Core? Or Link to relevant Agents?
                // For now, link to core to form a starburst, but if 'agent' field exists, use it.
                let linked = false;
                if (rep.entities && rep.entities.agents) {
                    rep.entities.agents.forEach(ag => {
                        const agId = `AGENT:${ag}`;
                        if (seen.has(agId)) {
                            links.push({ source: agId, target: id, color: 'rgba(255,255,255,0.1)' });
                            linked = true;
                        }
                    });
                }

                if (!linked) {
                    links.push({ source: 'ADAM_CORE', target: id, color: 'rgba(100,100,100,0.1)' });
                }
            }
        });

        graphData = { nodes, links };

        // Update Stats HUD
        elStatsNodes.textContent = nodes.length;
        elStatsLinks.textContent = links.length;
    }

    // --- Rendering ---
    function renderGraph() {
        Graph
            .graphData(graphData)
            .nodeLabel('desc')
            .nodeColor('color')
            .nodeVal('val')
            .linkColor(link => link.color || 'rgba(255,255,255,0.2)')
            .linkWidth(1)
            .enableNodeDrag(false)
            .backgroundColor('#050b14')
            .showNavInfo(false)

            // Hover Interaction
            .onNodeHover(node => {
                document.body.style.cursor = node ? 'pointer' : 'default';
                if (node) {
                    showNodeInfo(node);
                } else {
                    elNodeInfo.style.display = 'none';
                }
            })

            // Click Interaction
            .onNodeClick(node => {
                if (node.group === 'REPORT' && node.link) {
                    window.open(node.link, '_blank');
                } else {
                    // Fly to node
                    const distance = 40;
                    const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);
                    Graph.cameraPosition(
                        { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }, // new position
                        node, // lookAt ({ x, y, z })
                        3000  // ms transition duration
                    );
                }
            });

        // Add glow effect via PostProcessing? (Built-in bloom is heavy, skipping for speed)
    }

    function showNodeInfo(node) {
        elNodeInfo.style.display = 'block';
        document.getElementById('info-title').textContent = node.desc || node.id;
        document.getElementById('info-type').textContent = node.group;
        document.getElementById('info-id').textContent = node.id;
    }

    // --- Controls ---
    document.getElementById('btn-reset').addEventListener('click', () => {
        Graph.cameraPosition({ x: 0, y: 0, z: 600 }, { x: 0, y: 0, z: 0 }, 2000);
    });

    document.getElementById('btn-spin').addEventListener('click', () => {
        // Simple auto-rotate toggle logic could go here
        // For now, just a manual kick
        Graph.cameraPosition({ x: 500, y: 200, z: 500 }, null, 5000);
    });

    // Handle Resize
    window.addEventListener('resize', () => {
        Graph.width(window.innerWidth).height(window.innerHeight);
    });

    init();
});
