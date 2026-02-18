
/**
 * ADAM SYSTEM VISUALIZER
 * Handles the interactive canvas rendering of the system evolution.
 */

class SystemVisualizer {
    constructor(canvasId, containerId) {
        this.canvas = document.getElementById(canvasId);
        this.container = document.getElementById(containerId);
        this.ctx = this.canvas.getContext('2d');

        this.nodes = [];
        this.edges = [];
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;

        this.camera = { x: 0, y: 0, zoom: 0.8 };
        this.isDragging = false;
        this.lastMouse = { x: 0, y: 0 };

        this.resize();
        window.addEventListener('resize', () => this.resize());

        this.setupInteraction();
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
    }

    resize() {
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;
        this.canvas.width = this.width;
        this.canvas.height = this.height;
    }

    setupInteraction() {
        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.lastMouse = { x: e.clientX, y: e.clientY };
            this.canvas.style.cursor = 'grabbing';
        });

        window.addEventListener('mouseup', () => {
            this.isDragging = false;
            this.canvas.style.cursor = 'default';
        });

        window.addEventListener('mousemove', (e) => {
            if (this.isDragging) {
                const dx = e.clientX - this.lastMouse.x;
                const dy = e.clientY - this.lastMouse.y;
                this.camera.x += dx;
                this.camera.y += dy;
                this.lastMouse = { x: e.clientX, y: e.clientY };
            }
        });

        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomSpeed = 0.05;
            if (e.deltaY < 0) {
                this.camera.zoom *= (1 + zoomSpeed);
            } else {
                this.camera.zoom *= (1 - zoomSpeed);
            }
        });
    }

    setData(timelineData) {
        this.nodes = [];
        this.edges = [];

        // Layout: Spiral growing outward
        let angle = 0;
        let radius = 0;
        const angleStep = 0.6;
        const radiusStep = 20;

        timelineData.forEach((item, index) => {
            // Position calculation (Spiral)
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius;

            this.nodes.push({
                id: index,
                version: item.version,
                x: x,
                y: y,
                radius: 8 + (item.metrics.complexity / 30),
                color: this.getColor(item.version),
                active: true,
                data: item
            });

            if (index > 0) {
                this.edges.push({
                    source: index - 1,
                    target: index,
                    active: true
                });
            }

            angle += angleStep;
            radius += radiusStep;
        });

        // Center camera roughly
        this.camera.x = 0;
        this.camera.y = 0;
    }

    getColor(version) {
        if (version.includes('v26')) return '#00f3ff'; // Cyan
        if (version.includes('v24')) return '#bd00ff'; // Purple
        if (version.includes('v23')) return '#ffd700'; // Yellow
        return '#ffffff';
    }

    updateVisibility(progressIndex) {
        // Show nodes up to progressIndex
        this.nodes.forEach((node, index) => {
            node.active = index <= progressIndex;
        });
        this.edges.forEach((edge) => {
            edge.active = edge.target <= progressIndex;
        });
    }

    animate() {
        this.ctx.clearRect(0, 0, this.width, this.height);

        this.ctx.save();
        // Apply Camera
        this.ctx.translate(this.width/2, this.height/2); // Center of canvas is origin
        this.ctx.translate(this.camera.x, this.camera.y); // Apply pan
        this.ctx.scale(this.camera.zoom, this.camera.zoom); // Apply zoom around center

        // Draw Edges
        this.ctx.lineWidth = 2;
        this.edges.forEach(edge => {
            if (!edge.active) return;
            const source = this.nodes[edge.source];
            const target = this.nodes[edge.target];

            this.ctx.beginPath();
            this.ctx.moveTo(source.x, source.y);
            this.ctx.lineTo(target.x, target.y);
            this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
            this.ctx.stroke();
        });

        // Draw Nodes
        this.nodes.forEach(node => {
            if (!node.active) return;

            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
            this.ctx.fillStyle = node.color;
            this.ctx.shadowBlur = 15;
            this.ctx.shadowColor = node.color;
            this.ctx.fill();
            this.ctx.shadowBlur = 0;

            // Label
            this.ctx.fillStyle = '#fff';
            this.ctx.font = 'bold 12px JetBrains Mono';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(node.version, node.x, node.y - node.radius - 8);
        });

        this.ctx.restore();
        requestAnimationFrame(this.animate);
    }
}

// --- Integration Logic ---

let globalData = null;
let visualizer = null;
let charts = {};

document.addEventListener('DOMContentLoaded', async () => {
    visualizer = new SystemVisualizer('vizCanvas', 'system-visualizer');

    // Load Data
    try {
        const response = await fetch('data/evolution_data.json');
        const data = await response.json();

        // Reverse for chronological order (Oldest -> Newest) for the graph
        // But keep original for feed? No, feed should be Newest -> Oldest.

        // Data in JSON is Newest First (Changelog order).
        // So for graph we reverse it.
        const chronoData = [...data.timeline].reverse();

        visualizer.setData(chronoData);
        renderFeed(data.timeline);
        renderCharts(chronoData);

        // Setup Slider
        const slider = document.getElementById('timelineSlider');
        slider.max = chronoData.length - 1;
        slider.value = chronoData.length - 1;

        slider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            visualizer.updateVisibility(val);

            // Also scroll feed to relevant card?
            // Feed is reverse order. So index 'val' (chrono) corresponds to length - 1 - val in feed.
            const feedIndex = (chronoData.length - 1) - val;
            highlightFeedCard(feedIndex);
        });

        // Setup Controls
        document.getElementById('btn-reset').addEventListener('click', () => {
             visualizer.camera = { x: 0, y: 0, zoom: 0.8 };
        });

        document.getElementById('btn-play').addEventListener('click', () => {
            let val = 0;
            slider.value = 0;
            visualizer.updateVisibility(0);

            const interval = setInterval(() => {
                val++;
                if (val >= chronoData.length) {
                    clearInterval(interval);
                } else {
                    slider.value = val;
                    visualizer.updateVisibility(val);
                    const feedIndex = (chronoData.length - 1) - val;
                    highlightFeedCard(feedIndex);
                }
            }, 500); // 0.5s per step
        });

    } catch (e) {
        console.error("Failed to load data", e);
        document.getElementById('timeline-feed-content').innerHTML = '<p style="color:red; padding:20px;">Failed to load evolution data.</p>';
    }
});

function renderFeed(timeline) {
    const container = document.getElementById('timeline-feed-content');
    container.innerHTML = '';

    timeline.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = 'timeline-card';
        div.id = `card-${index}`;
        div.innerHTML = `
            <div class="version">
                <span>${item.version}</span>
                <span style="color: #666;">${item.metrics.complexity} pts</span>
            </div>
            <h3>${item.title}</h3>
            <div class="date mono" style="margin-bottom: 10px;">${item.date}</div>
            <p>${item.log || "No log entry available."}</p>

            <div class="tag-list">
                ${item.added.slice(0, 3).map(a => `<span class="tag added">ADDED</span>`).join('')}
                ${item.fixed.length > 0 ? `<span class="tag fixed">FIXED: ${item.fixed.length}</span>` : ''}
            </div>
        `;
        div.onclick = () => {
            // Scroll to this version in graph?
            // Graph index = length - 1 - index
            const graphIndex = (timeline.length - 1) - index;
            document.getElementById('timelineSlider').value = graphIndex;
            visualizer.updateVisibility(graphIndex);

            // Highlight self
            highlightFeedCard(index);
        };
        container.appendChild(div);
    });
}

function highlightFeedCard(index) {
    document.querySelectorAll('.timeline-card').forEach(c => c.classList.remove('active'));
    const card = document.getElementById(`card-${index}`);
    if (card) {
        card.classList.add('active');
        card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

function renderCharts(timeline) {
    const labels = timeline.map(t => t.version);
    const complexity = timeline.map(t => t.metrics.complexity);
    const entropy = timeline.map(t => t.metrics.entropy);

    const commonOptions = {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { x: { display: false }, y: { display: false } },
        elements: { point: { radius: 0 } }
    };

    new Chart(document.getElementById('chartComplexity'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                data: complexity,
                borderColor: '#00f3ff', borderWidth: 2, fill: true,
                backgroundColor: 'rgba(0, 243, 255, 0.1)'
            }]
        },
        options: commonOptions
    });

    new Chart(document.getElementById('chartEntropy'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: entropy,
                backgroundColor: '#bd00ff', borderRadius: 2
            }]
        },
        options: commonOptions
    });
}
