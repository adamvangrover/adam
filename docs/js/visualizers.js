// Canvas & Visual Simulation Handlers

class NetworkVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.nodes = [];
        this.connections = [];
        this.isActive = true;

        this.init();
        this.animate();

        window.addEventListener('resize', () => this.resize());
    }

    init() {
        this.resize();
        // Generate nodes
        const numNodes = Math.floor(window.innerWidth / 30);
        for(let i=0; i<numNodes; i++) {
            this.nodes.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                radius: Math.random() * 2 + 1,
                color: Math.random() > 0.5 ? 'rgba(6, 182, 212, 0.8)' : 'rgba(16, 185, 129, 0.8)' // Cyan or Emerald
            });
        }
    }

    resize() {
        this.canvas.width = this.canvas.parentElement.offsetWidth;
        this.canvas.height = this.canvas.parentElement.offsetHeight;
    }

    animate() {
        if(!this.isActive) return;

        // Use recursive requestAnimationFrame
        requestAnimationFrame(() => this.animate());

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Update nodes
        for(let i=0; i<this.nodes.length; i++) {
            let n = this.nodes[i];
            n.x += n.vx;
            n.y += n.vy;

            // Bounce
            if(n.x < 0 || n.x > this.canvas.width) n.vx *= -1;
            if(n.y < 0 || n.y > this.canvas.height) n.vy *= -1;

            // Draw node
            this.ctx.beginPath();
            this.ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
            this.ctx.fillStyle = n.color;
            this.ctx.fill();
        }

        // Draw connections
        this.ctx.lineWidth = 0.5;
        for(let i=0; i<this.nodes.length; i++) {
            for(let j=i+1; j<this.nodes.length; j++) {
                let dx = this.nodes[i].x - this.nodes[j].x;
                let dy = this.nodes[i].y - this.nodes[j].y;
                let dist = Math.sqrt(dx*dx + dy*dy);

                if(dist < 100) {
                    this.ctx.beginPath();
                    this.ctx.moveTo(this.nodes[i].x, this.nodes[i].y);
                    this.ctx.lineTo(this.nodes[j].x, this.nodes[j].y);
                    this.ctx.strokeStyle = \`rgba(30, 41, 59, \${1 - dist/100})\`;
                    this.ctx.stroke();
                }
            }
        }
    }

    stop() {
        this.isActive = false;
    }
}

// Chart.js Mock Generators (requires Chart.js CDN in HTML)
class RiskCharts {
    static renderSensitivityMatrix(ctxId) {
        if(!document.getElementById(ctxId) || typeof Chart === 'undefined') return;

        const ctx = document.getElementById(ctxId).getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Q1', 'Q2', 'Q3', 'Q4', 'Q1 (F)'],
                datasets: [
                    {
                        label: 'Base Case Yield',
                        data: [4.2, 4.5, 4.8, 5.1, 5.0],
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Stress Scenario (Tail Risk)',
                        data: [4.2, 4.9, 5.8, 6.5, 7.2],
                        borderColor: '#f59e0b',
                        borderDash: [5, 5],
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#94a3b8' } }
                },
                scales: {
                    y: { grid: { color: 'rgba(30, 41, 59, 0.5)' }, ticks: { color: '#94a3b8' } },
                    x: { grid: { color: 'rgba(30, 41, 59, 0.5)' }, ticks: { color: '#94a3b8' } }
                }
            }
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    if(document.getElementById('network-canvas')) {
        window.netViz = new NetworkVisualizer('network-canvas');
    }

    // Auto-init charts if present
    if(document.getElementById('risk-chart-1')) {
        // Assume Chart.js is loaded in the specific HTML file
        RiskCharts.renderSensitivityMatrix('risk-chart-1');
    }
});