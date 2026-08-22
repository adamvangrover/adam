// Canvas and Chart Visualization Hooks

function initNetworkCanvas() {
    const canvas = document.getElementById('network-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let width = canvas.width = canvas.parentElement.clientWidth;
    let height = canvas.height = 400; // Fixed height for hero

    // Resize handler
    window.addEventListener('resize', () => {
        width = canvas.width = canvas.parentElement.clientWidth;
    });

    const nodes = [];
    const numNodes = 60;

    for(let i=0; i<numNodes; i++) {
        nodes.push({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            radius: Math.random() * 2 + 1
        });
    }

    function draw() {
        ctx.clearRect(0, 0, width, height);

        // Update & Draw Nodes
        ctx.fillStyle = 'rgba(6, 182, 212, 0.8)'; // Cyan
        nodes.forEach(node => {
            node.x += node.vx;
            node.y += node.vy;

            // Bounce
            if(node.x < 0 || node.x > width) node.vx *= -1;
            if(node.y < 0 || node.y > height) node.vy *= -1;

            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
            ctx.fill();
        });

        // Draw Connections
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.15)';
        ctx.lineWidth = 1;
        for(let i=0; i<numNodes; i++) {
            for(let j=i+1; j<numNodes; j++) {
                const dx = nodes[i].x - nodes[j].x;
                const dy = nodes[i].y - nodes[j].y;
                const dist = Math.sqrt(dx*dx + dy*dy);

                if(dist < 100) {
                    ctx.beginPath();
                    ctx.moveTo(nodes[i].x, nodes[i].y);
                    ctx.lineTo(nodes[j].x, nodes[j].y);
                    ctx.stroke();
                }
            }
        }

        requestAnimationFrame(draw);
    }

    draw();
}

function initRiskChart() {
    const ctx = document.getElementById('risk-chart');
    if (!ctx || typeof Chart === 'undefined') return;

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Q1', 'Q2', 'Q3', 'Q4', 'Q1 (Proj)'],
            datasets: [{
                label: 'Covenant Breach Probability',
                data: [12, 19, 15, 25, 42],
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            },
            {
                label: 'Liquidity Runway (Days)',
                data: [180, 160, 145, 120, 90],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    grid: { color: 'rgba(30, 41, 59, 0.5)' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { color: 'rgba(30, 41, 59, 0.5)' },
                    ticks: { color: '#94a3b8' }
                }
            },
            plugins: {
                legend: { labels: { color: '#f8fafc', font: { family: 'JetBrains Mono' } } }
            }
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    initNetworkCanvas();
    // setTimeout to allow CDN scripts to load
    setTimeout(initRiskChart, 500);
});
