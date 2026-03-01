import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

/**
 * MARKET MAYHEM GRAPH V3.0 (HYBRID REALITY)
 * -----------------------------------------------------------------------------
 * 3D Visualization of Real Market Data + Monte Carlo Projections.
 * Features: Historical Replay, Stochastic Forecasting, System 2 Bias.
 */

// --- CONFIGURATION ---
const CONFIG = {
    futureSteps: 30, // Days to project
    simulationsPerEntity: 20,
    playSpeed: 0.5,
    colors: {
        bull: 0x00ff9d,
        bear: 0xff0055,
        neutral: 0xcccccc,
        history: 0x444444,
        projection: 0x00ccff,
        mean: 0xffaa00
    }
};

// --- STATE MANAGEMENT ---
const State = {
    isPlaying: false,
    currentStep: 0, // Float: 0 to (HistoryLength + FutureSteps)
    historyLength: 0,
    totalSteps: 0,
    data: [], // Loaded entities
    filter: 'ALL',
    sortBy: 'DEFAULT',
    showCone: true,
    useSystemBias: true,
    isStressMode: false,
    selectedEntity: null,
    interactiveNodes: []
};

// --- MONTE CARLO ENGINE ---
class MonteCarloEngine {
    static calculateStats(prices) {
        let returns = [];
        for (let i = 1; i < prices.length; i++) {
            returns.push(Math.log(prices[i] / prices[i - 1]));
        }

        const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
        const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
        const stdDev = Math.sqrt(variance);

        return { drift: mean, vol: stdDev, lastPrice: prices[prices.length - 1] };
    }

    static run(prices, outlook, riskScore, useBias) {
        const stats = this.calculateStats(prices);
        let paths = [];

        // System Bias Factor
        let bias = 0;
        if (useBias) {
            if (outlook.conviction === 'High') bias += 0.002; // +0.2% daily drift
            if (outlook.consensus === 'Buy') bias += 0.001;
            if (outlook.consensus === 'Sell') bias -= 0.002;
            if (riskScore > 80) bias -= 0.003; // High risk penalty
        }

        const drift = stats.drift + bias; // Adjusted Drift
        const dt = 1; // 1 day step

        for (let s = 0; s < CONFIG.simulationsPerEntity; s++) {
            let path = [stats.lastPrice];
            let currentPrice = stats.lastPrice;

            for (let t = 1; t <= CONFIG.futureSteps; t++) {
                // Geometric Brownian Motion: dS = S * (mu*dt + sigma*dW)
                const shock = (Math.random() + Math.random() + Math.random() + Math.random() - 2) / 2; // Approx Normal Dist
                // const shock = (Math.random() - 0.5) * 2; // Uniform (simpler)

                // Euler-Maruyama discretization
                const change = currentPrice * (drift * dt + stats.vol * shock * Math.sqrt(dt));
                currentPrice += change;
                if (currentPrice < 0.01) currentPrice = 0.01; // Floor
                path.push(currentPrice);
            }
            paths.push(path);
        }

        // Calculate Mean Path
        let meanPath = [];
        for (let t = 0; t <= CONFIG.futureSteps; t++) {
            let sum = 0;
            paths.forEach(p => sum += p[t]);
            meanPath.push(sum / CONFIG.simulationsPerEntity);
        }

        return { paths, meanPath, stats };
    }
}

// --- DATA MANAGER ---
class DataManager {
    async load() {
        try {
            const response = await fetch('data/sp500_market_data.json');
            const rawData = await response.json();

            // Process Entities
            State.data = rawData.map((item, index) => {
                const history = item.price_history;
                // Generate Projections
                const mc = MonteCarloEngine.run(history, item.outlook, item.risk_score, true);
                const mcUnbiased = MonteCarloEngine.run(history, item.outlook, item.risk_score, false); // For comparison if needed

                return {
                    id: index,
                    ticker: item.ticker,
                    sector: item.sector,
                    color: this.getSectorColor(item.sector),
                    history: history,
                    mc: mc, // { paths, meanPath, stats }
                    outlook: item.outlook,
                    risk: item.risk_score,
                    financials: item.financials,
                    credit: item.credit,
                    // Normalize positions for 3D space
                    // X: Time (handled by loop), Y: Price (Normalized), Z: Risk/Conviction
                    normFactor: 10 / history[0], // Scale initial price to ~10
                    zPos: (item.risk_score - 50) / 2 // Risk maps to Z depth
                };
            });

            // Set Global Time Params
            if (State.data.length > 0) {
                State.historyLength = State.data[0].history.length;
                State.totalSteps = State.historyLength + CONFIG.futureSteps;
                State.currentStep = State.historyLength - 1; // Start at "Today"
            }

            console.log(`Loaded ${State.data.length} entities. History: ${State.historyLength}, Future: ${CONFIG.futureSteps}`);
            return true;

        } catch (e) {
            console.error("Failed to load data:", e);
            document.getElementById('loading-text').innerText = "DATA ERROR - CHECK CONSOLE";
            document.getElementById('loading-text').style.color = "#ff0000";
            return false;
        }
    }

    getSectorColor(sector) {
        const map = {
            'Technology': 0x00f3ff,
            'Financials': 0x009900,
            'Healthcare': 0xff00ff,
            'Energy': 0xffaa00,
            'Consumer Discretionary': 0xff5500,
            'Industrials': 0xcccccc,
            'Utilities': 0xffff00
        };
        return map[sector] || 0x888888;
    }
}

// --- VISUALIZATION ENGINE ---
let scene, camera, renderer, labelRenderer, controls;
let entityGroup = new THREE.Group();
let raycaster, mouse;

const _tempVec = new THREE.Vector3();
const _tempInteractionVec = new THREE.Vector3();

// Helper to convert (Step, Price, Z) to Vector3
function getVector(entity, step, target) {
    // X Axis: Time (-50 to +50)
    const x = ((step / State.totalSteps) * 100) - 50;

    let price = 0;
    // Determine Price based on History or Projection
    if (step < State.historyLength) {
        // Historical
        const idx = Math.floor(step);
        const nextIdx = Math.min(idx + 1, State.historyLength - 1);
        const alpha = step - idx;
        price = THREE.MathUtils.lerp(entity.history[idx], entity.history[nextIdx], alpha);
    } else {
        // Projected (Use Mean Path)
        const idx = Math.floor(step - State.historyLength);
        const nextIdx = Math.min(idx + 1, CONFIG.futureSteps); // The meanPath includes t=0 (last history) to t=30
        const alpha = step - Math.floor(step);

        // Safety check for array bounds
        const p1 = entity.mc.meanPath[Math.min(idx, entity.mc.meanPath.length-1)];
        const p2 = entity.mc.meanPath[Math.min(nextIdx, entity.mc.meanPath.length-1)];

        price = THREE.MathUtils.lerp(p1, p2, alpha);
    }

    // Y Axis: Price Normalized (centered around 0 visually?)
    // Let's just map Price * normFactor
    const y = (price * entity.normFactor) - 10; // Shift down

    // Z Axis: Risk
    const z = entity.zPos;

    if (target) {
        target.set(x, y, z);
        return target;
    }
    return new THREE.Vector3(x, y, z);
}

async function init() {
    // 1. Scene Setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050505);
    scene.fog = new THREE.FogExp2(0x050505, 0.015); // Cyber fog

    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 10, 60);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    // CSS2D Renderer for Labels
    labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(window.innerWidth, window.innerHeight);
    labelRenderer.domElement.style.position = 'absolute';
    labelRenderer.domElement.style.top = '0px';
    labelRenderer.domElement.style.pointerEvents = 'none'; // Allow clicking through text
    document.getElementById('canvas-container').appendChild(labelRenderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = false;

    // Lights
    const ambientLight = new THREE.AmbientLight(0x404040, 2);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 1);
    dirLight.position.set(10, 20, 10);
    scene.add(dirLight);

    // Grid: Visual reference for Time (X) and Price (Y)
    const grid = new THREE.GridHelper(200, 100, 0x111111, 0x0a0a0a);
    grid.position.y = -10;
    scene.add(grid);

    // "Today" Line
    const todayGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, -50, 0), new THREE.Vector3(0, 50, 0)
    ]);
    const todayMat = new THREE.LineBasicMaterial({ color: 0xff0000, transparent: true, opacity: 0.5, dashSize: 1, gapSize: 1 });
    const todayLine = new THREE.Line(todayGeo, todayMat);
    // Position X calculation: ((historyLength-1) / total) * 100 - 50
    // We'll update this in render if needed, but it's roughly fixed relative to data structure
    scene.add(todayLine);

    // 2. Data Loading
    const manager = new DataManager();
    const loaded = await manager.load();
    if (!loaded) return;

    // Update "Today" Line Position
    const todayX = (((State.historyLength - 1) / State.totalSteps) * 100) - 50;
    todayLine.position.x = todayX;

    // 3. Create Objects
    scene.add(entityGroup);
    createVisuals();

    // 4. UI Events
    setupUI();

    // 5. Interaction
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('click', onMouseClick);
    window.addEventListener('resize', onWindowResize);

    // System 2 Feed Loop
    setInterval(updateNewsFeed, 4000);

    // 6. Start Loop
    document.getElementById('loading-overlay').style.display = 'none';
    animate();
}

function createVisuals() {
    entityGroup.clear();
    State.interactiveNodes = [];

    State.data.forEach(entity => {
        // Create a Group for the Entity
        const group = new THREE.Group();
        entity.group = group; // Store reference
        entityGroup.add(group);

        // --- 1. The Head Node (Sphere) ---
        const geometry = new THREE.SphereGeometry(0.4, 16, 16);
        const material = new THREE.MeshStandardMaterial({
            color: entity.color,
            emissive: entity.color,
            emissiveIntensity: 0.5
        });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.userData = { type: 'node', entity: entity };
        group.add(sphere);
        entity.mesh = sphere; // Link for updates
        State.interactiveNodes.push(sphere);

        // --- 1b. Label (CSS2D) ---
        const div = document.createElement('div');
        div.className = 'label-3d';
        div.textContent = entity.ticker;
        div.style.marginTop = '-1em';
        div.style.color = '#' + entity.color.toString(16);
        const label = new CSS2DObject(div);
        label.position.set(0, 1, 0); // Above sphere
        sphere.add(label);
        entity.label = label;
        // Show labels for high conviction by default, else hide
        label.visible = entity.outlook.conviction === 'High';

        // --- 2. Historical Trail (Line) ---
        const historyPoints = [];
        for (let i = 0; i < State.historyLength; i++) {
            historyPoints.push(getVector(entity, i));
        }
        const histGeo = new THREE.BufferGeometry().setFromPoints(historyPoints);
        const histMat = new THREE.LineBasicMaterial({ color: entity.color, transparent: true, opacity: 0.3 });
        const histLine = new THREE.Line(histGeo, histMat);
        group.add(histLine);

        // --- 3. Monte Carlo Cone (Group of Lines + Tube) ---
        // Only created if needed, toggled by visibility
        const coneGroup = new THREE.Group();
        coneGroup.visible = State.showCone;
        entity.coneGroup = coneGroup; // Link

        // Start point for projections
        const startVec = historyPoints[historyPoints.length - 1];

        // Normalized Factor for Y scaling
        const norm = entity.normFactor;
        const z = entity.zPos;

        // Render subset of paths to save FPS (Lines)
        const pathsToRender = entity.mc.paths.slice(0, 5);

        pathsToRender.forEach(path => {
            const points = [startVec]; // Start at last known
            for (let t = 1; t < path.length; t++) {
                // Calculate position relative to start
                const stepAbs = (State.historyLength - 1) + t;
                const x = ((stepAbs / State.totalSteps) * 100) - 50;
                const y = (path[t] * norm) - 10;
                points.push(new THREE.Vector3(x, y, z));
            }
            const geo = new THREE.BufferGeometry().setFromPoints(points);
            const mat = new THREE.LineBasicMaterial({
                color: entity.color,
                transparent: true,
                opacity: 0.05 // Very faint
            });
            const line = new THREE.Line(geo, mat);
            coneGroup.add(line);
        });

        // Add Mean Path (Brighter)
        const meanPoints = [startVec];
        for(let t=1; t<entity.mc.meanPath.length; t++){
             const stepAbs = (State.historyLength - 1) + t;
             const x = ((stepAbs / State.totalSteps) * 100) - 50;
             const y = (entity.mc.meanPath[t] * norm) - 10;
             meanPoints.push(new THREE.Vector3(x, y, z));
        }
        const meanGeo = new THREE.BufferGeometry().setFromPoints(meanPoints);
        const meanMat = new THREE.LineBasicMaterial({
            color: 0xffffff, // White/Bright for mean
            transparent: true,
            opacity: 0.4,
            dashSize: 0.5
        });
        const meanLine = new THREE.Line(meanGeo, meanMat);
        coneGroup.add(meanLine);

        // --- Confidence Tube ---
        // Create a tube around the mean path to show variance
        const tubePath = new THREE.CatmullRomCurve3(meanPoints);
        // Radius scales with time (t) to show increasing uncertainty
        const tubeGeo = new THREE.TubeGeometry(tubePath, 30, 0.1, 8, false);
        // Custom radius requires modifying geometry or using shader, here we use simple transparent mesh
        // To make it expand, we'd need a custom geometry generator, but for simplicity:
        // We'll just use a low-opacity mesh for visual flair
        const tubeMat = new THREE.MeshBasicMaterial({
            color: entity.color,
            transparent: true,
            opacity: 0.05,
            wireframe: true
        });
        const tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
        coneGroup.add(tubeMesh);

        group.add(coneGroup);
    });
}

function updatePositions() {
    const time = Date.now() * 0.005;

    State.data.forEach(entity => {
        // Filter Check
        const isVisible = (State.filter === 'ALL' || entity.sector === State.filter);
        
        // Hide entire group if filtered out
        if (entity.group) {
            entity.group.visible = isVisible;
        }
        
        if (!isVisible) return;

        // Toggle Cone visibility based on state
        if (entity.coneGroup) {
            entity.coneGroup.visible = State.showCone;
        }

        // Move Head Node
        getVector(entity, State.currentStep, entity.mesh.position);

        // Highlight & Stress Mode
        let baseEmissive = 0.5;
        let scale = 1;

        if (State.selectedEntity && State.selectedEntity.id === entity.id) {
            scale = 1.5;
            baseEmissive = 1.0;
            if(entity.label) entity.label.visible = true;
        } else {
            // Revert label visibility unless high conviction
            if(entity.label) entity.label.visible = entity.outlook.conviction === 'High';

            // Stress Mode Logic
            if (State.isStressMode && entity.risk > 80) {
                // Pulse red
                const pulse = (Math.sin(time) + 1) * 0.5; // 0 to 1
                entity.mesh.material.color.setHex(0xff0000);
                entity.mesh.material.emissive.setHex(0xff0000);
                baseEmissive = 0.5 + (pulse * 0.5);
                scale = 1 + (pulse * 0.2);
                if(entity.label) entity.label.visible = true; // Show label on stress
            } else {
                // Reset color if not stressed
                entity.mesh.material.color.setHex(entity.color);
                entity.mesh.material.emissive.setHex(entity.color);
            }
        }

        entity.mesh.scale.set(scale, scale, scale);
        entity.mesh.material.emissiveIntensity = baseEmissive;
    });
}

function animate() {
    requestAnimationFrame(animate);

    if (State.isPlaying) {
        State.currentStep += CONFIG.playSpeed;
        if (State.currentStep >= State.totalSteps - 1) {
            State.currentStep = 0; // Loop or pause? Loop for demo
        }

        // Update Slider UI
        const slider = document.getElementById('time-slider');
        if (slider) {
            slider.value = (State.currentStep / (State.totalSteps - 1)) * 100;
            updateTimeDisplay();
        }
    }

    // Camera Focus Logic
    if (State.selectedEntity) {
        // Find target position (head node)
        const targetPos = State.selectedEntity.mesh.position.clone();
        
        // Offset for camera
        // const offset = new THREE.Vector3(0, 5, 20); 
        // Simple lerp for smooth follow
        controls.target.lerp(targetPos, 0.05);
        // We don't force camera position to allow user rotation, just target
    }

    updatePositions();
    controls.update();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
    updateInteraction();
}

// --- UI HELPERS ---

function updateTimeDisplay() {
    const display = document.getElementById('time-display');
    const isFuture = State.currentStep >= State.historyLength;
    const day = Math.floor(State.currentStep);

    let text = `DAY ${day}`;
    if (isFuture) {
        text += ` [PROJECTED +${day - State.historyLength + 1}]`;
        display.style.color = "#00ccff";
    } else {
        text += ` [HISTORY]`;
        display.style.color = "#00ff9d";
    }
    display.innerText = text;
}

function setupUI() {
    // Play/Pause
    document.getElementById('btn-play').addEventListener('click', () => {
        State.isPlaying = !State.isPlaying;
        document.getElementById('btn-play').innerHTML = State.isPlaying ? '<i class="fas fa-pause"></i>' : '<i class="fas fa-play"></i>';
    });

    // Slider
    document.getElementById('time-slider').addEventListener('input', (e) => {
        State.isPlaying = false;
        document.getElementById('btn-play').innerHTML = '<i class="fas fa-play"></i>';
        const pct = parseFloat(e.target.value);
        State.currentStep = (pct / 100) * (State.totalSteps - 1);
        updateTimeDisplay();
    });

    // Filters & Sorts (Existing logic maps mostly same)
    document.getElementById('filter-sector').addEventListener('change', (e) => {
        State.filter = e.target.value;
    });

    // New Toggles
    const toggleCone = document.getElementById('toggle-cone');
    if (toggleCone) {
        toggleCone.addEventListener('change', (e) => {
            State.showCone = e.target.checked;
        });
    }

    const toggleStress = document.getElementById('toggle-stress');
    if (toggleStress) {
        toggleStress.addEventListener('change', (e) => {
            State.isStressMode = e.target.checked;
        });
    }

    // Panel Tabs
    document.querySelectorAll('.panel-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.panel-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            const target = tab.getAttribute('data-tab');
            document.getElementById(`tab-${target}`).classList.add('active');
        });
    });

    // Deep Dive Button
    document.getElementById('btn-deep-dive').addEventListener('click', () => {
        if (State.selectedEntity) {
            const ticker = State.selectedEntity.ticker.toLowerCase();
            // Try specific file first or fallback to query param logic (simulated here)
            const url = `${ticker}_company_report.html`; // Simple mapping
            window.open(url, '_blank');
        }
    });
}

function onMouseMove(event) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
}

function onMouseClick(event) {
    raycaster.setFromCamera(mouse, camera);
    // Raycast against Spheres only
    const intersects = raycaster.intersectObjects(State.interactiveNodes);

    if (intersects.length > 0) {
        const entity = intersects[0].object.userData.entity;
        selectEntity(entity);
    } else {
        State.selectedEntity = null;
        document.getElementById('attribution-panel').style.display = 'none';
    }
}

function updateInteraction() {
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(State.interactiveNodes);
    const tooltip = document.getElementById('tooltip');

    if (intersects.length > 0) {
        const entity = intersects[0].object.userData.entity;
        const vec = getVector(entity, State.currentStep, _tempInteractionVec); // Get current price from math, not just mesh pos

        // Denormalize price approximation
        // y = (price * norm) - 10  => price = (y+10)/norm
        const displayPrice = (vec.y + 10) / entity.normFactor;

        tooltip.style.display = 'block';
        tooltip.style.left = (event.clientX + 15) + 'px';
        tooltip.style.top = (event.clientY + 15) + 'px';

        document.getElementById('tt-title').innerText = entity.ticker;
        document.getElementById('tt-val1').innerText = `$${displayPrice.toFixed(2)}`;

        const isFuture = State.currentStep >= State.historyLength;
        document.getElementById('tt-val2').innerText = isFuture ? "Simulated" : "Real Data";
    } else {
        tooltip.style.display = 'none';
    }
}

function selectEntity(entity) {
    State.selectedEntity = entity;
    const panel = document.getElementById('attribution-panel');
    panel.style.display = 'block';

    // Overview Tab
    document.getElementById('attr-ticker').innerText = entity.ticker;
    document.getElementById('attr-name').innerText = "SECTOR: " + entity.sector.toUpperCase();
    document.getElementById('attr-thesis').innerText = entity.outlook.rationale;
    document.getElementById('attr-conviction').innerText = entity.outlook.conviction;
    document.getElementById('attr-consensus').innerText = entity.outlook.consensus;
    document.getElementById('attr-risk').innerText = entity.risk + "/100";

    // Financials Tab
    const finContainer = document.getElementById('fin-container');
    if (entity.financials) {
        const years = entity.financials.years || [];
        const rev = entity.financials.revenue || [];
        const eps = entity.financials.eps || [];
        
        let html = '<table style="width:100%; font-size:0.9em; border-collapse:collapse;">';
        html += '<tr style="border-bottom:1px solid #444; color:#888;"><th>Year</th><th>Rev ($B)</th><th>EPS</th></tr>';
        
        // Show last 3 years
        const start = Math.max(0, years.length - 3);
        for (let i = start; i < years.length; i++) {
            html += `<tr>
                <td style="padding:4px;">${years[i]}</td>
                <td style="color:#00ccff;">${rev[i]}</td>
                <td style="color:#00ff9d;">${eps[i]}</td>
            </tr>`;
        }
        html += '</table>';
        finContainer.innerHTML = html;
    } else {
        finContainer.innerHTML = '<div style="color:#666; font-style:italic;">Data Unavailable</div>';
    }

    // Credit Tab
    const creditContainer = document.getElementById('credit-container');
    if (entity.credit) {
        creditContainer.innerHTML = `
            <div class="data-row"><span>Rating:</span> <span style="color:#ffaa00;">${entity.credit.rating}</span></div>
            <div class="data-row"><span>Leverage:</span> <span>${entity.credit.leverage}x</span></div>
            <div class="data-row"><span>Coverage:</span> <span>${entity.credit.interest_coverage}x</span></div>
            <div class="data-row"><span>Liquidity:</span> <span>${entity.credit.liquidity_score}/100</span></div>
            <div style="margin-top:10px; font-size:0.7em; color:#666;">
                PD: ${entity.credit.pd_rating}<br>
                Reg: ${entity.credit.regulatory_rating}
            </div>
        `;
    } else {
        creditContainer.innerHTML = '<div style="color:#666; font-style:italic;">Credit Profile Unavailable</div>';
    }
}

function updateNewsFeed() {
    const feed = document.getElementById('news-feed');
    if (!feed || !State.data.length) return;

    // Pick random visible entity
    const visibleEntities = State.data.filter(e => e.mesh.visible);
    if (!visibleEntities.length) return;

    const entity = visibleEntities[Math.floor(Math.random() * visibleEntities.length)];
    
    // Use Real Data
    const change = ((entity.history[entity.history.length-1] - entity.history[entity.history.length-2])/entity.history[entity.history.length-2]*100).toFixed(2);
    const sign = change >= 0 ? '+' : '';
    
    const templates = [
        `> UPDATE: ${entity.ticker} ${sign}${change}% today. Outlook: ${entity.outlook.consensus}.`,
        `> RISK ALERT: ${entity.ticker} score at ${entity.risk}. ${entity.outlook.rationale.substring(0, 50)}...`,
        `> CONVICTION: System holds ${entity.outlook.conviction} view on ${entity.ticker}.`,
        `> SECTOR SCAN: ${entity.sector} moving. ${entity.ticker} implied volatility rising.`
    ];
    const text = templates[Math.floor(Math.random() * templates.length)];

    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.style.borderLeft = `2px solid #${entity.color.toString(16)}`;
    entry.style.paddingLeft = '5px';
    entry.innerText = text;
    
    feed.insertBefore(entry, feed.firstChild);
    
    // Keep only last 20
    if (feed.children.length > 20) {
        feed.removeChild(feed.lastChild);
    }
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    labelRenderer.setSize(window.innerWidth, window.innerHeight);
}

// --- BOOTSTRAP ---
init();
