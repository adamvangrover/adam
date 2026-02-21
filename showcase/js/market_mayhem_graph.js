import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/**
 * MARKET MAYHEM GRAPH V2.5 (HYBRID REALITY)
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
    selectedEntity: null
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
let scene, camera, renderer, controls;
let entityGroup = new THREE.Group();
let raycaster, mouse;

// Helper to convert (Step, Price, Z) to Vector3
function getVector(entity, step) {
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

    // 6. Start Loop
    document.getElementById('loading-overlay').style.display = 'none';
    animate();
}

function createVisuals() {
    entityGroup.clear();

    State.data.forEach(entity => {
        // --- 1. The Head Node (Sphere) ---
        const geometry = new THREE.SphereGeometry(0.4, 16, 16);
        const material = new THREE.MeshStandardMaterial({
            color: entity.color,
            emissive: entity.color,
            emissiveIntensity: 0.5
        });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.userData = { type: 'node', entity: entity };
        entityGroup.add(sphere);
        entity.mesh = sphere; // Link for updates

        // --- 2. Historical Trail (Line) ---
        const historyPoints = [];
        for (let i = 0; i < State.historyLength; i++) {
            historyPoints.push(getVector(entity, i));
        }
        const histGeo = new THREE.BufferGeometry().setFromPoints(historyPoints);
        const histMat = new THREE.LineBasicMaterial({ color: entity.color, transparent: true, opacity: 0.3 });
        const histLine = new THREE.Line(histGeo, histMat);
        entityGroup.add(histLine);

        // --- 3. Monte Carlo Cone (Group of Lines) ---
        // Only created if needed, toggled by visibility
        const coneGroup = new THREE.Group();
        coneGroup.visible = State.showCone;
        entity.coneGroup = coneGroup; // Link

        // Start point for projections
        const startVec = historyPoints[historyPoints.length - 1];

        // Normalized Factor for Y scaling
        const norm = entity.normFactor;
        const z = entity.zPos;

        // Render subset of paths to save FPS
        const pathsToRender = entity.mc.paths.slice(0, 10);

        pathsToRender.forEach(path => {
            const points = [startVec]; // Start at last known
            for (let t = 1; t < path.length; t++) {
                // Calculate position relative to start
                // Time advances from todayX
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

        entityGroup.add(coneGroup);
    });
}

function updatePositions() {
    State.data.forEach(entity => {
        // Filter Check
        const isVisible = (State.filter === 'ALL' || entity.sector === State.filter);
        entity.mesh.visible = isVisible;
        entity.coneGroup.visible = isVisible && State.showCone;
        // Also hide history lines if filtered? (Assuming yes for clarity)
        // Note: history lines are separate objects in group, simpler to just hide group or rebuild.
        // For performance, we'll just scale the node to 0 if hidden or use visible property if linked properly.
        // For this prototype, we just toggle the head node. Ideally we toggle the trails too.

        if (!isVisible) return;

        // Move Head Node
        const pos = getVector(entity, State.currentStep);
        entity.mesh.position.copy(pos);

        // Highlight
        if (State.selectedEntity && State.selectedEntity.id === entity.id) {
            entity.mesh.scale.set(1.5, 1.5, 1.5);
            entity.mesh.material.emissiveIntensity = 1.0;
        } else {
            entity.mesh.scale.set(1, 1, 1);
            entity.mesh.material.emissiveIntensity = 0.5;
        }
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

    updatePositions();
    controls.update();
    renderer.render(scene, camera);
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

    // New Toggles (Will be added to HTML next step)
    const toggleCone = document.getElementById('toggle-cone');
    if (toggleCone) {
        toggleCone.addEventListener('change', (e) => {
            State.showCone = e.target.checked;
        });
    }
}

function onMouseMove(event) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
}

function onMouseClick(event) {
    raycaster.setFromCamera(mouse, camera);
    // Raycast against Spheres only
    const nodes = entityGroup.children.filter(c => c.userData.type === 'node');
    const intersects = raycaster.intersectObjects(nodes);

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
    const nodes = entityGroup.children.filter(c => c.userData.type === 'node' && c.visible);
    const intersects = raycaster.intersectObjects(nodes);
    const tooltip = document.getElementById('tooltip');

    if (intersects.length > 0) {
        const entity = intersects[0].object.userData.entity;
        const vec = getVector(entity, State.currentStep); // Get current price from math, not just mesh pos

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

    document.getElementById('attr-agent-name').innerText = "SYSTEM ORACLE";
    document.getElementById('attr-prov-id').innerText = `RISK: ${entity.risk}/100`;
    document.getElementById('attr-thesis').innerText = entity.outlook.rationale;
    document.getElementById('attr-accuracy').innerText = `Bias: ${entity.outlook.conviction}`;
    document.getElementById('attr-direction').innerText = entity.outlook.consensus.toUpperCase();
    document.getElementById('attr-model').innerText = "Monte Carlo (Geometric Brownian)";
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// --- BOOTSTRAP ---
init();
