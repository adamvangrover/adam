import React, { useEffect, useState, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { useTranslation } from 'react-i18next';
import GlassCard from '../common/GlassCard';
import { Brain, Play, RotateCcw } from 'lucide-react';

const NeuralDashboard = () => {
    const { t } = useTranslation();
    const fgRef = useRef();
    const [graphData, setGraphData] = useState({ nodes: [], links: [] });
    const [activeNode, setActiveNode] = useState(null);
    const [query, setQuery] = useState("Analyze Apple Inc. credit risk");
    const [status, setStatus] = useState("Idle");

    // Mock graph structure for the "Cyclical Reasoning Engine"
    const initialGraph = {
        nodes: [
            { id: "MetaOrchestrator", group: 1, label: "Meta Orchestrator" },
            { id: "Planner", group: 2, label: "Neuro-Symbolic Planner" },
            { id: "GraphEngine", group: 3, label: "Reasoning Graph" },
            { id: "Draft", group: 4, label: "Draft Agent" },
            { id: "Critique", group: 4, label: "Critique Agent" },
            { id: "Refine", group: 4, label: "Refinement Node" },
            { id: "KnowledgeBase", group: 5, label: "Unified Knowledge Graph" }
        ],
        links: [
            { source: "MetaOrchestrator", target: "Planner" },
            { source: "Planner", target: "GraphEngine" },
            { source: "GraphEngine", target: "Draft" },
            { source: "Draft", target: "Critique" },
            { source: "Critique", target: "Refine" },
            { source: "Refine", target: "Draft" }, // The Cycle
            { source: "Draft", target: "KnowledgeBase" }
        ]
    };

    useEffect(() => {
        setGraphData(initialGraph);
    }, []);

    const runSimulation = () => {
        setStatus("Running...");
        let step = 0;
        const sequence = ["MetaOrchestrator", "Planner", "GraphEngine", "Draft", "KnowledgeBase", "Draft", "Critique", "Refine", "Draft", "Critique", "MetaOrchestrator"];

        const interval = setInterval(() => {
            if (step >= sequence.length) {
                clearInterval(interval);
                setStatus("Complete");
                setActiveNode(null);
                return;
            }

            const nodeId = sequence[step];
            setActiveNode(nodeId);

            // Focus on the active node
            if (fgRef.current) {
                // Find node object
                const node = graphData.nodes.find(n => n.id === nodeId);
                if (node) {
                     // fgRef.current.centerAt(node.x, node.y, 1000);
                }
            }

            step++;
        }, 1000);
    };

    const handleNodePaint = (node, ctx, globalScale) => {
        const isActive = node.id === activeNode;
        const fontSize = 12/globalScale;

        ctx.beginPath();
        const r = isActive ? 8 : 5;
        ctx.arc(node.x, node.y, r, 0, 2 * Math.PI, false);
        ctx.fillStyle = isActive ? '#06b6d4' : getNodeColor(node); // Cyan if active
        ctx.fill();

        if (isActive) {
             ctx.beginPath();
             ctx.arc(node.x, node.y, r + 4, 0, 2 * Math.PI, false);
             ctx.strokeStyle = '#06b6d4';
             ctx.lineWidth = 2 / globalScale;
             ctx.stroke();
        }

        ctx.font = `${fontSize}px Sans-Serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#fff';
        ctx.fillText(node.label, node.x, node.y + r + fontSize);
    };

    const getNodeColor = (node) => {
        const colors = ['#64748b', '#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6'];
        return colors[node.group] || '#fff';
    };

    return (
        <div className="space-y-6 h-full flex flex-col">
             <div className="flex justify-between items-center">
                <h2 className="text-xl font-bold text-white flex items-center">
                    <Brain className="mr-3 text-cyan-500" />
                    Neural Dashboard: Agent Reasoning Graph
                </h2>
                <div className="flex space-x-3">
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        className="bg-slate-900 border border-slate-700 rounded px-4 py-2 text-slate-200 w-96 focus:outline-none focus:border-cyan-500"
                    />
                    <button
                        onClick={runSimulation}
                        disabled={status === "Running..."}
                        className="flex items-center bg-cyan-600 hover:bg-cyan-700 text-white px-4 py-2 rounded transition-colors disabled:opacity-50"
                    >
                        <Play size={18} className="mr-2" />
                        Run Analysis
                    </button>
                    <button
                         onClick={() => { setActiveNode(null); setStatus("Idle"); }}
                         className="flex items-center bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded transition-colors"
                    >
                        <RotateCcw size={18} />
                    </button>
                </div>
            </div>

            <GlassCard className="flex-1 min-h-[500px] relative overflow-hidden border-cyan-500/30 shadow-[0_0_15px_rgba(6,182,212,0.15)]">
                 <div className="absolute top-4 left-4 z-10 bg-slate-900/80 p-3 rounded border border-slate-700 text-xs font-mono">
                    <div>STATUS: <span className={status === "Running..." ? "text-cyan-400 animate-pulse" : "text-slate-400"}>{status.toUpperCase()}</span></div>
                    <div>ACTIVE NODE: <span className="text-white">{activeNode || "None"}</span></div>
                 </div>

                 <ForceGraph2D
                    ref={fgRef}
                    graphData={graphData}
                    nodeLabel="label"
                    nodeCanvasObject={handleNodePaint}
                    linkColor={() => '#334155'}
                    backgroundColor="rgba(0,0,0,0)"
                    enableNodeDrag={true}
                    d3AlphaDecay={0.01}
                    d3VelocityDecay={0.3}
                 />
            </GlassCard>
        </div>
    );
};

export default NeuralDashboard;
