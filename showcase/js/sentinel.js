const { useState, useEffect, useCallback } = React;
const { ReactFlow, MiniMap, Controls, Background, useNodesState, useEdgesState } = window.ReactFlow;

// --- Mock Data ---
const MOCK_NPV_FEES = 1000000;
const SCALAR = 0.15;
const GATE = MOCK_NPV_FEES * SCALAR;

const initialNodes = [
  { id: '1', position: { x: 50, y: 50 }, data: { label: 'Data Ingestion (K-1s)' }, type: 'input' },
  { id: '2', position: { x: 50, y: 150 }, data: { label: 'Parse CreditMetrics' } },
  { id: '3', position: { x: 50, y: 250 }, data: { label: 'Risk Synthesis Engine' } },
  { id: '4', position: { x: 50, y: 350 }, data: { label: 'Decision Gate' } },
];

const initialEdges = [
  { id: 'e1-2', source: '1', target: '2' },
  { id: 'e2-3', source: '2', target: '3' },
  { id: 'e3-4', source: '3', target: '4' },
];

// --- Components ---

const FrictionModal = ({ data, onClose, onAuthorize }) => {
    const [rationale, setRationale] = useState("");

    return (
        <div className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-50 backdrop-blur-sm">
            <div className="panel p-6 w-[600px] border-cyber-danger shadow-[0_0_20px_rgba(255,51,102,0.3)]">
                <div className="flex justify-between items-center border-b border-cyber-border pb-4 mb-4">
                    <h2 className="text-xl font-mono text-cyber-danger font-bold flex items-center">
                        <span className="mr-2">⚠️</span> MFA STEP-UP REQUIRED
                    </h2>
                    <button onClick={onClose} className="text-gray-400 hover:text-white">✕</button>
                </div>

                <div className="mb-6">
                    <p className="text-sm text-gray-300 mb-4">
                        The requested action violates deterministic risk thresholds.
                    </p>
                    <div className="bg-black p-3 rounded border border-cyber-border font-mono text-xs mb-4">
                        <div className="grid grid-cols-2 gap-2">
                            <span className="text-gray-500">Expected Loss (EL):</span>
                            <span className="text-cyber-danger">${data.el.toLocaleString()}</span>
                            <span className="text-gray-500">Threshold Gate:</span>
                            <span className="text-cyber-accent">${data.gate.toLocaleString()}</span>
                            <span className="text-gray-500">Audit Hash:</span>
                            <span className="text-gray-400 truncate">{data.hash}</span>
                        </div>
                    </div>

                    <label className="block text-sm font-mono text-cyber-accent mb-2">AUTHORIZATION RATIONALE</label>
                    <textarea
                        className="w-full bg-black border border-cyber-border rounded p-2 text-white font-mono text-sm focus:border-cyber-accent focus:outline-none"
                        rows="3"
                        placeholder="Provide justification for overriding risk limits..."
                        value={rationale}
                        onChange={(e) => setRationale(e.target.value)}
                    ></textarea>
                </div>

                <div className="flex justify-end gap-4">
                    <button onClick={onClose} className="px-4 py-2 rounded font-mono text-sm border border-gray-600 hover:bg-gray-800 transition">
                        CANCEL
                    </button>
                    <button
                        onClick={() => onAuthorize(rationale)}
                        disabled={!rationale}
                        className={`px-4 py-2 rounded font-mono text-sm ${rationale ? 'glass-btn-danger' : 'opacity-50 cursor-not-allowed border border-cyber-danger text-cyber-danger'}`}
                    >
                        BIOMETRIC AUTHORIZE
                    </button>
                </div>
            </div>
        </div>
    );
};

const RegulatoryDAG = () => {
    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

    return (
        <div style={{ height: 300, width: '100%' }} className="bg-black rounded border border-cyber-border">
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                fitView
            >
                <Background color="#2a2a40" gap={16} />
                <Controls className="fill-cyber-accent" />
            </ReactFlow>
        </div>
    );
};

const SentinelDashboard = () => {
    const [feed, setFeed] = useState([]);
    const [modalData, setModalData] = useState(null);

    // Simulate incoming data
    useEffect(() => {
        const interval = setInterval(() => {
            const pd = Math.random() * 0.1;
            const lgd = 0.4 + Math.random() * 0.4;
            const ead = 1000000 + Math.random() * 4000000;
            const el = pd * lgd * ead;

            const item = {
                id: Math.random().toString(36).substr(2, 9),
                timestamp: new Date().toLocaleTimeString(),
                pd: pd.toFixed(4),
                lgd: lgd.toFixed(2),
                ead: Math.round(ead),
                el: Math.round(el),
                gate: GATE,
                conviction: (0.8 + Math.random() * 0.18).toFixed(2),
                status: el >= GATE ? 'HITL_TIER_3' : (Math.random() > 0.5 ? 'AUTOMATED' : 'HOTL'),
                hash: 'sha256-' + Math.random().toString(36).substr(2, 16)
            };

            setFeed(prev => [item, ...prev].slice(0, 10));
        }, 5000);

        return () => clearInterval(interval);
    }, []);

    const handleAction = (item) => {
        if (item.status === 'HITL_TIER_3') {
            setModalData(item);
        } else {
            alert(`Action: ${item.status}\nNo step-up required.`);
        }
    };

    return (
        <div className="min-h-screen p-6">
            <header className="mb-8 border-b border-cyber-border pb-4 flex justify-between items-end">
                <div>
                    <h1 className="text-3xl font-mono text-cyber-accent font-bold tracking-widest">PROJECT SENTINEL</h1>
                    <p className="text-sm text-gray-400 font-mono mt-1">AI GOVERNANCE HARNESS // ACTIVE</p>
                </div>
                <div className="text-right font-mono text-xs">
                    <p>Client NPV Fees: ${MOCK_NPV_FEES.toLocaleString()}</p>
                    <p>Risk Scalar: {SCALAR}</p>
                    <p className="text-cyber-accent mt-1 border-t border-cyber-border pt-1">Threshold Gate: ${GATE.toLocaleString()}</p>
                </div>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column: Data Feed */}
                <div className="lg:col-span-2 space-y-6">
                    <div className="panel p-4">
                        <h2 className="text-lg font-mono text-white mb-4 border-b border-cyber-border pb-2">LIVE INGESTION FEED</h2>
                        <div className="overflow-x-auto">
                            <table className="w-full text-left text-sm font-mono">
                                <thead>
                                    <tr className="text-gray-500 border-b border-cyber-border">
                                        <th className="pb-2">TIME</th>
                                        <th className="pb-2">EAD</th>
                                        <th className="pb-2">EXPECTED LOSS</th>
                                        <th className="pb-2">CONVICTION</th>
                                        <th className="pb-2">ROUTING PATH</th>
                                        <th className="pb-2">ACTION</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {feed.map((item) => (
                                        <tr key={item.id} className="border-b border-cyber-border/50 hover:bg-white/5 transition">
                                            <td className="py-3 text-gray-400">{item.timestamp}</td>
                                            <td className="py-3">${item.ead.toLocaleString()}</td>
                                            <td className={`py-3 ${item.el >= item.gate ? 'text-cyber-danger' : 'text-cyber-success'}`}>
                                                ${item.el.toLocaleString()}
                                            </td>
                                            <td className="py-3">{item.conviction}</td>
                                            <td className="py-3">
                                                <span className={`px-2 py-1 rounded text-xs ${
                                                    item.status === 'HITL_TIER_3' ? 'bg-cyber-danger/20 text-cyber-danger border border-cyber-danger/50' :
                                                    item.status === 'AUTOMATED' ? 'bg-cyber-success/20 text-cyber-success border border-cyber-success/50' :
                                                    'bg-cyber-warning/20 text-cyber-warning border border-cyber-warning/50'
                                                }`}>
                                                    {item.status}
                                                </span>
                                            </td>
                                            <td className="py-3">
                                                <button
                                                    onClick={() => handleAction(item)}
                                                    className={`px-3 py-1 text-xs rounded border ${
                                                        item.status === 'HITL_TIER_3' ? 'border-cyber-danger text-cyber-danger hover:bg-cyber-danger/10' :
                                                        'border-cyber-accent text-cyber-accent hover:bg-cyber-accent/10'
                                                    }`}
                                                >
                                                    REVIEW
                                                </button>
                                            </td>
                                        </tr>
                                    ))}
                                    {feed.length === 0 && (
                                        <tr>
                                            <td colSpan="6" className="py-4 text-center text-gray-500 italic">Waiting for data...</td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                {/* Right Column: DAG */}
                <div className="space-y-6">
                    <div className="panel p-4">
                        <h2 className="text-lg font-mono text-white mb-4 border-b border-cyber-border pb-2">PROOF OF THOUGHT (DAG)</h2>
                        <RegulatoryDAG />
                        <div className="mt-4 text-xs font-mono text-gray-400">
                            <p className="mb-1">Immutable Ledger Sync: <span className="text-cyber-success">ACTIVE</span></p>
                            <p>All nodes cryptographically signed.</p>
                        </div>
                    </div>
                </div>
            </div>

            {modalData && (
                <FrictionModal
                    data={modalData}
                    onClose={() => setModalData(null)}
                    onAuthorize={(rationale) => {
                        alert(`Authorization Confirmed.\nRationale: ${rationale}\nHash: ${modalData.hash}`);
                        setModalData(null);
                    }}
                />
            )}
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<SentinelDashboard />);
