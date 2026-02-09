import React, { useEffect, useState, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { dataManager } from '../utils/DataManager';
import { debounce } from '../utils/debounce';

// Helper to generate mock graph
const generateMockGraph = (n = 20) => {
    const nodes = Array.from({ length: n }, (_, i) => ({ id: i, name: `Node ${i}`, val: Math.random() * 5 + 1 }));
    const links = Array.from({ length: n * 2 }, () => ({
        source: Math.floor(Math.random() * n),
        target: Math.floor(Math.random() * n)
    }));
    return { nodes, links };
};

const KnowledgeGraph: React.FC = () => {
  const [graphData, setGraphData] = useState<any>({ nodes: [], links: [] });
  const [dimensions, setDimensions] = useState({ w: 800, h: 600 });
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Resize handler
    const handleResize = () => {
        if (containerRef.current) {
            setDimensions({
                w: containerRef.current.offsetWidth,
                h: containerRef.current.offsetHeight
            });
        }
    };

    // Bolt Optimization: Debounce resize to prevent canvas thrashing
    const debouncedResize = debounce(handleResize, 200);

    window.addEventListener('resize', debouncedResize);
    handleResize();

    // Fetch Data
    const loadData = async () => {
        try {
            // Try to get real KG
            const data = await dataManager.getData('/knowledge_graph');
            if (data && data.nodes && data.nodes.length > 0) {
                setGraphData(data);
            } else {
                // Fallback to mock
                setGraphData(generateMockGraph(50));
            }
        } catch (e) {
            setGraphData(generateMockGraph(50));
        }
    };
    loadData();

    return () => window.removeEventListener('resize', debouncedResize);
  }, []);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: '0 20px 20px 20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', height: '60px' }}>
        <h2 className="text-cyan mono-font">{'///'} KNOWLEDGE GRAPH VISUALIZER</h2>
        <div style={{ fontSize: '0.8rem', color: '#666' }}>
            NODES: {graphData.nodes.length} | EDGES: {graphData.links.length}
        </div>
      </div>

      <div ref={containerRef} className="cyber-panel" style={{ flexGrow: 1, overflow: 'hidden', position: 'relative' }}>
        <ForceGraph2D
            width={dimensions.w}
            height={dimensions.h}
            graphData={graphData}
            nodeLabel="name"
            nodeColor={() => '#00f3ff'}
            linkColor={() => 'rgba(0, 243, 255, 0.2)'}
            backgroundColor="#050b14"
            nodeRelSize={4}
        />
        <div style={{ position: 'absolute', bottom: '10px', right: '10px', fontSize: '0.7rem', color: '#444' }}>
            POWERED BY FORCE-GRAPH-2D
        </div>
      </div>
    </div>
  );
};

export default KnowledgeGraph;
