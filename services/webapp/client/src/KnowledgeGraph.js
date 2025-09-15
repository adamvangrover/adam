import React, { useState, useEffect } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

function KnowledgeGraph() {
  const [data, setData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [selectedNode, setSelectedNode] = useState(null);

  useEffect(() => {
    fetch('/api/knowledge_graph')
      .then(res => res.json())
      .then(data => {
        setData(data);
        setLoading(false);
      });
  }, []);

  const getNodeColor = (node) => {
    if (node.labels.includes('Company')) return 'blue';
    if (node.labels.includes('Person')) return 'green';
    return 'red';
  };

  const handleNodeClick = (node) => {
    setSelectedNode(node);
  };

  return (
    <div>
      <h2>Knowledge Graph</h2>
      <div style={{ display: 'flex' }}>
        <div className="Card" style={{ flex: 3 }}>
          {loading ? <p>Loading...</p> : (
            <ForceGraph2D
              graphData={data}
              nodeLabel={node => `${node.id} (${node.labels.join(', ')})`}
              nodeColor={getNodeColor}
              linkColor={() => 'gray'}
              linkDirectionalArrowLength={3.5}
              linkDirectionalArrowRelPos={1}
              linkLabel="type"
              onNodeClick={handleNodeClick}
            />
          )}
        </div>
        {selectedNode && (
          <div className="Card" style={{ flex: 1 }}>
            <h3>Node Details</h3>
            <pre>{JSON.stringify(selectedNode, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
}

export default KnowledgeGraph;
