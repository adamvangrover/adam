import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import ForceGraph2D from 'react-force-graph-2d';
import { getAuthHeaders } from './utils/auth';

const DetailsPanel = ({ data, onClear }) => {
    const { t } = useTranslation();
    return (
        <div className="Card" style={{ flex: 1, maxHeight: '80vh', overflowY: 'auto' }}>
            <button onClick={onClear}>{t('knowledgeGraph.close')}</button>
            <h3>{t('knowledgeGraph.details')}</h3>
            {data.type ? ( // It's a link
                <>
                    <h4>{t('knowledgeGraph.linkDetails')}</h4>
                    <p><strong>{t('knowledgeGraph.type')}:</strong> {data.type}</p>
                    <p><strong>{t('knowledgeGraph.source')}:</strong> {data.source.id}</p>
                    <p><strong>{t('knowledgeGraph.target')}:</strong> {data.target.id}</p>
                </>
            ) : ( // It's a node
                <>
                    <h4>{t('knowledgeGraph.nodeDetails')}</h4>
                    <p><strong>{t('knowledgeGraph.id')}:</strong> {data.id}</p>
                    <p><strong>{t('knowledgeGraph.labels')}:</strong> {data.labels.join(', ')}</p>
                    <h5>{t('knowledgeGraph.properties')}:</h5>
                    <pre>{JSON.stringify(data.properties, null, 2)}</pre>
                </>
            )}
        </div>
    );
}

function KnowledgeGraph() {
  const { t } = useTranslation();
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [selectedData, setSelectedData] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const fgRef = useRef();


  const fetchGraphData = useCallback(async (query = '') => {
    setLoading(true);
    const headers = await getAuthHeaders();
    const url = query ? `/api/knowledge_graph?query=${query}` : '/api/knowledge_graph';
    const response = await fetch(url, { headers });
    const data = await response.json();

    // Prevent duplicates
    const existingNodes = new Set(graphData.nodes.map(n => n.id));
    const newNodes = data.nodes.filter(n => !existingNodes.has(n.id));

    const existingLinks = new Set(graphData.links.map(l => `${l.source.id}-${l.target.id}`));
    const newLinks = data.links.filter(l => !existingLinks.has(`${l.source}-${l.target}`));


    setGraphData(prevData => ({
        nodes: [...prevData.nodes, ...newNodes],
        links: [...prevData.links, ...newLinks]
    }));
    setLoading(false);
  }, [graphData]);

  useEffect(() => {
    fetchGraphData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Initial fetch only

  const handleSearch = (e) => {
    e.preventDefault();
    setGraphData({ nodes: [], links: [] }); // Clear existing graph for new search
    fetchGraphData(searchQuery);
  };

  const getNodeColor = (node) => {
    if (!node.labels) return 'red';
    if (node.labels.includes('Company')) return 'blue';
    if (node.labels.includes('Person')) return 'green';
    return 'red';
  };

  const handleNodeClick = (node) => {
    // Center view on node
    fgRef.current.centerAt(node.x, node.y, 1000);
    fgRef.current.zoom(2.5, 1000);
    setSelectedData(node);
  };

  const handleLinkClick = (link) => {
      setSelectedData(link);
  }

  return (
    <div>
      <h2>{t('knowledgeGraph.title')}</h2>
      <div className="Card">
          <form onSubmit={handleSearch}>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder={t('knowledgeGraph.searchPlaceholder')}
              />
              <button type="submit">{t('knowledgeGraph.search')}</button>
              <button type="button" onClick={() => { setSearchQuery(''); setGraphData({ nodes: [], links: [] }); fetchGraphData(); }}>{t('knowledgeGraph.reset')}</button>
          </form>
      </div>
      <div style={{ display: 'flex' }}>
        <div className="Card" style={{ flex: 3, position: 'relative' }}>
          {loading && <p style={{position: 'absolute', top: '50%', left: '50%'}}>{t('analysisTools.loading')}</p>}
          <ForceGraph2D
            ref={fgRef}
            graphData={graphData}
            nodeLabel="id"
            nodeColor={getNodeColor}
            linkColor={() => 'rgba(100, 100, 100, 0.5)'}
            linkDirectionalArrowLength={3.5}
            linkDirectionalArrowRelPos={1}
            linkLabel="type"
            onNodeClick={handleNodeClick}
            onLinkClick={handleLinkClick}
          />
        </div>
        {selectedData && (
          <DetailsPanel data={selectedData} onClear={() => setSelectedData(null)} />
        )}
      </div>
    </div>
  );
}

export default KnowledgeGraph;
