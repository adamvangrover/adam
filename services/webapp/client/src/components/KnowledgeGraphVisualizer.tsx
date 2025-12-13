import React, { useMemo } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

// Placeholder types if we don't have full type defs for the library
interface GraphNode {
    id: string;
    group: number;
    val: number;
    name?: string;
}

interface GraphLink {
    source: string;
    target: string;
}

interface VisualizerProps {
    data: any; // Using any for the HDKG schema input for flexibility
}

export const KnowledgeGraphVisualizer: React.FC<VisualizerProps> = ({ data }) => {
    // Bolt: Memoize graph data transformation to prevent unnecessary recalculations on re-renders
    const graphData = useMemo(() => {
        const nodes: GraphNode[] = [];
        const links: GraphLink[] = [];

        if (data && data.v23_knowledge_graph && data.v23_knowledge_graph.nodes) {
            const root = data.v23_knowledge_graph.nodes;

            // Target Node
            nodes.push({ id: "Target", group: 1, val: 20, name: data.v23_knowledge_graph.meta.target });

            // Entity Ecosystem
            if (root.entity_ecosystem) {
                nodes.push({ id: "Entity", group: 2, val: 10, name: "Entity Ecosystem" });
                links.push({ source: "Target", target: "Entity" });

                if (root.entity_ecosystem.management_assessment) {
                    nodes.push({ id: "Management", group: 3, val: 5, name: "Management" });
                    links.push({ source: "Entity", target: "Management" });
                }
            }

            // Equity Analysis
            if (root.equity_analysis) {
                nodes.push({ id: "Equity", group: 2, val: 10, name: "Equity Analysis" });
                links.push({ source: "Target", target: "Equity" });

                if (root.equity_analysis.valuation_engine?.dcf_model) {
                    nodes.push({ id: "DCF", group: 3, val: 5, name: `DCF: ${root.equity_analysis.valuation_engine.dcf_model.intrinsic_value}` });
                    links.push({ source: "Equity", target: "DCF" });
                }
            }

            // Credit Analysis
            if (root.credit_analysis) {
                nodes.push({ id: "Credit", group: 2, val: 10, name: "Credit Analysis" });
                links.push({ source: "Target", target: "Credit" });

                if (root.credit_analysis.snc_rating_model) {
                    nodes.push({ id: "SNC", group: 3, val: 5, name: `Rating: ${root.credit_analysis.snc_rating_model.overall_borrower_rating}` });
                    links.push({ source: "Credit", target: "SNC" });
                }
            }

            // Simulation
            if (root.simulation_engine) {
                nodes.push({ id: "Risk", group: 2, val: 10, name: "Simulation Engine" });
                links.push({ source: "Target", target: "Risk" });
            }
        }
        return { nodes, links };
    }, [data]);

    return (
        <div style={{ height: '500px', border: '1px solid #ccc', borderRadius: '8px', overflow: 'hidden' }}>
             {graphData.nodes.length > 0 ? (
                 <ForceGraph2D
                    graphData={graphData}
                    nodeAutoColorBy="group"
                    nodeLabel="name"
                    enableNodeDrag={true}
                 />
             ) : (
                 <div style={{ padding: '20px', textAlign: 'center' }}>No Graph Data Available</div>
             )}
        </div>
    );
};
