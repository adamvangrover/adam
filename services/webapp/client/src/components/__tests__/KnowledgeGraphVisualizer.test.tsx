import React from 'react';
import { render, screen } from '@testing-library/react';
import { KnowledgeGraphVisualizer } from '../KnowledgeGraphVisualizer';

// Mock react-force-graph-2d
jest.mock('react-force-graph-2d', () => {
    return function DummyGraph(props: any) {
        return (
            <div data-testid="force-graph">
                {JSON.stringify(props.graphData)}
            </div>
        );
    };
});

const mockData = {
    v23_knowledge_graph: {
        meta: { target: "Apple Inc" },
        nodes: {
            entity_ecosystem: {
                management_assessment: { CEO: "Tim Cook" }
            },
            equity_analysis: {
                valuation_engine: {
                    dcf_model: { intrinsic_value: 150.00 }
                }
            },
            credit_analysis: {
                snc_rating_model: { overall_borrower_rating: "Pass" }
            },
            simulation_engine: {}
        }
    }
};

describe('KnowledgeGraphVisualizer', () => {
    it('renders without crashing and transforms data', () => {
        render(<KnowledgeGraphVisualizer data={mockData} />);
        const graphElement = screen.getByTestId('force-graph');
        expect(graphElement).toBeInTheDocument();

        const graphData = JSON.parse(graphElement.textContent || "{}");
        // Expected nodes: Target(1) + Entity(1)+Management(1) + Equity(1)+DCF(1) + Credit(1)+SNC(1) + Risk(1) = 8
        expect(graphData.nodes).toHaveLength(8);
        expect(graphData.links).toHaveLength(7);
    });

    it('renders empty state when data is missing', () => {
        render(<KnowledgeGraphVisualizer data={{}} />);
        expect(screen.getByText('No Graph Data Available')).toBeInTheDocument();
    });
});
