import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import AuditLogViewer from '../AuditLogViewer';

// Mock fetch
global.fetch = jest.fn();

describe('AuditLogViewer', () => {
    beforeEach(() => {
        fetch.mockClear();
    });

    test('renders loading state initially', async () => {
        // Mock fetch that hangs to test loading state
        fetch.mockImplementationOnce(() => new Promise(() => {}));
        render(<AuditLogViewer />);
        expect(screen.getByText(/Loading Neural Audit Logs.../i)).toBeInTheDocument();
    });

    test('renders traces when data is fetched', async () => {
        const mockTraces = {
            traces: [
                {
                    trace_id: '123',
                    session_id: 'session-1',
                    start_time: '2023-01-01T12:00:00Z',
                    events: [
                        {
                            component: 'RedTeam',
                            event_type: 'ATTACK',
                            payload: { analysis: 'Testing attack' }
                        }
                    ]
                }
            ]
        };

        fetch.mockResolvedValue({
            ok: true,
            json: async () => mockTraces,
        });

        await act(async () => {
            render(<AuditLogViewer />);
        });

        await waitFor(() => {
            expect(screen.getByText('SYSTEM_AUDIT_LOG::V26.0')).toBeInTheDocument();
            expect(screen.getByText('SESSION: session-1')).toBeInTheDocument();
            expect(screen.getByText('[RedTeam]')).toBeInTheDocument();
            expect(screen.getByText('ATTACK')).toBeInTheDocument();
        });
    });

    test('renders empty state if no traces', async () => {
        fetch.mockResolvedValue({
            ok: true,
            json: async () => ({ traces: [] }),
        });

        await act(async () => {
            render(<AuditLogViewer />);
        });

        // Should show header but no traces
        await waitFor(() => {
             expect(screen.getByText('SYSTEM_AUDIT_LOG::V26.0')).toBeInTheDocument();
        });
        expect(screen.queryByText('SESSION:')).not.toBeInTheDocument();
    });
});
