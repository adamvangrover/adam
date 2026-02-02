import React from 'react';
import { render, screen } from '@testing-library/react';
import AgentStatus from './AgentStatus';
import { dataManager } from '../utils/DataManager';

// Mock the DataManager
jest.mock('../utils/DataManager', () => ({
  dataManager: {
    getManifest: jest.fn(),
  },
}));

describe('AgentStatus Component', () => {
  beforeEach(() => {
    (dataManager.getManifest as jest.Mock).mockResolvedValue({
      agents: [
        { id: '1', name: 'Test Agent', status: 'Active', specialization: 'Testing' },
      ],
      reports: []
    });
  });

  test('renders the filter input with accessible label', async () => {
    render(<AgentStatus />);

    // Use findByText to wait for the element to appear
    await screen.findByText('Test Agent');

    // This should fail initially because the input lacks the aria-label
    // we use getByLabelText to assert accessibility
    const input = screen.getByLabelText('Filter agents');
    expect(input).toBeInTheDocument();
  });
});
