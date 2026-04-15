import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import EvalDashboard from './EvalDashboard';

describe('EvalDashboard', () => {
  it('renders dashboard title', () => {
    render(<EvalDashboard />);
    expect(screen.getByText('Adam v30 Telemetry Dashboard')).toBeInTheDocument();
  });
});
