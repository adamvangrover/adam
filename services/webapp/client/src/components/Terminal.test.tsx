import React from 'react';
import { render, screen } from '@testing-library/react';
import Terminal from './Terminal';

// Mock the dataManager
jest.mock('../utils/DataManager', () => ({
  dataManager: {
    checkConnection: jest.fn(),
    getManifest: jest.fn(),
    isOfflineMode: jest.fn(),
    toggleSimulationMode: jest.fn(),
  },
}));

// Mock scrollIntoView since it's not implemented in JSDOM
window.HTMLElement.prototype.scrollIntoView = jest.fn();

test('renders terminal with accessible input and output', () => {
  render(<Terminal />);

  // Check for input with correct accessible name
  const input = screen.getByLabelText(/terminal command input/i);
  expect(input).toBeInTheDocument();
  expect(input).toHaveAttribute('placeholder', "Type 'help' for commands...");
  expect(input).toHaveAttribute('spellcheck', 'false');

  // Check for output log with role and aria-live
  const log = screen.getByRole('log');
  expect(log).toBeInTheDocument();
  expect(log).toHaveAttribute('aria-live', 'polite');
  expect(log).toHaveAttribute('aria-label', 'Terminal Output');
});
