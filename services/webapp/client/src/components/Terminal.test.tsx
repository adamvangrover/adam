import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
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

test('handles command history navigation', () => {
  render(<Terminal />);
  const input = screen.getByLabelText(/terminal command input/i);

  // Type and submit 'command1'
  fireEvent.change(input, { target: { value: 'command1' } });
  fireEvent.keyDown(input, { key: 'Enter', code: 'Enter', charCode: 13 });

  // Type and submit 'command2'
  fireEvent.change(input, { target: { value: 'command2' } });
  fireEvent.keyDown(input, { key: 'Enter', code: 'Enter', charCode: 13 });

  // Press ArrowUp -> command2
  fireEvent.keyDown(input, { key: 'ArrowUp', code: 'ArrowUp' });
  expect(input).toHaveValue('command2');

  // Press ArrowUp -> command1
  fireEvent.keyDown(input, { key: 'ArrowUp', code: 'ArrowUp' });
  expect(input).toHaveValue('command1');

  // Press ArrowDown -> command2
  fireEvent.keyDown(input, { key: 'ArrowDown', code: 'ArrowDown' });
  expect(input).toHaveValue('command2');

  // Press ArrowDown -> empty
  fireEvent.keyDown(input, { key: 'ArrowDown', code: 'ArrowDown' });
  expect(input).toHaveValue('');
});

test('handles tab autocomplete', () => {
  render(<Terminal />);
  const input = screen.getByLabelText(/terminal command input/i);

  // Type 'st' and Tab
  fireEvent.change(input, { target: { value: 'st' } });
  fireEvent.keyDown(input, { key: 'Tab', code: 'Tab' });
  expect(input).toHaveValue('status');

  // Type 'scan' and Tab
  fireEvent.change(input, { target: { value: 'scan' } });
  fireEvent.keyDown(input, { key: 'Tab', code: 'Tab' });
  expect(input).toHaveValue('scan agents');
});
