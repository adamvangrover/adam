import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import AgentIntercom from './AgentIntercom';

// Mock fetch globally
global.fetch = jest.fn();

// Mock localStorage
const localStorageMock = (function() {
  let store: any = {};
  return {
    getItem: function(key: string) {
      return store[key] || null;
    },
    setItem: function(key: string, value: string) {
      store[key] = value.toString();
    },
    clear: function() {
      store = {};
    },
    removeItem: function(key: string) {
      delete store[key];
    }
  };
})();
Object.defineProperty(window, 'localStorage', { value: localStorageMock });

// Mock scrollIntoView since it's not implemented in jsdom
Element.prototype.scrollIntoView = jest.fn();

describe('AgentIntercom', () => {
  beforeEach(() => {
    (global.fetch as jest.Mock).mockClear();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('renders initial loading state', () => {
    // Mock fetch to return pending promise
    (global.fetch as jest.Mock).mockReturnValue(new Promise(() => {}));

    render(<AgentIntercom />);
    expect(screen.getByText(/ESTABLISHING NEURAL LINK/i)).toBeInTheDocument();
  });

  test('fetches and renders thoughts', async () => {
    const mockThoughts = [
      { id: '1', text: 'Thought 1' },
      { id: '2', text: 'Thought 2' }
    ];

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockThoughts
    });

    render(<AgentIntercom />);

    await waitFor(() => {
        expect(screen.getByText('Thought 1')).toBeInTheDocument();
        expect(screen.getByText('Thought 2')).toBeInTheDocument();
    });
  });

  test('updates with new thoughts', async () => {
      const initialThoughts = [{ id: '1', text: 'Initial Thought' }];
      const newThoughts = [{ id: '2', text: 'New Thought' }, { id: '1', text: 'Initial Thought' }];

      (global.fetch as jest.Mock)
          .mockResolvedValueOnce({
              ok: true,
              json: async () => initialThoughts
          })
          .mockResolvedValue({ // Subsequent calls
              ok: true,
              json: async () => newThoughts
          });

      render(<AgentIntercom />);

      // First render
      await waitFor(() => {
          expect(screen.getByText('Initial Thought')).toBeInTheDocument();
      });

      // Advance timers to trigger next fetch (2000ms)
      // Since it uses recursive setTimeout, verifying update requires triggering it
      act(() => {
          jest.advanceTimersByTime(2500); // 2000ms timeout + buffer
      });

      // Wait for update
      await waitFor(() => {
          expect(screen.getByText('New Thought')).toBeInTheDocument();
      });
  });
});
