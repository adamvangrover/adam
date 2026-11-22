import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import GlobalNav from './GlobalNav';
import { dataManager } from '../utils/DataManager';

// Mock dependencies
vi.mock('react-router-dom', () => ({
  useNavigate: () => vi.fn(),
}));

vi.mock('../utils/DataManager', () => ({
  dataManager: {
    checkConnection: vi.fn(),
    toggleSimulationMode: vi.fn(),
  },
}));

vi.mock('lucide-react', () => ({
  Loader2: () => <div data-testid="spinner">Spinner</div>,
}));

describe('GlobalNav Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Mock DataManager
    (dataManager.checkConnection as ReturnType<typeof vi.fn>).mockResolvedValue({ status: 'ONLINE' });

    // Mock global fetch
    global.fetch = vi.fn(() =>
        Promise.resolve({
            json: () => Promise.resolve({ mode: 'LIVE' }),
            ok: true,
            status: 200,
        })
    ) as ReturnType<typeof vi.fn>;
  });

  afterEach(() => {
    // Clean up fetch mock if needed, though beforeEach handles it
    vi.restoreAllMocks();
  });

  test('toggle mode shows loading state and disables button', async () => {
    let resolveCheck: (value: any) => void;
    const checkPromise = new Promise(resolve => {
        resolveCheck = resolve;
    });

    (dataManager.checkConnection as ReturnType<typeof vi.fn>).mockResolvedValue({ status: 'ONLINE' });

    await act(async () => {
      render(<GlobalNav />);
    });

    // We start in LIVE mode (based on default mock).
    // Click to go to ARCHIVE
    const toggleButton = screen.getByRole('button', { name: /Switch to Archive Mode/i });
    fireEvent.click(toggleButton);

    await waitFor(() => {
        expect(screen.getByText(/ARCHIVE MODE/i)).toBeInTheDocument();
    });

    // Now mock the connection check to hang
    (dataManager.checkConnection as ReturnType<typeof vi.fn>).mockReturnValue(checkPromise);

    // Click to go back to LIVE
    const archiveButton = screen.getByRole('button', { name: /Switch to Live Mode/i });
    fireEvent.click(archiveButton);

    // Expect Loading State
    expect(screen.getByText(/CHECKING.../i)).toBeInTheDocument();
    expect(screen.getByTestId('spinner')).toBeInTheDocument();
    expect(archiveButton).toBeDisabled();

    // Resolve promise
    await act(async () => {
        resolveCheck!({ status: 'ONLINE' });
    });

    // Should be LIVE now
    expect(screen.getByText(/LIVE MODE/i)).toBeInTheDocument();
    expect(screen.queryByText(/CHECKING.../i)).not.toBeInTheDocument();
  });

  test('search input has unique aria-label', async () => {
    await act(async () => {
      render(<GlobalNav />);
    });
    const inputs = screen.getAllByLabelText(/Global Search/i);
    expect(inputs).toHaveLength(1);
  });
});
