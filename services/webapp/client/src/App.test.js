import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import * as auth from './utils/auth';
import App from './App';
import translationEN from './locales/en/translation.json';

// Mocks to handle ESM modules and side effects
jest.mock('react-force-graph-2d', () => {
  return function ForceGraph2D() {
    return <div>ForceGraph2D</div>;
  };
});

jest.mock('socket.io-client', () => {
  const mSocket = {
    on: jest.fn(),
    off: jest.fn(),
    emit: jest.fn(),
  };
  return jest.fn(() => mSocket);
});

jest.mock('react-chartjs-2', () => ({
  Line: () => null,
  Bar: () => null,
}));

jest.mock('chart.js', () => ({
  Chart: { register: jest.fn() },
  CategoryScale: jest.fn(),
  LinearScale: jest.fn(),
  PointElement: jest.fn(),
  LineElement: jest.fn(),
  Title: jest.fn(),
  Tooltip: jest.fn(),
  Legend: jest.fn(),
  BarElement: jest.fn(),
}));

// Mock i18next to use the actual translation file
jest.mock('react-i18next', () => ({
  useTranslation: () => {
    const translation = require('./locales/en/translation.json');
    return {
      t: (key) => {
          const parts = key.split('.');
          let val = translation;
          for (const part of parts) {
              val = val && val[part];
          }
          return val || key;
      },
      i18n: {
        changeLanguage: jest.fn(),
      },
    };
  },
}));

test('renders app header', async () => {
  jest.spyOn(auth, 'getToken').mockReturnValue('fake-token');
  jest.spyOn(auth, 'logout').mockImplementation(() => {});
  // getAuthHeaders is not used in this test but good to have

  render(
    <MemoryRouter>
      <App />
    </MemoryRouter>
  );

  await waitFor(() => {
      const linkElement = screen.getByText(/Adam v22.0/i);
      expect(linkElement).toBeInTheDocument();
  });
});
