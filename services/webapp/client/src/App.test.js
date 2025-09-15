import { render, screen } from '@testing-library/react';
import App from './App';

test('renders app header', () => {
  render(<App />);
  const linkElement = screen.getByText(/Adam v19.2/i);
  expect(linkElement).toBeInTheDocument();
});
