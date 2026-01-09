import React from 'react';
import { render, screen } from '@testing-library/react';
import { ConvictionMeter } from './ConvictionMeter';

describe('ConvictionMeter Component', () => {
    const defaultProps = {
        score: 7,
        reasoning: ['Strong revenue growth', 'Solid management team']
    };

    test('renders score text correctly', () => {
        render(<ConvictionMeter {...defaultProps} />);
        expect(screen.getByText('7')).toBeInTheDocument();
        expect(screen.getByText('/ 10')).toBeInTheDocument();
    });

    test('renders reasoning trace items', () => {
        render(<ConvictionMeter {...defaultProps} />);
        expect(screen.getByText('Strong revenue growth')).toBeInTheDocument();
        expect(screen.getByText('Solid management team')).toBeInTheDocument();
    });

    test('has accessible meter attributes', () => {
        render(<ConvictionMeter {...defaultProps} />);

        const meter = screen.getByRole('progressbar', { name: /conviction score/i });

        expect(meter).toBeInTheDocument();
        expect(meter).toHaveAttribute('aria-valuenow', '7');
        expect(meter).toHaveAttribute('aria-valuemin', '0');
        expect(meter).toHaveAttribute('aria-valuemax', '10');
    });
});
