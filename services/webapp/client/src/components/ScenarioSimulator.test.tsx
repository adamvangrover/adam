import React from 'react';
import { render, fireEvent, act } from '@testing-library/react';
import { ScenarioSimulator } from './ScenarioSimulator';

jest.useFakeTimers();

describe('ScenarioSimulator', () => {
    it('debounces the onSimulate callback', () => {
        const handleSimulate = jest.fn();
        const { container } = render(<ScenarioSimulator onSimulate={handleSimulate} />);

        // Initial render triggers useEffect, but should be skipped by isMounted check
        act(() => {
            jest.runAllTimers();
        });
        expect(handleSimulate).not.toHaveBeenCalled();

        // Find input (Volatility is the first one)
        const inputs = container.querySelectorAll('input[type="range"]');
        const volatilityInput = inputs[0];

        // Change value
        fireEvent.change(volatilityInput, { target: { value: '0.30' } });

        // Should NOT be called immediately due to debounce
        expect(handleSimulate).not.toHaveBeenCalled();

        // Fast forward time
        act(() => {
            jest.advanceTimersByTime(500);
        });

        // Should be called now
        expect(handleSimulate).toHaveBeenCalledTimes(1);
        expect(handleSimulate).toHaveBeenCalledWith({ volatility: 0.3, stress: 0 });
    });
});
