import { debounce } from '../debounce';

vi.useFakeTimers();

describe('debounce', () => {
  it('should execute the function after the specified wait time', () => {
    const func = vi.fn();
    const debouncedFunc = debounce(func, 1000);

    debouncedFunc();
    expect(func).not.toHaveBeenCalled();

    vi.advanceTimersByTime(500);
    expect(func).not.toHaveBeenCalled();

    vi.advanceTimersByTime(500);
    expect(func).toHaveBeenCalledTimes(1);
  });

  it('should reset the timer if called again within the wait time', () => {
    const func = vi.fn();
    const debouncedFunc = debounce(func, 1000);

    debouncedFunc();
    vi.advanceTimersByTime(500);
    debouncedFunc(); // Reset timer

    vi.advanceTimersByTime(500);
    expect(func).not.toHaveBeenCalled();

    vi.advanceTimersByTime(500);
    expect(func).toHaveBeenCalledTimes(1);
  });
});
