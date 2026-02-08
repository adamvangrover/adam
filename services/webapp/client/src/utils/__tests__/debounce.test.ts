import { debounce } from '../debounce';

jest.useFakeTimers();

describe('debounce', () => {
  it('should execute the function after the specified wait time', () => {
    const func = jest.fn();
    const debouncedFunc = debounce(func, 1000);

    debouncedFunc();
    expect(func).not.toHaveBeenCalled();

    jest.advanceTimersByTime(500);
    expect(func).not.toHaveBeenCalled();

    jest.advanceTimersByTime(500);
    expect(func).toHaveBeenCalledTimes(1);
  });

  it('should reset the timer if called again within the wait time', () => {
    const func = jest.fn();
    const debouncedFunc = debounce(func, 1000);

    debouncedFunc();
    jest.advanceTimersByTime(500);
    debouncedFunc(); // Reset timer

    jest.advanceTimersByTime(500);
    expect(func).not.toHaveBeenCalled();

    jest.advanceTimersByTime(500);
    expect(func).toHaveBeenCalledTimes(1);
  });
});
