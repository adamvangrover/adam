/**
 * ADAM v24.0 State Manager
 * -----------------------------------------------------------------------------
 * Provides centralized state management with persistence capabilities.
 * Designed to handle UI state (tabs, filters, toggles) across sessions.
 *
 * Features:
 * - Reactive state (subscribe/notify)
 * - LocalStorage persistence (optional)
 * - Default values
 * -----------------------------------------------------------------------------
 */

class StateManager {
    constructor() {
        this.state = {};
        this.subscribers = {};
        this.persistentKeys = new Set();
        this.prefix = 'ADAM_STATE_';
    }

    /**
     * Initialize a state key with a default value.
     * Loads from localStorage if 'persist' is true and value exists.
     * @param {string} key - The state key.
     * @param {any} defaultValue - Default value.
     * @param {boolean} persist - Whether to save to localStorage.
     */
    init(key, defaultValue, persist = false) {
        if (persist) {
            this.persistentKeys.add(key);
            const saved = localStorage.getItem(this.prefix + key);
            if (saved !== null) {
                try {
                    this.state[key] = JSON.parse(saved);
                } catch (e) {
                    console.warn(`[StateManager] Failed to parse saved state for ${key}`, e);
                    this.state[key] = defaultValue;
                }
            } else {
                this.state[key] = defaultValue;
            }
        } else {
            this.state[key] = defaultValue;
        }

        // Ensure subscriber array exists
        if (!this.subscribers[key]) {
            this.subscribers[key] = [];
        }
    }

    /**
     * Get the current value of a state key.
     * @param {string} key
     */
    getState(key) {
        return this.state[key];
    }

    /**
     * Update the state.
     * @param {string} key
     * @param {any} value
     */
    setState(key, value) {
        if (this.state[key] === value) return;

        this.state[key] = value;

        if (this.persistentKeys.has(key)) {
            try {
                localStorage.setItem(this.prefix + key, JSON.stringify(value));
            } catch (e) {
                console.warn(`[StateManager] Failed to save state for ${key}`, e);
            }
        }

        this.notify(key, value);
    }

    /**
     * Subscribe to changes for a specific key.
     * @param {string} key
     * @param {function} callback
     */
    subscribe(key, callback) {
        if (!this.subscribers[key]) {
            this.subscribers[key] = [];
        }
        this.subscribers[key].push(callback);
        // Call immediately with current value if exists
        if (this.state[key] !== undefined) {
            callback(this.state[key]);
        }
    }

    notify(key, value) {
        if (this.subscribers[key]) {
            this.subscribers[key].forEach(cb => cb(value));
        }
    }

    /**
     * Reset a specific key to default (removes from storage if persistent)
     */
    reset(key, defaultValue) {
         if (this.persistentKeys.has(key)) {
            localStorage.removeItem(this.prefix + key);
         }
         this.state[key] = defaultValue;
         this.notify(key, defaultValue);
    }
}

// Global Instance
window.stateManager = new StateManager();
