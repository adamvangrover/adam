import { jwtDecode } from 'jwt-decode';

const API_URL = '/api';

export const getToken = () => {
    return localStorage.getItem('token');
};

export const getRefreshToken = () => {
    return localStorage.getItem('refreshToken');
}

export const setToken = (token, refreshToken) => {
    localStorage.setItem('token', token);
    if (refreshToken) {
        localStorage.setItem('refreshToken', refreshToken);
    }
};

export const removeToken = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('refreshToken');
};

export const login = async (username, password) => {
    const response = await fetch(`${API_URL}/login`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
    });

    if (!response.ok) {
        throw new Error('Failed to login');
    }

    const data = await response.json();
    setToken(data.access_token, data.refresh_token);
};

export const logout = async () => {
    const token = getToken();
    try {
        await fetch(`${API_URL}/logout`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            }
        });
    } finally {
        removeToken();
    }
};

export const isTokenExpired = (token) => {
    if (!token) {
        return true;
    }
    try {
        const decoded = jwtDecode(token);
        const now = Date.now() / 1000;
        return decoded.exp < now;
    } catch (error) {
        return true;
    }
};

export const refreshToken = async () => {
    const rToken = getRefreshToken();
    if (!rToken) {
        throw new Error("No refresh token available");
    }

    try {
        const response = await fetch(`${API_URL}/refresh`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${rToken}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to refresh token');
        }

        const data = await response.json();
        setToken(data.access_token);
        return data.access_token;
    } catch (error) {
        removeToken(); // Logout the user if refresh fails
        throw error;
    }
};

export const getAuthHeaders = async () => {
    let token = getToken();
    if (isTokenExpired(token)) {
        try {
            token = await refreshToken();
        } catch (error) {
            console.error("Session expired, please log in again.");
            removeToken();
            window.location.href = '/login'; // Force redirect to login
            return {};
        }
    }

    if (token) {
        return {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
        };
    }
    return {
        'Content-Type': 'application/json',
    };
};
