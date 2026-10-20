// Proctorify Authentication Helper
// Handles token storage, API calls, and auth state across all pages.

const TOKEN_KEY = 'proctorify_token';
const USER_KEY = 'proctorify_user';

// --- Token management ---

function getToken() {
    return localStorage.getItem(TOKEN_KEY);
}

function setToken(token, userData) {
    localStorage.setItem(TOKEN_KEY, token);
    if (userData) {
        localStorage.setItem(USER_KEY, JSON.stringify(userData));
    }
}

function clearToken() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
}

function getStoredUser() {
    const data = localStorage.getItem(USER_KEY);
    return data ? JSON.parse(data) : null;
}

// --- API call helper ---

async function apiCall(url, options = {}) {
    const token = getToken();
    const headers = {
        'Content-Type': 'application/json',
        ...options.headers
    };
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(url, {
        ...options,
        headers
    });

    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || `Request failed: ${response.status}`);
    }

    return data;
}

// --- Auth checks ---

async function requireAuth() {
    // Redirect to login if not authenticated
    const token = getToken();
    if (!token) {
        window.location.href = '/';
        return null;
    }

    try {
        const user = await apiCall('/api/me');
        return user;
    } catch (e) {
        // Token is invalid or expired
        clearToken();
        window.location.href = '/';
        return null;
    }
}

async function redirectIfAuth() {
    // Redirect to lobby if already logged in (used on login/register pages)
    const token = getToken();
    if (!token) return;

    try {
        await apiCall('/api/me');
        window.location.href = '/lobby';
    } catch (e) {
        // Token invalid, stay on page
        clearToken();
    }
}

// --- Logout ---

async function logout() {
    try {
        await apiCall('/api/logout', { method: 'POST' });
    } catch (e) {
        // Ignore errors, clear local state anyway
    }
    clearToken();
    window.location.href = '/';
}

// --- Theme management ---

function applyTheme(theme) {
    const htmlEl = document.documentElement;
    if (theme === 'light') {
        htmlEl.setAttribute('data-theme', 'light');
    } else {
        htmlEl.removeAttribute('data-theme');
    }
    localStorage.setItem('proctorify_theme', theme);
}

function initTheme() {
    const savedTheme = localStorage.getItem('proctorify_theme') || 'dark';
    applyTheme(savedTheme);
    return savedTheme;
}

function toggleTheme() {
    const current = localStorage.getItem('proctorify_theme') || 'dark';
    const newTheme = current === 'dark' ? 'light' : 'dark';
    applyTheme(newTheme);
    return newTheme;
}