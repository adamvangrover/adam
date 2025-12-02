/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        slate: {
          800: '#1e293b',
          900: '#0f172a',
          950: '#020617',
        },
        cyan: {
          500: '#06b6d4',
          400: '#22d3ee',
        },
        emerald: {
          500: '#10b981',
          400: '#34d399',
        },
        amber: {
          500: '#f59e0b',
          400: '#fbbf24',
        },
        rose: {
          500: '#f43f5e',
          400: '#fb7185',
        },
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', "Liberation Mono", "Courier New", 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      backgroundImage: {
        'grid-pattern': "linear-gradient(to right, #334155 1px, transparent 1px), linear-gradient(to bottom, #334155 1px, transparent 1px)",
      },
    },
  },
  plugins: [],
}
