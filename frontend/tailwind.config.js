/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        aeris: {
          bg: '#0b1329',
          card: '#131e3a',
          border: '#1e2d54',
          accent: '#00d2ff',
          watch: '#eab308',
          moderate: '#f97316',
          severe: '#ef4444',
        }
      }
    },
  },
  plugins: [],
}
