import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'AERIS — AI Extreme Weather Intelligence System',
  description: 'SIH 2026 Problem Statement 26078: AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies in Medium-Range Forecasts.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased bg-aeris-bg min-h-screen flex flex-col overflow-x-hidden">
        {children}
      </body>
    </html>
  );
}
