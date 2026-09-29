'use client';

import React from 'react';
import { SystemHealth, DatasetMetadata } from '../lib/api';
import { Server, Database, Layers, CheckCircle2, XCircle } from 'lucide-react';

interface SystemStatusFooterProps {
  health: SystemHealth | null;
  metadata: DatasetMetadata | null;
  totalEvents: number;
}

export default function SystemStatusFooter({ health, metadata, totalEvents }: SystemStatusFooterProps) {
  const isOnline = health?.status === 'ok';

  return (
    <footer className="w-full bg-aeris-card/95 border-t border-aeris-border px-6 py-2.5 flex flex-wrap items-center justify-between text-xs text-gray-300 gap-4 shadow-2xl font-mono">
      {/* API Health Status */}
      <div className="flex items-center gap-2">
        <Server className="w-4 h-4 text-aeris-accent" />
        <span className="text-gray-400">FastAPI Backend:</span>
        {isOnline ? (
          <span className="flex items-center gap-1 text-emerald-400 font-bold">
            <CheckCircle2 className="w-3.5 h-3.5" /> ONLINE (v{health?.version || '0.1.0'})
          </span>
        ) : (
          <span className="flex items-center gap-1 text-red-400 font-bold">
            <XCircle className="w-3.5 h-3.5" /> DISCONNECTED
          </span>
        )}
      </div>

      {/* Dataset & Metadata Info */}
      <div className="flex flex-wrap items-center gap-6 text-[11px]">
        <div className="flex items-center gap-1.5">
          <Database className="w-3.5 h-3.5 text-blue-400" />
          <span className="text-gray-400">Source:</span>
          <span className="text-white font-semibold">{metadata?.dataset_name || 'india_weather_sample'}</span>
        </div>

        <div className="flex items-center gap-1.5">
          <Layers className="w-3.5 h-3.5 text-amber-400" />
          <span className="text-gray-400">Grid Resolution:</span>
          <span className="text-white">{metadata?.resolution_km || 25} km</span>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="text-gray-400">Time Bounds:</span>
          <span className="text-emerald-400">{metadata?.time_start || 'N/A'} &rarr; {metadata?.time_end || 'N/A'}</span>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="text-gray-400">Events Detected:</span>
          <span className="text-aeris-accent font-bold px-2 py-0.5 bg-aeris-bg rounded border border-aeris-border">
            {totalEvents} Active
          </span>
        </div>
      </div>
    </footer>
  );
}
