'use client';

import React from 'react';
import { Filter, Calendar, Zap, AlertTriangle } from 'lucide-react';

interface EventFiltersProps {
  availableTypes: string[];
  availableSeverities: string[];
  availableTimestamps: string[];
  selectedType: string;
  selectedSeverity: string;
  selectedTimestamp: string;
  onTypeChange: (type: string) => void;
  onSeverityChange: (sev: string) => void;
  onTimestampChange: (ts: string) => void;
  onResetFilters: () => void;
}

export default function EventFilters({
  availableTypes,
  availableSeverities,
  availableTimestamps,
  selectedType,
  selectedSeverity,
  selectedTimestamp,
  onTypeChange,
  onSeverityChange,
  onTimestampChange,
  onResetFilters,
}: EventFiltersProps) {
  return (
    <div className="w-full bg-aeris-card/90 backdrop-blur-md border border-aeris-border rounded-xl p-4 flex flex-wrap items-center justify-between gap-4 shadow-lg">
      <div className="flex items-center gap-2 text-aeris-accent font-semibold text-sm">
        <Filter className="w-4 h-4" />
        <span>ANOMALY FILTERS</span>
      </div>

      <div className="flex flex-wrap items-center gap-3 text-xs">
        {/* Event Type Filter */}
        <div className="flex items-center gap-1.5 bg-aeris-bg/80 border border-aeris-border px-3 py-1.5 rounded-lg">
          <Zap className="w-3.5 h-3.5 text-yellow-400" />
          <span className="text-gray-400 font-medium">Variable / Type:</span>
          <select
            value={selectedType}
            onChange={(e) => onTypeChange(e.target.value)}
            className="bg-transparent text-white font-mono focus:outline-none cursor-pointer"
          >
            <option value="ALL" className="bg-aeris-card text-white">All Event Types</option>
            {availableTypes.map((t) => (
              <option key={t} value={t} className="bg-aeris-card text-white">
                {t.toUpperCase()}
              </option>
            ))}
          </select>
        </div>

        {/* Severity Filter */}
        <div className="flex items-center gap-1.5 bg-aeris-bg/80 border border-aeris-border px-3 py-1.5 rounded-lg">
          <AlertTriangle className="w-3.5 h-3.5 text-orange-400" />
          <span className="text-gray-400 font-medium">Analytical Severity:</span>
          <select
            value={selectedSeverity}
            onChange={(e) => onSeverityChange(e.target.value)}
            className="bg-transparent text-white font-mono focus:outline-none cursor-pointer"
          >
            <option value="ALL" className="bg-aeris-card text-white">All Severities</option>
            {availableSeverities.map((s) => (
              <option key={s} value={s} className="bg-aeris-card text-white">
                {s.toUpperCase()}
              </option>
            ))}
          </select>
        </div>

        {/* Forecast Timestamp Filter */}
        <div className="flex items-center gap-1.5 bg-aeris-bg/80 border border-aeris-border px-3 py-1.5 rounded-lg">
          <Calendar className="w-3.5 h-3.5 text-emerald-400" />
          <span className="text-gray-400 font-medium">Forecast Time:</span>
          <select
            value={selectedTimestamp}
            onChange={(e) => onTimestampChange(e.target.value)}
            className="bg-transparent text-white font-mono focus:outline-none cursor-pointer"
          >
            <option value="ALL" className="bg-aeris-card text-white">All Available Timestamps</option>
            {availableTimestamps.map((ts) => (
              <option key={ts} value={ts} className="bg-aeris-card text-white">
                {ts}
              </option>
            ))}
          </select>
        </div>
      </div>

      {(selectedType !== 'ALL' || selectedSeverity !== 'ALL' || selectedTimestamp !== 'ALL') && (
        <button
          onClick={onResetFilters}
          className="text-xs text-gray-400 hover:text-white underline underline-offset-2 transition"
        >
          Reset Filters
        </button>
      )}
    </div>
  );
}
