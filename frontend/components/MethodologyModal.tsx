'use client';

import React from 'react';
import { X, CheckCircle2, Clock, Info } from 'lucide-react';

interface MethodologyModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function MethodologyModal({ isOpen, onClose }: MethodologyModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[2000] bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-aeris-card border border-aeris-border rounded-2xl max-w-2xl w-full max-h-[85vh] overflow-y-auto custom-scrollbar p-6 shadow-2xl text-gray-200">
        <div className="flex items-center justify-between border-b border-aeris-border pb-4 mb-4">
          <div className="flex items-center gap-2">
            <Info className="w-5 h-5 text-aeris-accent" />
            <h2 className="text-lg font-bold text-white tracking-wide">
              AERIS Scientific Methodology & System Scope
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-gray-400 hover:text-white rounded-lg transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <p className="text-xs text-gray-300 leading-relaxed mb-6">
          SIH 2026 Problem Statement 26078 &bull; <strong>AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies in Medium-Range Forecasts</strong>. AERIS combines rigorous meteorological data ingestion, baseline climatology z-score calculations, spatial anomaly region clustering, and geodesic spatio-temporal event tracking.
        </p>

        {/* Implemented Core Pipeline */}
        <div className="space-y-4 mb-6">
          <h3 className="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4" /> IMPLEMENTED MVP PIPELINE
          </h3>

          <div className="space-y-3 text-xs">
            <div className="bg-aeris-bg/80 border border-emerald-500/30 p-3 rounded-xl">
              <div className="font-bold text-white mb-1">1. Multi-Format NWP Data Ingestion</div>
              <div className="text-gray-300">
                Standardizes NetCDF4 / GRIB2 grids, validates coordinate ordering (lat/lon), parses timestamps, and converts physical units (K &rarr; °C, Pa &rarr; hPa, m/s &rarr; km/h).
              </div>
            </div>

            <div className="bg-aeris-bg/80 border border-emerald-500/30 p-3 rounded-xl">
              <div className="font-bold text-white mb-1">2. Climatological Baseline Engine</div>
              <div className="text-gray-300">
                Computes historical mean ($\mu$) and standard deviation ($\sigma$) across rolling spatial/temporal windows to establish a rigorous physical reference.
              </div>
            </div>

            <div className="bg-aeris-bg/80 border border-emerald-500/30 p-3 rounded-xl">
              <div className="font-bold text-white mb-1">3. Statistical Anomaly & Region Detection</div>
              <div className="text-gray-300">
                Calculates standardized z-score anomalies ($Z = (X - \mu) / \sigma$), identifies extreme grid cells ($|Z| \ge 2.0$), and clusters spatial anomaly regions via 8-connectivity.
              </div>
            </div>

            <div className="bg-aeris-bg/80 border border-emerald-500/30 p-3 rounded-xl">
              <div className="font-bold text-white mb-1">4. Spatio-Temporal Tracking & Trajectories</div>
              <div className="text-gray-300">
                Links event centroids across timesteps using geodesic Haversine distance, area ratios, and intensity overlap to calculate track direction, velocity vectors, and future position projections.
              </div>
            </div>
          </div>
        </div>

        {/* Planned / Research Extensions */}
        <div className="space-y-4">
          <h3 className="text-xs font-bold text-amber-400 uppercase tracking-wider flex items-center gap-2">
            <Clock className="w-4 h-4" /> PLANNED / RESEARCH EXTENSIONS
          </h3>

          <div className="space-y-3 text-xs">
            <div className="bg-aeris-bg/80 border border-amber-500/30 p-3 rounded-xl">
              <div className="font-bold text-white mb-1">5. Graph Neural Network (GNN) Spatio-Temporal Prediction</div>
              <div className="text-gray-300">
                Graph Transformer / Spherical GNN model for learning non-linear atmospheric fluid motion dynamics and multi-variable interaction trajectories across global medium-range forecasts.
              </div>
            </div>

            <div className="bg-aeris-bg/80 border border-amber-500/30 p-3 rounded-xl">
              <div className="font-bold text-white mb-1">6. Physics-Guided Diffusion Downscaling</div>
              <div className="text-gray-300">
                Generative score-based diffusion model conditioned on high-resolution topography for downscaling coarse NWP forecasts (25 km &rarr; 3 km) while preserving mass and energy conservation laws.
              </div>
            </div>
          </div>
        </div>

        <div className="mt-6 pt-4 border-t border-aeris-border flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-aeris-accent text-aeris-bg font-bold text-xs rounded-lg hover:bg-cyan-400 transition"
          >
            Close Scope Overview
          </button>
        </div>
      </div>
    </div>
  );
}
