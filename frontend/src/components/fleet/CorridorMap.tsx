import React from 'react';
import { BusState } from '../../types/api';

interface CorridorMapProps {
  buses: BusState[];
}

export const CorridorMap: React.FC<CorridorMapProps> = ({ buses }) => {
  // 29 corridor segments along Kandy Route 654
  const totalSegments = 29;
  const segments = Array.from({ length: totalSegments }, (_, i) => i + 1);

  // Group buses by segment
  const busesBySegment: Record<number, BusState[]> = {};
  buses.forEach((b) => {
    const seg = b.current_segment;
    if (!busesBySegment[seg]) busesBySegment[seg] = [];
    busesBySegment[seg].push(b);
  });

  const getStopLandmark = (seg: number) => {
    if (seg === 1) return 'Kandy GS';
    if (seg === 5) return 'Clock Tower';
    if (seg === 10) return 'Mahamaya';
    if (seg === 15) return 'Tennekumbura';
    if (seg === 20) return 'Kundasale';
    if (seg === 25) return 'Digana';
    if (seg === 29) return 'Teldeniya';
    return null;
  };

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">Route 654 Corridor Topological Schematic</div>
          <div className="card-subtitle">Kandy Goods Shed → Teldeniya (29 Segments)</div>
        </div>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
          Active Fleet: <span style={{ color: 'var(--color-primary)', fontWeight: 700 }}>{buses.length} Vehicles</span>
        </div>
      </div>

      <div className="corridor-track">
        {segments.map((seg) => {
          const activeBuses = busesBySegment[seg] || [];
          const hasBus = activeBuses.length > 0;
          const landmark = getStopLandmark(seg);

          return (
            <div
              key={seg}
              className={`corridor-segment ${hasBus ? 'has-bus' : ''}`}
              title={`Segment ${seg}${landmark ? ` (${landmark})` : ''} - ${activeBuses.length} active buses`}
            >
              {hasBus && (
                <div className="bus-marker">
                  🚌 {activeBuses.map((b) => b.deviceid).join(', ')}
                </div>
              )}
              <div className="segment-number">S{seg}</div>
              {landmark && (
                <div style={{ fontSize: '0.62rem', color: 'var(--color-primary)', fontWeight: 600, marginTop: '2px', textAlign: 'center' }}>
                  {landmark}
                </div>
              )}
            </div>
          );
        })}
      </div>
      
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '16px', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
        <span>📍 Origin: Kandy Goods Shed Terminal (S1)</span>
        <span>⚡ Real-Time Stream Progression</span>
        <span>🏁 Destination: Teldeniya Bus Stand (S29)</span>
      </div>
    </div>
  );
};
