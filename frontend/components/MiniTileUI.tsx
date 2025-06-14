
import React from 'react';
import { TileDataInSnapshot } from '../types';

interface MiniTileUIProps {
  tile: TileDataInSnapshot;
  scale: number;
  isPlayerClickedTile?: boolean;
  isAgentSuggestedTile?: boolean;
  isSlotTile?: boolean; // To slightly change style if it's in the slot display
}

export const MiniTileUI: React.FC<MiniTileUIProps> = ({
  tile,
  scale,
  isPlayerClickedTile = false,
  isAgentSuggestedTile = false,
  isSlotTile = false,
}) => {
  const TILE_WIDTH = 64; // Base width from constants
  const TILE_HEIGHT = 80; // Base height from constants

  const scaledWidth = TILE_WIDTH * scale;
  const scaledHeight = TILE_HEIGHT * scale;
  const fontSize = Math.max(10, 24 * scale); // Ensure font size is not too small

  let tileStyles = `
    absolute flex items-center justify-center
    border rounded shadow-sm
    text-center overflow-hidden whitespace-nowrap
    transition-all duration-100
  `;

  if (isSlotTile) {
     tileStyles = `
        flex items-center justify-center
        border rounded shadow-sm
        text-center overflow-hidden whitespace-nowrap
        w-full h-full bg-slate-300 border-slate-400 text-slate-800
     `;
  } else if (tile.isAccessibleInSnapshot) {
    tileStyles += ' bg-slate-300 border-slate-400 text-slate-800';
  } else {
    tileStyles += ' bg-slate-500 border-slate-600 text-slate-400 opacity-70';
  }

  if (isPlayerClickedTile) {
    tileStyles += ' ring-2 ring-offset-1 ring-yellow-400 scale-105 z-[100]'; // Player's move highlight
  } else if (isAgentSuggestedTile) {
    tileStyles += ' ring-2 ring-offset-1 ring-sky-400 scale-105 z-[99]'; // Agent's suggestion highlight
     // Add a subtle pulse for agent suggestion
    tileStyles += ' animate-pulse-light';
  }
  
  // Add a Tailwind like animation for pulsing if not available by default.
  // This would typically be in your global CSS or index.html <style>
  // For brevity, I'll define a keyframe style here, but it's better placed globally.
  const keyframes = `
    @keyframes pulse-light {
        0%, 100% { opacity: 1; box-shadow: 0 0 3px 1px rgba(56, 189, 248, 0.5); } /* sky-400 */
        50% { opacity: 0.8; box-shadow: 0 0 6px 2px rgba(56, 189, 248, 0.7); }
    }
    .animate-pulse-light {
        animation: pulse-light 1.5s infinite;
    }
  `;


  return (
    <>
      {isAgentSuggestedTile && <style>{keyframes}</style>}
      <div
        style={ isSlotTile ? { fontSize: `${fontSize}px` } : {
          left: `${tile.x * scale}px`,
          top: `${tile.y * scale}px`,
          width: `${scaledWidth}px`,
          height: `${scaledHeight}px`,
          zIndex: tile.layer + (isPlayerClickedTile || isAgentSuggestedTile ? 50 : 0),
          fontSize: `${fontSize}px`,
        }}
        className={tileStyles}
        title={`Tile ${tile.symbol} (Layer ${tile.layer})${tile.isAccessibleInSnapshot ? '' : ' - Covered'}`}
      >
        {tile.symbol}
      </div>
    </>
  );
};
