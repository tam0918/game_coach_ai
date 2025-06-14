
import React from 'react';
import { TileData } from '../types';
import { TILE_WIDTH_CLASS, TILE_HEIGHT_CLASS } from '../constants';

interface TileUIProps {
  tile: TileData;
  onClick: (id: string) => void;
  isAccessible: boolean;
}

export const TileUI: React.FC<TileUIProps> = ({ tile, onClick, isAccessible }) => {
  const handleClick = () => {
    if (isAccessible) {
      onClick(tile.id);
    }
  };

  const tileBaseStyle = `
    absolute flex items-center justify-center 
    ${TILE_WIDTH_CLASS} ${TILE_HEIGHT_CLASS}
    border-2 rounded-lg shadow-lg 
    transform transition-all duration-150 ease-in-out
    text-4xl
  `;

  const accessibleStyle = `
    bg-slate-200 border-slate-400 text-slate-800 
    cursor-pointer hover:scale-105 hover:shadow-xl hover:border-yellow-400
  `;
  
  const inaccessibleStyle = `
    bg-slate-500 border-slate-600 text-slate-400 
    opacity-60 cursor-not-allowed
  `;

  return (
    <div
      style={{
        left: `${tile.x}px`,
        top: `${tile.y}px`,
        zIndex: tile.layer, // CSS z-index based on layer
      }}
      className={`${tileBaseStyle} ${isAccessible ? accessibleStyle : inaccessibleStyle}`}
      onClick={handleClick}
      role="button"
      aria-label={`Tile ${tile.symbol} ${isAccessible ? '(clickable)' : '(covered)'}`}
      tabIndex={isAccessible ? 0 : -1}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleClick();}}
    >
      {tile.symbol}
    </div>
  );
};
    