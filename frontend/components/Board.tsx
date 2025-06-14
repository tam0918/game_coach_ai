
import React from 'react';
import { TileUI } from './TileUI';
import { TileData } from '../types';
import { TILE_WIDTH, TILE_HEIGHT, BOARD_PADDING } from '../constants';

interface BoardProps {
  tiles: TileData[]; // Only non-collected, non-matched tiles
  onTileClick: (id: string) => void;
  allTilesForAccessibilityCheck: TileData[]; // All tiles for context to determine which are accessible
  boardDimensions: { width: number; height: number };
}

// Helper function (could be moved to utils if shared or kept here if specific to Board's needs)
// This function determines if a tile *would be* accessible if it weren't collected/matched
const isTileStructurallyAccessible = (tileId: string, allTiles: TileData[]): boolean => {
    const targetTile = allTiles.find(t => t.id === tileId);
    if (!targetTile) return false; // Should not happen if tileId comes from 'tiles' prop

    // Check against all *other* tiles that are still on the board (not collected, not matched)
    for (const otherTile of allTiles) {
      if (otherTile.id === targetTile.id || otherTile.isCollected || otherTile.isMatched) continue;

      // Only consider tiles on a higher layer for occlusion
      if (otherTile.layer > targetTile.layer) {
        // AABB collision check
        const t1_left = targetTile.x;
        const t1_right = targetTile.x + TILE_WIDTH;
        const t1_top = targetTile.y;
        const t1_bottom = targetTile.y + TILE_HEIGHT;

        const t2_left = otherTile.x;
        const t2_right = otherTile.x + TILE_WIDTH;
        const t2_top = otherTile.y;
        const t2_bottom = otherTile.y + TILE_HEIGHT;
        
        const xOverlap = t1_left < t2_right && t1_right > t2_left;
        const yOverlap = t1_top < t2_bottom && t1_bottom > t2_top;

        if (xOverlap && yOverlap) {
          return false; 
        }
      }
    }
    return true;
  };


export const Board: React.FC<BoardProps> = ({ tiles, onTileClick, allTilesForAccessibilityCheck, boardDimensions }) => {
  
  // Sort tiles for consistent rendering order, especially if z-index ties occur.
  const sortedTiles = [...tiles].sort((a, b) => {
    if (a.layer !== b.layer) {
      return a.layer - b.layer;
    }
    if (a.y !== b.y) {
      return a.y - b.y;
    }
    return a.x - b.x;
  });

  return (
    <div 
      className="relative bg-slate-700/30 shadow-2xl rounded-lg border border-slate-600 overflow-auto"
      style={{ 
        width: `${boardDimensions.width}px`, 
        height: `${boardDimensions.height}px`,
        maxWidth: '100%', 
        maxHeight: 'calc(100vh - 250px)', 
      }}
      role="grid"
      aria-label="Game Board"
    >
      <div className="relative" style={{width: `${boardDimensions.width - BOARD_PADDING*2}px`, height: `${boardDimensions.height - BOARD_PADDING*2}px`, margin: `${BOARD_PADDING}px`}}>
        {sortedTiles.map(tile => (
          <TileUI
            key={tile.id}
            tile={tile}
            onClick={onTileClick}
            // A tile is considered "UI accessible" if it's structurally accessible
            // AND not already collected or matched. App.tsx handles the click logic,
            // this prop is mainly for styling in TileUI.
            isAccessible={isTileStructurallyAccessible(tile.id, allTilesForAccessibilityCheck)}
          />
        ))}
      </div>
    </div>
  );
};
