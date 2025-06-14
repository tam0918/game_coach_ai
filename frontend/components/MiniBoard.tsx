
import React from 'react';
import { TileData, TileDataInSnapshot } from '../types';
import { MiniTileUI } from './MiniTileUI';
// FIX: Remove COLLECTION_SLOT_CAPACITY import as it's not a direct export and varies by difficulty. It will be passed as a prop.
import { TILE_WIDTH, TILE_HEIGHT } from '../constants';

interface MiniBoardProps {
  tilesOnBoard: TileDataInSnapshot[];
  collectionSlot: TileData[];
  playerClickedTileId?: string;
  agentSuggestedTileId?: string;
  // FIX: Add collectionSlotCapacity prop
  collectionSlotCapacity: number;
}

const MINI_TILE_SCALE = 0.5; // Adjust this to change the size of the mini tiles
const MINI_TILE_WIDTH = TILE_WIDTH * MINI_TILE_SCALE;
const MINI_TILE_HEIGHT = TILE_HEIGHT * MINI_TILE_SCALE;

export const MiniBoard: React.FC<MiniBoardProps> = ({
  tilesOnBoard,
  collectionSlot,
  playerClickedTileId,
  agentSuggestedTileId,
  // FIX: Destructure collectionSlotCapacity from props
  collectionSlotCapacity,
}) => {
  let maxBoardX = 0;
  let maxBoardY = 0;

  tilesOnBoard.forEach(tile => {
    maxBoardX = Math.max(maxBoardX, tile.x + TILE_WIDTH);
    maxBoardY = Math.max(maxBoardY, tile.y + TILE_HEIGHT);
  });
  
  const boardContainerWidth = maxBoardX * MINI_TILE_SCALE;
  const boardContainerHeight = maxBoardY * MINI_TILE_SCALE;

  const sortedTiles = [...tilesOnBoard].sort((a, b) => {
    if (a.layer !== b.layer) return a.layer - b.layer;
    if (a.y !== b.y) return a.y - b.y;
    return a.x - b.x;
  });

  return (
    <div className="bg-slate-700/50 p-2 rounded-md border border-slate-600 aspect-[4/3]">
      {/* Board Area */}
      <div
        className="relative mx-auto overflow-hidden bg-slate-600/30 rounded"
        style={{
          width: `${boardContainerWidth}px`,
          height: `${boardContainerHeight}px`,
          // Ensure it doesn't overflow its container too much if board is huge
          maxWidth: '100%', 
          maxHeight: 'calc(100% - 30px)', // leave space for slot
        }}
      >
        {sortedTiles.map(tile => (
          <MiniTileUI
            key={`mini-${tile.id}`}
            tile={tile}
            scale={MINI_TILE_SCALE}
            isPlayerClickedTile={tile.id === playerClickedTileId}
            isAgentSuggestedTile={tile.id === agentSuggestedTileId}
          />
        ))}
      </div>

      {/* Collection Slot Area */}
      {/* FIX: Use collectionSlotCapacity prop for Array.from length */}
      <div className="mt-1.5 flex justify-center items-center space-x-0.5" style={{ minHeight: `${MINI_TILE_HEIGHT + 4}px`}}>
        {Array.from({ length: collectionSlotCapacity }).map((_, index) => {
          const tile = collectionSlot[index];
          return (
            <div
              key={`slot-${index}`}
              className="flex items-center justify-center border border-slate-500 rounded-sm"
              style={{
                width: `${MINI_TILE_WIDTH}px`,
                height: `${MINI_TILE_HEIGHT}px`,
                fontSize: `${12 * MINI_TILE_SCALE * 2}px` // Adjust font size based on scale
              }}
            >
              {tile ? (
                <MiniTileUI
                    tile={{...tile, isAccessibleInSnapshot: true}} // Slot tiles are always "accessible" in their context
                    scale={MINI_TILE_SCALE}
                    isPlayerClickedTile={tile.id === playerClickedTileId && collectionSlot.some(cs => cs.id === playerClickedTileId)}
                    isAgentSuggestedTile={false} // Not highlighting agent suggestions in slot for now
                    isSlotTile={true}
                />
              ) : (
                <div className="w-full h-full bg-slate-600/70 rounded-sm"></div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
