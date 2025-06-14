
import React from 'react';
import { TileData } from '../types';
import { TILE_WIDTH_CLASS, TILE_HEIGHT_CLASS } from '../constants';

interface CollectionBarProps {
  tiles: TileData[];
  capacity: number;
}

export const CollectionBar: React.FC<CollectionBarProps> = ({ tiles, capacity }) => {
  return (
    <div className="w-full max-w-2xl p-3 my-4 bg-slate-700 rounded-xl shadow-inner border border-slate-600">
      <div className="flex justify-center items-center space-x-1 h-[90px]"> {/* Fixed height for bar */}
        {Array.from({ length: capacity }).map((_, index) => {
          const tile = tiles[index];
          return (
            <div
              key={index}
              className={`
                ${TILE_WIDTH_CLASS} ${TILE_HEIGHT_CLASS}
                flex items-center justify-center 
                border-2 rounded-md
                ${tile ? 'bg-slate-200 border-slate-400 text-slate-800 text-3xl shadow-md' : 'bg-slate-600 border-slate-500 opacity-50'}
              `}
            >
              {tile ? tile.symbol : ''}
            </div>
          );
        })}
      </div>
    </div>
  );
};
    