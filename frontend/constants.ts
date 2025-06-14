
import { DifficultyLevel, DifficultySettings } from './types';

export const TILE_WIDTH = 64; // px
export const TILE_HEIGHT = 80; // px
export const TILE_WIDTH_CLASS = 'w-16'; // Corresponds to 64px if 1rem = 16px
export const TILE_HEIGHT_CLASS = 'h-20'; // Corresponds to 80px

export const SYMBOLS: string[] = ['🍓', '🍌', '🍉', '🍇', '🍊', '🍍', '🥝', '🍒', '🍑', '🥭', '🥥', '🥑', '🍋', '🍐', '🍅', '🍆']; // Added more symbols

export const TILES_PER_MATCH_SET_CONST = 3; // Base constant, actual value per difficulty

export const BOARD_PADDING = 20; // Padding around the tile area

export const DIFFICULTY_LEVELS: Record<DifficultyLevel, DifficultySettings> = {
  easy: {
    NAME: "Easy",
    NUM_UNIQUE_SYMBOLS_TO_USE: 6,
    NUM_SETS_PER_SYMBOL: 2,
    TILES_PER_MATCH_SET: 3,
    COLLECTION_SLOT_CAPACITY: 7,
    LAYERS_CONFIG: [
      { tileCount: 20, cols: 5, rows: 4, xOffset: TILE_WIDTH / 3, yOffset: TILE_HEIGHT / 3, layerZ: 0 },
      { tileCount: 16, cols: 4, rows: 4, xOffset: TILE_WIDTH / 1.5, yOffset: TILE_HEIGHT / 1.5, layerZ: 1 },
    ],
    TOTAL_TILES_TO_GENERATE: 36, // 6 * 2 * 3
  },
  medium: {
    NAME: "Medium",
    NUM_UNIQUE_SYMBOLS_TO_USE: 8,
    NUM_SETS_PER_SYMBOL: 2,
    TILES_PER_MATCH_SET: 3,
    COLLECTION_SLOT_CAPACITY: 7,
    LAYERS_CONFIG: [
      { tileCount: 24, cols: 6, rows: 4, xOffset: 0, yOffset: 0, layerZ: 0 },
      { tileCount: 15, cols: 5, rows: 3, xOffset: TILE_WIDTH / 2.5, yOffset: TILE_HEIGHT / 2.5, layerZ: 1 },
      { tileCount: 9, cols: 3, rows: 3, xOffset: TILE_WIDTH / 1.5, yOffset: TILE_HEIGHT / 1.5, layerZ: 2 },
    ],
    TOTAL_TILES_TO_GENERATE: 48, // 8 * 2 * 3
  },
  hard: {
    NAME: "Hard",
    NUM_UNIQUE_SYMBOLS_TO_USE: 10,
    NUM_SETS_PER_SYMBOL: 2,
    TILES_PER_MATCH_SET: 3,
    COLLECTION_SLOT_CAPACITY: 7, // Could be tighter, e.g., 6
    LAYERS_CONFIG: [
      { tileCount: 25, cols: 5, rows: 5, xOffset: 0, yOffset: 0, layerZ: 0 }, // Denser bottom
      { tileCount: 20, cols: 5, rows: 4, xOffset: TILE_WIDTH / 3, yOffset: TILE_HEIGHT / 3, layerZ: 1 }, // Denser middle
      { tileCount: 15, cols: 5, rows: 3, xOffset: TILE_WIDTH / 1.5, yOffset: TILE_HEIGHT / 1.5, layerZ: 2 }, // Wider top
    ],
    TOTAL_TILES_TO_GENERATE: 60, // 10 * 2 * 3
  },
  hell: {
    NAME: "Hell",
    NUM_UNIQUE_SYMBOLS_TO_USE: 12,
    NUM_SETS_PER_SYMBOL: 2,
    TILES_PER_MATCH_SET: 3,
    COLLECTION_SLOT_CAPACITY: 7, // Changed from 5 to 7
    LAYERS_CONFIG: [
      { tileCount: 28, cols: 7, rows: 4, xOffset: 0, yOffset: 0, layerZ: 0 }, // Wide base
      { tileCount: 20, cols: 5, rows: 4, xOffset: TILE_WIDTH * 0.8, yOffset: TILE_HEIGHT * 0.5, layerZ: 1 }, // Significant offset
      { tileCount: 12, cols: 4, rows: 3, xOffset: TILE_WIDTH * 0.3, yOffset: TILE_HEIGHT * 1.2, layerZ: 2 }, // Different offset
      { tileCount: 8, cols: 2, rows: 4, xOffset: TILE_WIDTH * 2.5, yOffset: TILE_HEIGHT * 0.8, layerZ: 3 }, // Tall, narrow stack
      { tileCount: 4, cols: 2, rows: 2, xOffset: TILE_WIDTH * 1.2, yOffset: TILE_HEIGHT * 2.0, layerZ: 4 }, // Small capstone
    ],
    TOTAL_TILES_TO_GENERATE: 72, // 12 * 2 * 3
  },
};

// Validate configurations
Object.values(DIFFICULTY_LEVELS).forEach(settings => {
  const totalTilesFromSymbols = settings.NUM_UNIQUE_SYMBOLS_TO_USE * settings.NUM_SETS_PER_SYMBOL * settings.TILES_PER_MATCH_SET;
  const totalTilesFromLayers = settings.LAYERS_CONFIG.reduce((sum, layer) => sum + layer.tileCount, 0);
  if (totalTilesFromSymbols !== settings.TOTAL_TILES_TO_GENERATE) {
    console.warn(
      `Mismatch in TOTAL_TILES_TO_GENERATE for difficulty ${settings.NAME}. Expected ${totalTilesFromSymbols}, got ${settings.TOTAL_TILES_TO_GENERATE}.`
    );
  }
  if (totalTilesFromLayers !== settings.TOTAL_TILES_TO_GENERATE) {
     console.warn(
      `Mismatch in tile count from LAYERS_CONFIG for difficulty ${settings.NAME}. Layers sum to ${totalTilesFromLayers}, but TOTAL_TILES_TO_GENERATE is ${settings.TOTAL_TILES_TO_GENERATE}. Ensure layers sum to NUM_UNIQUE_SYMBOLS * NUM_SETS_PER_SYMBOL * TILES_PER_MATCH_SET.`
    );
  }
});
