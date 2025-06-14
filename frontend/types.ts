

export type SymbolString = string; // Using emojis or simple strings as symbols

export interface TileData {
  id: string;
  symbol: SymbolString;
  x: number; // top-left x coordinate on board
  y: number; // top-left y coordinate on board
  layer: number; // z-index layer, higher is more on top
  isCollected: boolean;
  isMatched: boolean;
}

export interface TileDataInSnapshot extends TileData {
  isAccessibleInSnapshot: boolean;
}

export type GameState = 'loading' | 'playing' | 'won' | 'lost';

export interface BoardTile extends TileData {
  // Currently same as TileData, can be extended if board-specific properties differ
}

// --- Difficulty Level Types ---
export type DifficultyLevel = "easy" | "medium" | "hard" | "hell";

export interface LayerConfigItem {
  tileCount: number;
  cols: number;
  rows: number;
  xOffset: number;
  yOffset: number;
  layerZ: number;
}
export interface DifficultySettings {
  NAME: string;
  NUM_UNIQUE_SYMBOLS_TO_USE: number;
  NUM_SETS_PER_SYMBOL: number;
  TILES_PER_MATCH_SET: number;
  LAYERS_CONFIG: LayerConfigItem[];
  COLLECTION_SLOT_CAPACITY: number;
  TOTAL_TILES_TO_GENERATE: number;
}


// --- Local RL Agent Analysis Types ---
export type MoveEvaluationCategory = "Genius" | "Good" | "Average" | "Bad" | "Stupid" | "Info";

export interface PredictedMove {
  tileIdPicked: string;
  symbolPicked: SymbolString;
  matchMade: boolean;
  newBoardSnapshot: TileDataInSnapshot[];
  newCollectionSlotSnapshot: TileData[];
  // Optional: Add agent's evaluation of this hypothetical state if needed
  // simulatedBoardValueAfterThisPredictedMove?: number; 
}

export interface MissedOpportunityDetails {
  tileSymbol: SymbolString;
  tileId: string;
  reasoning: string; // Why this was a better move
  simulatedPolicyConfidence?: number; // Optional: agent's confidence in this alternative
  predictedSequence?: PredictedMove[]; // For visualizing AI lookahead
}

export interface LocalAgentMoveEvaluation {
  moveNumber: number;
  playerTileClickedSymbol: SymbolString;
  playerTileClickedId: string;
  evaluation: MoveEvaluationCategory;      // The agent's direct evaluation of the player's move
  agentReasoning: string;                 // Why the player's move got this evaluation
  missedOpportunity?: MissedOpportunityDetails; // Details if a significantly better move was missed
  // Conceptual NN-like outputs, primarily used for rich reasoning text
  simulatedPlayerMovePolicyConfidence?: number;
  simulatedBoardValueAfterPlayerMove?: number;
}

export interface BoardStateBeforeMove { // For Agent's logical analysis
  accessibleTiles: Pick<TileData, 'id' | 'symbol' | 'layer'>[];
  totalTilesOnBoard: number; // Total non-matched, non-collected tiles
  tilesBySymbol: Record<SymbolString, number>; // Count of each symbol among all non-matched, non-collected tiles
}

export interface MoveRecord {
  moveNumber: number;
  tileClicked: TileData; // The tile that was clicked
  
  // For Agent's logical analysis:
  boardStateBeforeMove: BoardStateBeforeMove;
  collectionSlotBeforeMove: TileData[]; // Tiles in slot before this move (logical state)

  // For Visual MiniBoard rendering:
  tilesOnBoardSnapshot: TileDataInSnapshot[]; // Full snapshot of all board tiles' state and accessibility
  collectionSlotSnapshot: TileData[];      // Full snapshot of collection slot tiles

  matchMade: boolean;
  collectionSlotAfterMove: TileData[];
}

// --- Saved Game Record Type ---
// FIX: Define and export SavedGameRecord interface
export interface SavedGameRecord {
  gameId: string;
  timestamp: number;
  difficulty: DifficultyLevel;
  outcome: GameState; // Typically 'won' or 'lost'
  moveHistory: MoveRecord[];
  analysisResults: LocalAgentMoveEvaluation[] | null;
}