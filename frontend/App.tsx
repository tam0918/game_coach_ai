import React, { useState, useEffect, useCallback } from 'react';
import { Board } from './components/Board';
import { CollectionBar } from './components/CollectionBar';
import { GameStatusModal } from './components/GameStatusModal';
import { GameAnalysisModal } from './components/GameAnalysisModal';
import { PostMatchReviewModal } from './components/PostMatchReviewModal';
// Assuming MatchHistoryModal exists for the broader feature
// import { MatchHistoryModal } from './components/MatchHistoryModal'; 
import { TileData, GameState, SymbolString, MoveRecord, LocalAgentMoveEvaluation, BoardStateBeforeMove, MoveEvaluationCategory, MissedOpportunityDetails, TileDataInSnapshot, DifficultyLevel, DifficultySettings, PredictedMove, SavedGameRecord } from './types';
import {
  SYMBOLS,
  TILE_WIDTH,
  TILE_HEIGHT,
  BOARD_PADDING,
  DIFFICULTY_LEVELS,
  // Constants for match history
  // LOCAL_STORAGE_KEY_SAVED_GAMES, 
  // MAX_SAVED_GAMES 
} from './constants';
import ApiClient from './apiClient';

// Define these constants here if not properly exported or if constants.ts is not updated
const LOCAL_STORAGE_KEY_SAVED_GAMES = 'tripleTileMatchFun_savedGames';
const MAX_SAVED_GAMES = 10;


const generateId = (): string => Math.random().toString(36).substring(2, 9);

const shuffleArray = <T,>(array: T[]): T[] => {
  const newArray = [...array];
  for (let i = newArray.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [newArray[i], newArray[j]] = [newArray[j], newArray[i]];
  }
  return newArray;
};

const checkTileAccessibilityGlobal = (tileId: string, allTiles: ReadonlyArray<TileData | TileDataInSnapshot>): boolean => {
    const targetTile = allTiles.find(t => t.id === tileId);
    if (!targetTile || targetTile.isCollected || targetTile.isMatched) return false;

    for (const otherTile of allTiles) {
      if (otherTile.id === targetTile.id || otherTile.isCollected || otherTile.isMatched) continue;
      
      // Only consider tiles on a higher layer for occlusion
      if (otherTile.layer > targetTile.layer) {
        // AABB collision check for more precise occlusion
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
          return false; // Target tile is covered by otherTile
        }
      }
    }
    return true; // Target tile is not covered by any other higher-layer tile
  };

const getUnblockingPotential = (tileToRemoveId: string, currentBoardTiles: ReadonlyArray<TileData | TileDataInSnapshot>): number => {
    const tilesWithoutTarget = currentBoardTiles.filter(t => t.id !== tileToRemoveId && !t.isCollected && !t.isMatched);
    let newlyAccessibleCount = 0;
    const originalAccessibilityCache = new Map<string, boolean>();
    const tileToRemove = currentBoardTiles.find(t => t.id === tileToRemoveId);

    tilesWithoutTarget.forEach(candidate => {
      if (tileToRemove && candidate.layer < tileToRemove.layer) { 
         originalAccessibilityCache.set(candidate.id, checkTileAccessibilityGlobal(candidate.id, currentBoardTiles));
      }
    });
    
    for (const candidate of tilesWithoutTarget) {
        if (originalAccessibilityCache.has(candidate.id) && !originalAccessibilityCache.get(candidate.id)) { 
            if (checkTileAccessibilityGlobal(candidate.id, tilesWithoutTarget)) { 
                newlyAccessibleCount++;
            }
        }
    }
    return newlyAccessibleCount;
};

const normalizeScoresToPolicy = (movesWithScores: { tile: Pick<TileData, 'id'|'symbol'|'layer'>, score: number, desc: string, unblocks: number }[]) => {
    if (!movesWithScores.length) return [];
    const minScore = Math.min(...movesWithScores.map(m => m.score));
    const nonNegativeScores = movesWithScores.map(m => ({...m, score: m.score - minScore + 1})); 

    const sumOfScores = nonNegativeScores.reduce((sum, m) => sum + m.score, 0);
    if (sumOfScores === 0) { 
        return nonNegativeScores.map(m => ({ ...m, confidence: 1 / nonNegativeScores.length }));
    }
    return nonNegativeScores.map(m => ({ ...m, confidence: m.score / sumOfScores }));
};

const calculateSimulatedBoardValue = (
    bestAgentMoveScore: number | null,
    collectionSlotState: ReadonlyArray<TileData>, 
    COLLECTION_SLOT_CAPACITY: number,
    currentBoardTilesForSim: ReadonlyArray<TileDataInSnapshot | TileData>,
    TILES_PER_MATCH_SET: number
): number => {
    let value = 0;
    if (bestAgentMoveScore !== null) {
       value += Math.max(-0.5, Math.min(0.5, (bestAgentMoveScore - 75) / 100));
    }
    
    const slotRatio = collectionSlotState.length / COLLECTION_SLOT_CAPACITY;
    if (slotRatio > 0.7) value -= slotRatio * 0.3; 
    if (collectionSlotState.length >= COLLECTION_SLOT_CAPACITY -1 && (bestAgentMoveScore === null || bestAgentMoveScore < 100)) { 
        value = Math.min(value, -0.8); 
    }

    const symbolCountsOnBoard: Record<SymbolString, number> = {};
    currentBoardTilesForSim.forEach(t => {
        if (!t.isCollected && !t.isMatched) { // Only count tiles actually on board
             symbolCountsOnBoard[t.symbol] = (symbolCountsOnBoard[t.symbol] || 0) + 1;
        }
    });
    let deadSymbolPenalty = 0;
    for (const symbol in symbolCountsOnBoard) {
        if (symbolCountsOnBoard[symbol] < TILES_PER_MATCH_SET && symbolCountsOnBoard[symbol] > 0) {
            const inSlotCount = collectionSlotState.filter(t => t.symbol === symbol).length;
            if (inSlotCount > 0 && (inSlotCount + symbolCountsOnBoard[symbol]) < TILES_PER_MATCH_SET) {
                deadSymbolPenalty += 0.05 * inSlotCount; 
            }
        }
    }
    value -= deadSymbolPenalty;

    const remainingPlayableTiles = currentBoardTilesForSim.filter(t => !t.isCollected && !t.isMatched).length;
    if (remainingPlayableTiles === 0 && collectionSlotState.length === 0) return 1.0;
    if (collectionSlotState.length >= COLLECTION_SLOT_CAPACITY && (bestAgentMoveScore === null || bestAgentMoveScore < 100)) return -1.0;

    return Math.max(-1, Math.min(1, parseFloat(value.toFixed(2)))); 
};


const localRLAgent = {
  isInitialized: false,
  simulatedNNModelName: "TileNet-v0-Heuristic",
  simulatedNNAccuracy: 0.50, 
  
  async initialize() {
    await new Promise(resolve => setTimeout(resolve, 100)); 
    this.isInitialized = true;
    console.log("Local RL Agent initialized.");
  },

  async simulateTraining(numGames: number, updateProgress: (progress: number) => void): Promise<string> {
    console.log(`Simulating NN model calibration for ${numGames} conceptual epochs...`);
    this.simulatedNNModelName = "TileNet-v0-Heuristic"; 
    this.simulatedNNAccuracy = 0.50;

    for (let i = 0; i < numGames; i++) {
      await new Promise(resolve => setTimeout(resolve, 20)); 
      const progressPercentage = ((i + 1) / numGames) * 100;
      updateProgress(progressPercentage);
    }
    
    this.simulatedNNModelName = "TileNet-v1-Calibrated";
    this.simulatedNNAccuracy = parseFloat((0.65 + Math.random() * 0.15).toFixed(2)); 
    console.log(`NN model "${this.simulatedNNModelName}" calibrated. Simulated accuracy: ${this.simulatedNNAccuracy * 100}%`);
    return `Simulated NN model "${this.simulatedNNModelName}" calibrated. Accuracy estimate: ${(this.simulatedNNAccuracy * 100).toFixed(0)}%.`;
  },

  calculateRawAgentMoves(
    logicalAccessibleTiles: Pick<TileData, 'id' | 'symbol' | 'layer'>[],
    currentBoardTilesForSim: ReadonlyArray<TileData | TileDataInSnapshot>, // Full board for unblocking potential
    collectionSlotState: ReadonlyArray<TileData>,
    difficultySettings: DifficultySettings,
    allBoardSymbolCounts: Record<SymbolString, number> // Pre-calculated counts on the board
  ): { tile: Pick<TileData, 'id'|'symbol'|'layer'>, score: number, desc: string, unblocks: number }[] {
    const { TILES_PER_MATCH_SET, COLLECTION_SLOT_CAPACITY } = difficultySettings;
    let rawAgentMoves: { tile: Pick<TileData, 'id'|'symbol'|'layer'>, score: number, desc: string, unblocks: number }[] = [];

    const slotSymbolCounts: Record<SymbolString, { count: number, ids: string[] }> = {};
    collectionSlotState.forEach(tile => {
      if (!slotSymbolCounts[tile.symbol]) slotSymbolCounts[tile.symbol] = { count: 0, ids: [] };
      slotSymbolCounts[tile.symbol].count++;
      slotSymbolCounts[tile.symbol].ids.push(tile.id);
    });

    logicalAccessibleTiles.forEach(accTile => {
        let score = 0;
        let desc = "";
        const unblocks = getUnblockingPotential(accTile.id, currentBoardTilesForSim);

        if (slotSymbolCounts[accTile.symbol]?.count === TILES_PER_MATCH_SET - 1) {
            score = 100 + unblocks * 2 + accTile.layer; 
            desc = `completes an immediate triple for ${accTile.symbol}`;
        } else if (collectionSlotState.length >= COLLECTION_SLOT_CAPACITY - 2 && slotSymbolCounts[accTile.symbol]?.count > 0) {
            score = 80 + unblocks + accTile.layer; 
            desc = `consolidates existing ${accTile.symbol} in a nearly full slot`;
        } else if (slotSymbolCounts[accTile.symbol]?.count === TILES_PER_MATCH_SET - 2 && collectionSlotState.length < COLLECTION_SLOT_CAPACITY -1) {
            let thirdTileAvailabilityScore = logicalAccessibleTiles.some(t => t.symbol === accTile.symbol && t.id !== accTile.id) ? 10 : 0;
            score = 70 + thirdTileAvailabilityScore + unblocks + accTile.layer; 
            desc = `forms a pair for ${accTile.symbol}`;
        } else if (collectionSlotState.length < COLLECTION_SLOT_CAPACITY - 2 ) {
            const otherAccessibleOfSymbol = logicalAccessibleTiles.filter(t => t.symbol === accTile.symbol && t.id !== accTile.id).length;
            if (otherAccessibleOfSymbol >=1 && (!slotSymbolCounts[accTile.symbol] || slotSymbolCounts[accTile.symbol].count === 0)) {
                 score = 60 + accTile.layer + unblocks + otherAccessibleOfSymbol * 2; 
                 desc = `starts a new pair for ${accTile.symbol} from board tiles`;
            } else {
                 score = 40 + accTile.layer * 2 + unblocks * 3 + (allBoardSymbolCounts[accTile.symbol] || 0); 
                 desc = `is a strategic layer ${accTile.layer} tile (${accTile.symbol})`;
            }
        } else { 
            score = 30 + accTile.layer + unblocks * 2;
            desc = `is a layer ${accTile.layer} tile (${accTile.symbol})`;
        }
        if (unblocks > 0) desc += ` (unblocks ${unblocks})`;
        rawAgentMoves.push({ tile: accTile, score, desc, unblocks });
    });
    return rawAgentMoves;
  },
  
  simulateLookahead(
    initialBoardTiles: ReadonlyArray<TileDataInSnapshot>,
    initialCollectionSlot: ReadonlyArray<TileData>,
    firstMoveTileData: TileData, 
    difficultySettings: DifficultySettings,
    maxDepth: number = 2 
  ): PredictedMove[] {
    const sequence: PredictedMove[] = [];
    let currentBoardWorkingCopy: TileDataInSnapshot[] = JSON.parse(JSON.stringify(initialBoardTiles));
    let currentSlotWorkingCopy: TileData[] = JSON.parse(JSON.stringify(initialCollectionSlot));
    let nextTileToPickFullData: TileData | undefined = JSON.parse(JSON.stringify(firstMoveTileData));

    for (let depth = 0; depth < maxDepth; depth++) {
        if (!nextTileToPickFullData) break;

        const pickedTileInLoop = nextTileToPickFullData;

        // 1. Simulate adding pickedTileInLoop to slot
        currentSlotWorkingCopy.push(pickedTileInLoop);
        currentBoardWorkingCopy = currentBoardWorkingCopy.map(t => 
            t.id === pickedTileInLoop.id ? { ...t, isCollected: true, isAccessibleInSnapshot: false } : t
        );
        
        // 2. Check for matches
        let matchMadeThisStep = false;
        const slotSymbolCounts: Record<string, TileData[]> = {};
        currentSlotWorkingCopy.forEach(tile => {
            if (!slotSymbolCounts[tile.symbol]) slotSymbolCounts[tile.symbol] = [];
            slotSymbolCounts[tile.symbol].push(tile);
        });

        for (const symbol in slotSymbolCounts) {
            if (slotSymbolCounts[symbol].length >= difficultySettings.TILES_PER_MATCH_SET) {
                matchMadeThisStep = true;
                const idsToMatch = slotSymbolCounts[symbol].slice(0, difficultySettings.TILES_PER_MATCH_SET).map(t => t.id);
                currentSlotWorkingCopy = currentSlotWorkingCopy.filter(t => !idsToMatch.includes(t.id)); 
                currentBoardWorkingCopy = currentBoardWorkingCopy.map(t => 
                    idsToMatch.includes(t.id) ? { ...t, isMatched: true, isCollected: false } : t
                );
                break; 
            }
        }
        
        // Update accessibility for the *next* step on currentBoardWorkingCopy based on changes
        currentBoardWorkingCopy = currentBoardWorkingCopy.map(t => ({
            ...t,
            isAccessibleInSnapshot: (!t.isMatched && !t.isCollected) ? checkTileAccessibilityGlobal(t.id, currentBoardWorkingCopy) : false,
        }));

        sequence.push({
            tileIdPicked: pickedTileInLoop.id,
            symbolPicked: pickedTileInLoop.symbol,
            matchMade: matchMadeThisStep,
            newBoardSnapshot: JSON.parse(JSON.stringify(currentBoardWorkingCopy)),
            newCollectionSlotSnapshot: JSON.parse(JSON.stringify(currentSlotWorkingCopy)),
        });

        if (currentSlotWorkingCopy.length >= difficultySettings.COLLECTION_SLOT_CAPACITY && !matchMadeThisStep) break;
        const remainingOnBoard = currentBoardWorkingCopy.filter(t => !t.isMatched && !t.isCollected).length;
        if (remainingOnBoard === 0 && currentSlotWorkingCopy.length === 0) break;

        const logicalAccessibleTilesForNextStep = currentBoardWorkingCopy
            .filter(t => t.isAccessibleInSnapshot)
            .map(t => ({ id: t.id, symbol: t.symbol, layer: t.layer }));

        if (logicalAccessibleTilesForNextStep.length === 0) break;

        const allBoardSymbolCountsNextStep: Record<SymbolString, number> = {};
        currentBoardWorkingCopy.filter(t => !t.isMatched && !t.isCollected).forEach(t => {
            allBoardSymbolCountsNextStep[t.symbol] = (allBoardSymbolCountsNextStep[t.symbol] || 0) + 1;
        });

        const rawAgentMovesForNextStep = this.calculateRawAgentMoves(
            logicalAccessibleTilesForNextStep,
            currentBoardWorkingCopy,
            currentSlotWorkingCopy,
            difficultySettings,
            allBoardSymbolCountsNextStep
        );
        
        const simulatedPolicyForNextStep = normalizeScoresToPolicy(rawAgentMovesForNextStep);
        simulatedPolicyForNextStep.sort((a,b) => b.confidence - a.confidence);

        if (simulatedPolicyForNextStep.length > 0) {
            const bestNextAgentMoveInfo = simulatedPolicyForNextStep[0];
            // Find the full tile data from the current (copied and modified) board
            nextTileToPickFullData = currentBoardWorkingCopy.find(t => t.id === bestNextAgentMoveInfo.tile.id);
        } else {
            nextTileToPickFullData = undefined;
        }
    }
    return sequence;
  },


  evaluatePlayerMove(
    boardStateBeforePlayerMove: BoardStateBeforeMove, // Logical state before player's move
    fullBoardSnapshotBeforePlayerMove: ReadonlyArray<TileDataInSnapshot>, // Visual snapshot before player's move
    collectionSlotStateBeforePlayerMove: ReadonlyArray<TileData>, // Slot state before player's move
    playerClickedTile: TileData, // The actual tile object player clicked
    currentDifficultySettings: DifficultySettings
  ): Omit<LocalAgentMoveEvaluation, 'moveNumber' | 'playerTileClickedSymbol' | 'playerTileClickedId'> {
    if (!this.isInitialized) {
      return { evaluation: "Info", agentReasoning: "Agent not ready." };
    }
    
    const TILES_PER_MATCH_SET = currentDifficultySettings.TILES_PER_MATCH_SET;
    const COLLECTION_SLOT_CAPACITY = currentDifficultySettings.COLLECTION_SLOT_CAPACITY;
    const agentStatusPrefix = `Agent (${this.simulatedNNModelName} - Acc: ~${(this.simulatedNNAccuracy * 100).toFixed(0)}%): `;
    
    const { accessibleTiles: logicalAccessibleTiles, tilesBySymbol: allBoardSymbolCounts } = boardStateBeforePlayerMove;

    if (!logicalAccessibleTiles || logicalAccessibleTiles.length === 0) {
        return { evaluation: "Info", agentReasoning: agentStatusPrefix + "No accessible tiles to evaluate."}
    }
    
    const currentBoardTilesForSim = fullBoardSnapshotBeforePlayerMove.filter(t => !t.isCollected && !t.isMatched);

    const slotSymbolCountsBeforePlayerMove: Record<SymbolString, { count: number, ids: string[] }> = {};
    collectionSlotStateBeforePlayerMove.forEach(tile => {
      if (!slotSymbolCountsBeforePlayerMove[tile.symbol]) slotSymbolCountsBeforePlayerMove[tile.symbol] = { count: 0, ids: [] };
      slotSymbolCountsBeforePlayerMove[tile.symbol].count++;
      slotSymbolCountsBeforePlayerMove[tile.symbol].ids.push(tile.id);
    });

    const playerTileSymbol = playerClickedTile.symbol;
    const playerTileId = playerClickedTile.id;

    const rawAgentMoves = this.calculateRawAgentMoves(
        logicalAccessibleTiles,
        currentBoardTilesForSim, // For unblocking potential
        collectionSlotStateBeforePlayerMove,
        currentDifficultySettings,
        allBoardSymbolCounts
    );
    
    const simulatedPolicy = normalizeScoresToPolicy(rawAgentMoves);
    simulatedPolicy.sort((a,b) => b.confidence - a.confidence); 

    const playerMoveFormsImmediateTriple = slotSymbolCountsBeforePlayerMove[playerTileSymbol]?.count === TILES_PER_MATCH_SET - 1;
    
    let evaluation: MoveEvaluationCategory = "Average";
    let reasoning = "";
    let missedOpportunity: MissedOpportunityDetails | undefined = undefined;

    const bestAgentPolicyMove = simulatedPolicy.length > 0 ? simulatedPolicy[0] : null;
    const playerMoveInPolicy = simulatedPolicy.find(p => p.tile.id === playerTileId);
    const playerMoveSimulatedConfidence = playerMoveInPolicy?.confidence || 0;

    let tempSlotAfterPlayer = [...collectionSlotStateBeforePlayerMove, playerClickedTile];
    let tempBoardAfterPlayer = fullBoardSnapshotBeforePlayerMove.filter(t => t.id !== playerTileId);
    let matchMadeByPlayer = false;
    if (playerMoveFormsImmediateTriple) {
        matchMadeByPlayer = true;
        const symbolToMatch = playerClickedTile.symbol;
        let count = 0;
        tempSlotAfterPlayer = tempSlotAfterPlayer.filter(t => {
            if (t.symbol === symbolToMatch && count < TILES_PER_MATCH_SET) {
                count++;
                return false; 
            }
            return true;
        });
    }
    const simulatedBoardValueAfterPlayerMove = calculateSimulatedBoardValue(
        playerMoveInPolicy ? playerMoveInPolicy.score : null, 
        tempSlotAfterPlayer, 
        COLLECTION_SLOT_CAPACITY, 
        matchMadeByPlayer ? tempBoardAfterPlayer.map(t => t.symbol === playerClickedTile.symbol ? {...t, isMatched:true}: t) : tempBoardAfterPlayer, 
        TILES_PER_MATCH_SET
    );
    
    const initialBoardValue = calculateSimulatedBoardValue(
        bestAgentPolicyMove ? bestAgentPolicyMove.score : null,
        collectionSlotStateBeforePlayerMove,
        COLLECTION_SLOT_CAPACITY,
        currentBoardTilesForSim,
        TILES_PER_MATCH_SET
    );

    if (collectionSlotStateBeforePlayerMove.length >= COLLECTION_SLOT_CAPACITY -1 && !playerMoveFormsImmediateTriple) {
         evaluation = "Stupid";
         reasoning = agentStatusPrefix + `Critical error! Adding ${playerTileSymbol} filled the slot without a match, leading to a loss. Board value plummeted to ${simulatedBoardValueAfterPlayerMove.toFixed(2)}.`;
         if (bestAgentPolicyMove && bestAgentPolicyMove.tile.id !== playerTileId && bestAgentPolicyMove.confidence > 0.1) { 
             const firstMoveForLookahead = fullBoardSnapshotBeforePlayerMove.find(t => t.id === bestAgentPolicyMove.tile.id);
             let predictedLookaheadSequence: PredictedMove[] | undefined = undefined;
             if (firstMoveForLookahead) {
                predictedLookaheadSequence = this.simulateLookahead(fullBoardSnapshotBeforePlayerMove, collectionSlotStateBeforePlayerMove, firstMoveForLookahead, currentDifficultySettings, 2);
             }
             missedOpportunity = { 
                 tileId: bestAgentPolicyMove.tile.id, 
                 tileSymbol: bestAgentPolicyMove.tile.symbol, 
                 reasoning: `Policy network preferred ${bestAgentPolicyMove.tile.symbol} (Confidence: ${(bestAgentPolicyMove.confidence * 100).toFixed(0)}%), which ${bestAgentPolicyMove.desc}.`,
                 simulatedPolicyConfidence: bestAgentPolicyMove.confidence,
                 predictedSequence: predictedLookaheadSequence
                };
         }
         return { evaluation, agentReasoning: reasoning, missedOpportunity, simulatedPlayerMovePolicyConfidence: playerMoveSimulatedConfidence, simulatedBoardValueAfterPlayerMove };
    }

    if (playerMoveFormsImmediateTriple) {
        let isGenius = false;
        // Stricter Genius conditions
        const playerMoveScore = playerMoveInPolicy?.score ?? 0;

        if (collectionSlotStateBeforePlayerMove.length >= COLLECTION_SLOT_CAPACITY - 1 && playerMoveScore >= 102) {
            // Critical save. How many ways to make *any* triple were there?
            const allTripleMakingMoves = rawAgentMoves.filter(m => m.score >= 100);
            if (allTripleMakingMoves.length <= 2) { // Player found one of few solutions
                isGenius = true;
            }
        } else if (playerMoveScore >= 115) { // Exceptionally high-impact triple
             const bestNonTripleScore = Math.max(0, ...rawAgentMoves.filter(m => m.score < 100 && m.tile.id !== playerTileId).map(m => m.score));
             if (playerMoveScore > bestNonTripleScore + 30) { // Significantly better than other options
                isGenius = true;
             }
        }
        
        if (isGenius) {
            evaluation = "Genius";
            reasoning = agentStatusPrefix + `Genius! A critical and high-impact triple with ${playerTileSymbol}. Policy Confidence: ${(playerMoveSimulatedConfidence * 100).toFixed(0)}%. Board value shifted from ${initialBoardValue.toFixed(2)} to ${simulatedBoardValueAfterPlayerMove.toFixed(2)}.`;
        } else {
            evaluation = "Good";
            reasoning = agentStatusPrefix + `Excellent! Completing the triple for ${playerTileSymbol}. Policy Confidence: ${(playerMoveSimulatedConfidence * 100).toFixed(0)}%. Board value: ${simulatedBoardValueAfterPlayerMove.toFixed(2)}.`;
        }
    
    } else if (bestAgentPolicyMove && bestAgentPolicyMove.tile.id !== playerTileId && bestAgentPolicyMove.confidence > playerMoveSimulatedConfidence + 0.25 ) { 
        const firstMoveForLookahead = fullBoardSnapshotBeforePlayerMove.find(t => t.id === bestAgentPolicyMove.tile.id);
        let predictedLookaheadSequence: PredictedMove[] | undefined = undefined;
        
        const valueDrop = initialBoardValue - simulatedBoardValueAfterPlayerMove;
        if (valueDrop > 0.3 || bestAgentPolicyMove.confidence > 0.5 && playerMoveSimulatedConfidence < 0.15) {
            evaluation = "Bad";
            reasoning = agentStatusPrefix + `Suboptimal move with ${playerTileSymbol} (Confidence: ${(playerMoveSimulatedConfidence * 100).toFixed(0)}%). Missed a stronger play. Projected board value: ${simulatedBoardValueAfterPlayerMove.toFixed(2)}.`;
        } else {
            evaluation = "Average";
            reasoning = agentStatusPrefix + `Player's choice of ${playerTileSymbol} (Confidence: ${(playerMoveSimulatedConfidence * 100).toFixed(0)}%) was acceptable. Projected board value: ${simulatedBoardValueAfterPlayerMove.toFixed(2)}. Agent saw a slightly better option.`;
        }
        
        if (firstMoveForLookahead && (bestAgentPolicyMove.confidence > 0.3 || evaluation === "Bad")) { 
           predictedLookaheadSequence = this.simulateLookahead(fullBoardSnapshotBeforePlayerMove, collectionSlotStateBeforePlayerMove, firstMoveForLookahead, currentDifficultySettings, 2);
        }
         missedOpportunity = { 
            tileId: bestAgentPolicyMove.tile.id, 
            tileSymbol: bestAgentPolicyMove.tile.symbol, 
            reasoning: `Policy network preferred ${bestAgentPolicyMove.tile.symbol} (ID: ${bestAgentPolicyMove.tile.id.substring(0,4)}) with ${(bestAgentPolicyMove.confidence * 100).toFixed(0)}% confidence because it ${bestAgentPolicyMove.desc}`,
            simulatedPolicyConfidence: bestAgentPolicyMove.confidence,
            predictedSequence: predictedLookaheadSequence
        };

    } else if (slotSymbolCountsBeforePlayerMove[playerTileSymbol]?.count === TILES_PER_MATCH_SET - 2 && collectionSlotStateBeforePlayerMove.length < COLLECTION_SLOT_CAPACITY -1) { 
        evaluation = "Good";
        reasoning = agentStatusPrefix + `Good setup! Forming a pair for ${playerTileSymbol}. Policy Confidence: ${(playerMoveSimulatedConfidence * 100).toFixed(0)}%. Board value: ${simulatedBoardValueAfterPlayerMove.toFixed(2)}.`;
    } else { 
        if (playerTileId === bestAgentPolicyMove?.tile.id && playerMoveSimulatedConfidence > 0.2) { 
            evaluation = "Good"; 
            reasoning = agentStatusPrefix + `Solid move. Picking ${playerTileSymbol} aligns with policy network's top choice (Confidence: ${(playerMoveSimulatedConfidence * 100).toFixed(0)}%). Board value: ${simulatedBoardValueAfterPlayerMove.toFixed(2)}.`;
        } else if (simulatedBoardValueAfterPlayerMove > initialBoardValue + 0.1 && playerMoveSimulatedConfidence > 0.15) {
            evaluation = "Good";
            reasoning = agentStatusPrefix + `A thoughtful move with ${playerTileSymbol}, improving board value to ${simulatedBoardValueAfterPlayerMove.toFixed(2)}. Policy Confidence: ${(playerMoveSimulatedConfidence * 100).toFixed(0)}%.`;
        }
         else if (playerMoveSimulatedConfidence > 0.1) {
            evaluation = "Average";
            reasoning = agentStatusPrefix + `Reasonable move with ${playerTileSymbol}. Policy Confidence: ${(playerMoveSimulatedConfidence * 100).toFixed(0)}%. Board value: ${simulatedBoardValueAfterPlayerMove.toFixed(2)}.`;
        } else {
            evaluation = "Average";
            reasoning = agentStatusPrefix + `Move with ${playerTileSymbol} is acceptable but low confidence (${(playerMoveSimulatedConfidence * 100).toFixed(0)}%). Board value: ${simulatedBoardValueAfterPlayerMove.toFixed(2)}. Consider higher impact plays.`;
        }
    }
    
    if (evaluation !== "Genius" && bestAgentPolicyMove && bestAgentPolicyMove.score >= 100 && playerMoveSimulatedConfidence < 0.1 && !playerMoveFormsImmediateTriple) {
        evaluation = "Bad"; 
        reasoning = agentStatusPrefix + `Missed a clear winning opportunity! Policy network strongly indicated a triple with ${bestAgentPolicyMove.tile.symbol} (Confidence: ${(bestAgentPolicyMove.confidence * 100).toFixed(0)}%).`;
        const firstMoveForLookahead = fullBoardSnapshotBeforePlayerMove.find(t => t.id === bestAgentPolicyMove.tile.id);
        let predictedLookaheadSequence: PredictedMove[] | undefined = undefined;
        if (firstMoveForLookahead) {
           predictedLookaheadSequence = this.simulateLookahead(fullBoardSnapshotBeforePlayerMove, collectionSlotStateBeforePlayerMove, firstMoveForLookahead, currentDifficultySettings, 2);
        }
        missedOpportunity = { 
            tileId: bestAgentPolicyMove.tile.id, 
            tileSymbol: bestAgentPolicyMove.tile.symbol, 
            reasoning: `Policy network strongly preferred ${bestAgentPolicyMove.tile.symbol} (ID: ${bestAgentPolicyMove.tile.id.substring(0,4)}) which ${bestAgentPolicyMove.desc}`,
            simulatedPolicyConfidence: bestAgentPolicyMove.confidence,
            predictedSequence: predictedLookaheadSequence
        };
    }
    return { evaluation, agentReasoning: reasoning, missedOpportunity, simulatedPlayerMovePolicyConfidence: playerMoveSimulatedConfidence, simulatedBoardValueAfterPlayerMove };
  }
};


const App: React.FC = () => {
  const [currentDifficulty, setCurrentDifficulty] = useState<DifficultyLevel>('medium');
  const [allTiles, setAllTiles] = useState<TileData[]>([]);
  const [collectedTileIds, setCollectedTileIds] = useState<string[]>([]);
  const [gameState, setGameState] = useState<GameState>('loading');
  const [boardDimensions, setBoardDimensions] = useState({ width: 0, height: 0 });
  const [moveHistory, setMoveHistory] = useState<MoveRecord[]>([]);
  
  const [localAnalysisResults, setLocalAnalysisResults] = useState<LocalAgentMoveEvaluation[] | null>(null);
  const [isPerformingLocalAnalysis, setIsPerformingLocalAnalysis] = useState<boolean>(false);
  const [localAnalysisError, setLocalAnalysisError] = useState<string | null>(null);
  const [showAnalysisModal, setShowAnalysisModal] = useState<boolean>(false);

  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingProgress, setTrainingProgress] = useState<number>(0);
  const [trainingMessage, setTrainingMessage] = useState<string | null>(null);

  // --- Match History States ---
  const [savedGames, setSavedGames] = useState<SavedGameRecord[]>([]);
  const [showMatchHistoryModal, setShowMatchHistoryModal] = useState<boolean>(false);
  const [isCurrentGameSaved, setIsCurrentGameSaved] = useState<boolean>(false);
  
  // --- Post-Match Review States ---
  const [showPostMatchReviewModal, setShowPostMatchReviewModal] = useState<boolean>(false);

  // --- Helper Ability States ---
  const [canShuffle, setCanShuffle] = useState<boolean>(true);
  const [canRemove, setCanRemove] = useState<boolean>(true);


  useEffect(() => {
    localRLAgent.initialize();
    // Load saved games from localStorage on initial mount
    try {
        const storedGames = localStorage.getItem(LOCAL_STORAGE_KEY_SAVED_GAMES);
        if (storedGames) {
            setSavedGames(JSON.parse(storedGames));
        }
    } catch (error) {
        console.error("Error loading saved games from localStorage:", error);
        setSavedGames([]); // Reset to empty if parsing fails
    }
  }, []);

  const generateInitialTiles = useCallback((difficultySettings: DifficultySettings): TileData[] => {
    const { NUM_UNIQUE_SYMBOLS_TO_USE, NUM_SETS_PER_SYMBOL, TILES_PER_MATCH_SET, TOTAL_TILES_TO_GENERATE, LAYERS_CONFIG } = difficultySettings;

    const selectedSymbols = shuffleArray(SYMBOLS).slice(0, NUM_UNIQUE_SYMBOLS_TO_USE);
    let tileDeck: SymbolString[] = [];
    selectedSymbols.forEach(symbol => {
      for (let i = 0; i < NUM_SETS_PER_SYMBOL * TILES_PER_MATCH_SET; i++) {
        tileDeck.push(symbol);
      }
    });

    if (tileDeck.length < TOTAL_TILES_TO_GENERATE) {
        console.warn(`Tile deck (length ${tileDeck.length}) is smaller than TOTAL_TILES_TO_GENERATE (${TOTAL_TILES_TO_GENERATE}). Padding...`);
        let i = 0;
        while(tileDeck.length < TOTAL_TILES_TO_GENERATE) {
            tileDeck.push(selectedSymbols[i % selectedSymbols.length]);
            i++;
        }
    } else if (tileDeck.length > TOTAL_TILES_TO_GENERATE) {
        console.warn(`Tile deck (length ${tileDeck.length}) is larger than TOTAL_TILES_TO_GENERATE (${TOTAL_TILES_TO_GENERATE}). Truncating...`);
        tileDeck = tileDeck.slice(0, TOTAL_TILES_TO_GENERATE);
    }
    
    const shuffledDeck = shuffleArray(tileDeck);
    const generatedTiles: TileData[] = [];
    let tileIndex = 0;
    let maxBoardWidth = 0;
    let maxBoardHeight = 0;

    LAYERS_CONFIG.forEach(layerConfig => {
      for (let i = 0; i < layerConfig.tileCount; i++) {
        if (tileIndex >= shuffledDeck.length) {
            console.error("Ran out of tiles in shuffledDeck before filling all layers. Check TOTAL_TILES_TO_GENERATE and LAYERS_CONFIG sum.");
            break; 
        }

        const col = i % layerConfig.cols;
        const row = Math.floor(i / layerConfig.cols);

        const x = col * TILE_WIDTH + layerConfig.xOffset;
        const y = row * TILE_HEIGHT + layerConfig.yOffset;

        generatedTiles.push({
          id: generateId(),
          symbol: shuffledDeck[tileIndex],
          x,
          y,
          layer: layerConfig.layerZ,
          isCollected: false,
          isMatched: false,
        });
        tileIndex++;
        maxBoardWidth = Math.max(maxBoardWidth, x + TILE_WIDTH);
        maxBoardHeight = Math.max(maxBoardHeight, y + TILE_HEIGHT);
      }
      if (tileIndex >= shuffledDeck.length && layerConfig.tileCount > 0 && LAYERS_CONFIG.indexOf(layerConfig) < LAYERS_CONFIG.length -1 ) {
            console.error("Ran out of tiles mid-layer generation for difficulty:", difficultySettings.NAME);
      }
    });
    if (tileIndex < TOTAL_TILES_TO_GENERATE) {
        console.error(`Generated ${tileIndex} tiles, but expected ${TOTAL_TILES_TO_GENERATE} for difficulty ${difficultySettings.NAME}. Layer config might not sum up correctly or ran out of symbols.`);
    }

    setBoardDimensions({ width: maxBoardWidth + BOARD_PADDING * 2, height: maxBoardHeight + BOARD_PADDING * 2});
    return generatedTiles;
  }, []);

  const resetGame = useCallback((newDifficulty?: DifficultyLevel) => {
    const difficultyToUse = newDifficulty || currentDifficulty;
    setIsCurrentGameSaved(false); // Reset save status for the new game
    setCanShuffle(true); // Reset shuffle availability
    setCanRemove(true);  // Reset remove availability
    
    if (newDifficulty && newDifficulty !== currentDifficulty) {
        setCurrentDifficulty(newDifficulty); 
    } else {
        setGameState('loading');
        const currentSettings = DIFFICULTY_LEVELS[difficultyToUse];
        setAllTiles(generateInitialTiles(currentSettings));
        setCollectedTileIds([]);
        setMoveHistory([]);
        setLocalAnalysisResults(null);
        setIsPerformingLocalAnalysis(false);
        setShowAnalysisModal(false);
        setLocalAnalysisError(null);
        setTrainingMessage(null); 
        setGameState('playing');
    }
  }, [currentDifficulty, generateInitialTiles]);


  useEffect(() => {
    setGameState('loading');
    const currentSettings = DIFFICULTY_LEVELS[currentDifficulty];
    setAllTiles(generateInitialTiles(currentSettings));
    setCollectedTileIds([]);
    setMoveHistory([]);
    setLocalAnalysisResults(null);
    setIsPerformingLocalAnalysis(false);
    setShowAnalysisModal(false);
    setLocalAnalysisError(null);
    setIsCurrentGameSaved(false); // Ensure save status is reset when difficulty changes directly
    setCanShuffle(true); // Reset shuffle on difficulty change
    setCanRemove(true);  // Reset remove on difficulty change
    setGameState('playing');
  }, [currentDifficulty, generateInitialTiles]);


  const processCollectionSlot = useCallback((
    currentAllTiles: ReadonlyArray<TileData>,
    currentCollectedTileIds: ReadonlyArray<string>,
    idOfTileJustClicked: string,
    boardSnapshotForHistory: ReadonlyArray<TileDataInSnapshot>,
    slotSnapshotForHistory: ReadonlyArray<TileData>,
    originalClickedTileDataForHistory: Readonly<TileData>
  ) => {
    const clickedTile = currentAllTiles.find(t => t.id === idOfTileJustClicked);
    if (!clickedTile) {
      console.error("Clicked tile not found in currentAllTiles during processCollectionSlot.");
      return;
    }

    let newAllTilesState = currentAllTiles.map(t =>
      t.id === idOfTileJustClicked ? { ...t, isCollected: true, isMatched: false } : t
    );

    let orderedCollectedIds: string[] = [];
    const existingIdsInSlot = [...currentCollectedTileIds];
    const clickedSymbol = clickedTile.symbol;

    let lastSimilarSymbolIndex = -1;
    for (let i = existingIdsInSlot.length - 1; i >= 0; i--) {
      const existingTileId = existingIdsInSlot[i];
      const tileInSlot = currentAllTiles.find(t => t.id === existingTileId);
      if (tileInSlot && tileInSlot.symbol === clickedSymbol) {
        lastSimilarSymbolIndex = i;
        break;
      }
    }

    if (lastSimilarSymbolIndex !== -1) {
      orderedCollectedIds = [
        ...existingIdsInSlot.slice(0, lastSimilarSymbolIndex + 1),
        idOfTileJustClicked,
        ...existingIdsInSlot.slice(lastSimilarSymbolIndex + 1)
      ];
    } else {
      orderedCollectedIds = [...existingIdsInSlot, idOfTileJustClicked];
    }
    
    let finalCollectedIdsInSlot = orderedCollectedIds;
    let matchMadeThisTurn = false;

    const difficultySettings = DIFFICULTY_LEVELS[currentDifficulty];
    const TILES_PER_MATCH_SET = difficultySettings.TILES_PER_MATCH_SET;
    const COLLECTION_SLOT_CAPACITY = difficultySettings.COLLECTION_SLOT_CAPACITY;

    const currentSlotObjects = finalCollectedIdsInSlot
        .map(id => newAllTilesState.find(t => t.id === id))
        .filter(Boolean) as TileData[];

    const symbolCountsInSlot: Record<SymbolString, string[]> = {};
    currentSlotObjects.forEach(tileInSlot => {
      if (!symbolCountsInSlot[tileInSlot.symbol]) {
        symbolCountsInSlot[tileInSlot.symbol] = [];
      }
      symbolCountsInSlot[tileInSlot.symbol].push(tileInSlot.id);
    });

    for (const symbol in symbolCountsInSlot) {
      if (symbolCountsInSlot[symbol].length >= TILES_PER_MATCH_SET) {
        matchMadeThisTurn = true;
        const idsToMatchAndRemoveFromSlot = symbolCountsInSlot[symbol].slice(0, TILES_PER_MATCH_SET);
        
        finalCollectedIdsInSlot = finalCollectedIdsInSlot.filter(id => !idsToMatchAndRemoveFromSlot.includes(id));
        newAllTilesState = newAllTilesState.map(tile =>
          idsToMatchAndRemoveFromSlot.includes(tile.id) ? { ...tile, isMatched: true, isCollected: false } : tile
        );
        break; 
      }
    }
    
    const agentLogicAccessibleTiles = boardSnapshotForHistory
        .filter(t => t.isAccessibleInSnapshot) 
        .map(t => ({ id: t.id, symbol: t.symbol, layer: t.layer }));

    const agentLogicTilesOnBoardUncollected = boardSnapshotForHistory.filter(t => !t.isCollected && !t.isMatched); 
    const agentLogicTilesBySymbol: Record<SymbolString, number> = {};
    agentLogicTilesOnBoardUncollected.forEach(t => {
        agentLogicTilesBySymbol[t.symbol] = (agentLogicTilesBySymbol[t.symbol] || 0) + 1;
    });
    
    // FIX: Ensure all properties assigned to MoveRecord match their expected mutable types
    // Specifically, spread ReadonlyArray to create mutable arrays and Readonly<Object> for objects.
    setMoveHistory(prevHistory => [...prevHistory, {
        moveNumber: prevHistory.length + 1,
        tileClicked: { ...originalClickedTileDataForHistory },
        tilesOnBoardSnapshot: [...boardSnapshotForHistory], 
        collectionSlotSnapshot: [...slotSnapshotForHistory], 
        boardStateBeforeMove: { 
            accessibleTiles: agentLogicAccessibleTiles, // Already a new array from .map
            totalTilesOnBoard: agentLogicTilesOnBoardUncollected.length,
            tilesBySymbol: agentLogicTilesBySymbol,
        },
        collectionSlotBeforeMove: [...slotSnapshotForHistory], // This was the primary source of the error
        matchMade: matchMadeThisTurn, 
        collectionSlotAfterMove: finalCollectedIdsInSlot
            .map(id => newAllTilesState.find(t => t.id === id))
            .filter(Boolean) as TileData[], // Already a new array
    }]);

    setAllTiles(newAllTilesState);
    setCollectedTileIds(finalCollectedIdsInSlot);

    const remainingUnmatchedTiles = newAllTilesState.filter(tile => !tile.isMatched).length;
    if (remainingUnmatchedTiles === 0 && finalCollectedIdsInSlot.length === 0) {
      setGameState('won');
    } else if (finalCollectedIdsInSlot.length >= COLLECTION_SLOT_CAPACITY && !matchMadeThisTurn) { 
      setGameState('lost');
    }
  }, [currentDifficulty, setMoveHistory, setAllTiles, setCollectedTileIds, setGameState]);

  const handleTileClick = useCallback((tileId: string) => {
    if (gameState !== 'playing' || isTraining) return; 

    const tileToClick = allTiles.find(t => t.id === tileId);
    if (!tileToClick || tileToClick.isCollected || tileToClick.isMatched) return;

    if (!checkTileAccessibilityGlobal(tileId, allTiles)) return;

    const boardSnapshotForHistory: TileDataInSnapshot[] = allTiles
      .map(t => ({
        ...t, 
        isAccessibleInSnapshot: (!t.isMatched && !t.isCollected) ? checkTileAccessibilityGlobal(t.id, allTiles) : false
      }));
      
    const slotSnapshotForHistory: TileData[] = collectedTileIds
      .map(id => allTiles.find(t => t.id === id)) 
      .filter(Boolean) as TileData[];

    const clickedTileObjectForHistory = {...tileToClick}; 
      
    processCollectionSlot(
        allTiles,
        collectedTileIds,
        tileId, // ID of the tile just clicked
        boardSnapshotForHistory,
        slotSnapshotForHistory,
        clickedTileObjectForHistory
    );
    
  }, [allTiles, collectedTileIds, gameState, processCollectionSlot, isTraining]);

  const handleShowPostMatchReview = useCallback(async () => {
    console.log('Post-match review requested, analysis results:', localAnalysisResults);
    console.log('Move history:', moveHistory);
    
    // Run the analysis first if we don't already have analysis results
    if (!localAnalysisResults) {
      console.log('No existing analysis results, generating new analysis');
      
      if (!localRLAgent.isInitialized) {
        alert("Local agent is not ready. Please wait or try calibrating the agent.");
        return;
      }
      if (!moveHistory || moveHistory.length === 0) {
        alert("No moves recorded to analyze.");
        return;
      }
      
      // Show loading state
      setIsPerformingLocalAnalysis(true);
      
      try {
        // Perform the analysis for all moves
        console.log("Running post-match analysis for all moves...");
        const currentDifficultySettings = DIFFICULTY_LEVELS[currentDifficulty];
        
        // First check if the backend API is available
        let healthStatus;
        try {
          healthStatus = await ApiClient.checkHealth();
          console.log("Backend API health status:", healthStatus);
        } catch (error) {
          console.warn("Backend API health check failed:", error);
          healthStatus = { rationale_model_loaded: false };
        }
        
        // Get analysis results for each move
        const analysisResults = await Promise.all(
          moveHistory.map(async (move, index) => {
            try {
              console.log(`Analyzing move ${index + 1}/${moveHistory.length}`);
              const playerClickedTileInfo = move.tileClicked;
              let agentEval: any = null;
              
              // Try to use the backend rationale predictor first if available
              if (healthStatus?.rationale_model_loaded) {
                try {
                  console.log('Using backend rationale predictor for analysis');
                  
                  // Prepare data for API request (same as in handleRequestLocalAnalysis)
                  const moveData = {
                    difficulty: currentDifficulty,
                    turn_number: move.moveNumber,
                    move_tile_symbol: move.tileClicked.symbol,
                    move_tile_layer: move.tileClicked.layer,
                    gs_board_tile_count: move.tilesOnBoardSnapshot.filter((t: any) => !t.isCollected && !t.isMatched).length,
                    gs_accessible_tile_count: move.boardStateBeforeMove.accessibleTiles.length,
                    gs_collection_fill_ratio: move.collectionSlotBeforeMove.length / currentDifficultySettings.COLLECTION_SLOT_CAPACITY,
                    collection_slot_contents: move.collectionSlotBeforeMove.map((t: any) => t.symbol),
                    board_tiles_by_symbol: Object.fromEntries(
                      Array.from(new Set(move.tilesOnBoardSnapshot
                        .filter((t: any) => !t.isCollected && !t.isMatched)
                        .map((t: any) => t.symbol)))
                        .map(symbol => [
                          symbol, 
                          move.tilesOnBoardSnapshot.filter((t: any) => !t.isCollected && !t.isMatched && t.symbol === symbol).length
                        ])
                    ),
                    win_loss: gameState === 'won' ? 1 : (gameState === 'lost' ? 0 : 0.5)
                  };
                  
                  // Get rationale prediction
                  const rationaleResult = await ApiClient.getRationalePrediction(moveData);
                  console.log('Rationale result:', rationaleResult);
                  
                  if (!rationaleResult.error) {
                    // Convert the rationale API response to the format expected by UI
                    const category = rationaleResult.category || 'Average';
                    let evaluationType: MoveEvaluationCategory = 'Average';
                    
                    if (['Brilliant', 'Good'].includes(category)) evaluationType = 'Good';
                    else if (category === 'Strategic') evaluationType = 'Average';
                    else evaluationType = 'Bad';
                    
                    agentEval = {
                      evaluation: evaluationType,
                      agentReasoning: `${category}: ${rationaleResult.rationale}`,
                      score: rationaleResult.normalized_score || 5.0
                    };
                  } else {
                    console.warn("Error from rationale API:", rationaleResult.error);
                    agentEval = null;
                  }
                } catch (apiError) {
                  console.error('Error with rationale API:', apiError);
                  agentEval = null;
                }
              }
              
              // Fall back to local agent if API call failed or model not available
              if (!agentEval) {
                console.log('Using local agent for analysis (fallback)');
                agentEval = localRLAgent.evaluatePlayerMove(
                  move.boardStateBeforeMove, 
                  move.tilesOnBoardSnapshot, 
                  move.collectionSlotBeforeMove, 
                  playerClickedTileInfo,
                  currentDifficultySettings
                );
              }
              
              return {
                moveNumber: move.moveNumber,
                playerTileClickedSymbol: move.tileClicked.symbol,
                playerTileClickedId: move.tileClicked.id,
                evaluation: agentEval.evaluation,
                agentReasoning: agentEval.agentReasoning,
                missedOpportunity: agentEval.missedOpportunity,
                simulatedPlayerMovePolicyConfidence: agentEval.simulatedPlayerMovePolicyConfidence || 0,
                simulatedBoardValueAfterPlayerMove: agentEval.simulatedBoardValueAfterPlayerMove || 0,
              };
            } catch (moveError) {
              console.error(`Error analyzing move ${index + 1}:`, moveError);
              // Return a default analysis result for this move to prevent the entire Promise.all from failing
              return {
                moveNumber: move.moveNumber,
                playerTileClickedSymbol: move.tileClicked.symbol,
                playerTileClickedId: move.tileClicked.id,
                evaluation: 'Average' as MoveEvaluationCategory,
                agentReasoning: 'Unable to analyze this move.',
                missedOpportunity: undefined,
                simulatedPlayerMovePolicyConfidence: 0,
                simulatedBoardValueAfterPlayerMove: 0,
              };
            }
          })
        );
        
        console.log("Analysis complete. Setting results:", analysisResults);
        setLocalAnalysisResults(analysisResults);
      } catch (error) {
        console.error("Error during post-match analysis:", error);
        alert("Failed to generate post-match analysis. Please try again.");
        return;
      } finally {
        setIsPerformingLocalAnalysis(false);
      }
    } else {
      console.log("Using existing analysis results:", localAnalysisResults);
    }
    
    // Now show the post-match review modal
    setShowPostMatchReviewModal(true);
  }, [localAnalysisResults, localRLAgent, moveHistory, currentDifficulty, gameState, DIFFICULTY_LEVELS]);

  const handleRequestLocalAnalysis = useCallback(async () => {
    if (!localRLAgent.isInitialized) {
        setLocalAnalysisError("Local Agent is not ready.");
        setShowAnalysisModal(true);
        return;
    }
    if (!moveHistory || moveHistory.length === 0) {
        setLocalAnalysisError("No moves recorded to analyze.");
        setShowAnalysisModal(true);
        return;
    }

    setIsPerformingLocalAnalysis(true);
    setLocalAnalysisError(null);
    setLocalAnalysisResults(null); // Clear previous results before new analysis

    await new Promise(resolve => setTimeout(resolve, 300)); 
    const currentDifficultySettings = DIFFICULTY_LEVELS[currentDifficulty];

    try {
      // First check if the backend API is available
      const healthStatus = await ApiClient.checkHealth();
      
      // Get analysis results for each move
      const analysisResults: LocalAgentMoveEvaluation[] = await Promise.all(
        moveHistory.map(async (move) => {
          const playerClickedTileInfo: TileData = move.tileClicked;
          let agentEval: any = null;
          
          // Try to use the backend rationale predictor first if available
          if (healthStatus.rationale_model_loaded) {
            try {
              console.log('Using backend rationale predictor for analysis');
              
              // Prepare data for API request
              const moveData = {
                difficulty: currentDifficulty,
                turn_number: move.moveNumber,
                move_tile_symbol: move.tileClicked.symbol,
                move_tile_layer: move.tileClicked.layer,
                gs_board_tile_count: move.tilesOnBoardSnapshot.filter(t => !t.isCollected && !t.isMatched).length,
                gs_accessible_tile_count: move.boardStateBeforeMove.accessibleTiles.length,
                gs_collection_fill_ratio: move.collectionSlotBeforeMove.length / currentDifficultySettings.COLLECTION_SLOT_CAPACITY,
                collection_slot_contents: move.collectionSlotBeforeMove.map(t => t.symbol),
                // Convert board tiles to expected format
                board_tiles_by_symbol: Object.fromEntries(
                  Array.from(new Set(move.tilesOnBoardSnapshot
                    .filter(t => !t.isCollected && !t.isMatched)
                    .map(t => t.symbol)))
                    .map(symbol => [
                      symbol, 
                      move.tilesOnBoardSnapshot.filter(t => !t.isCollected && !t.isMatched && t.symbol === symbol).length
                    ])
                ),
                win_loss: gameState === 'won' ? 1 : (gameState === 'lost' ? 0 : 0.5)
              };
              
              // Get rationale prediction
              const rationaleResult = await ApiClient.getRationalePrediction(moveData);
              
              if (!rationaleResult.error) {
                // Convert the rationale API response to the format expected by UI
                const category = rationaleResult.category || 'Average';
                let evaluationType: MoveEvaluationCategory = 'Average';
                
                if (['Brilliant', 'Good'].includes(category)) evaluationType = 'Good';
                else if (category === 'Strategic') evaluationType = 'Average';
                else evaluationType = 'Bad';
                
                agentEval = {
                  evaluation: evaluationType,
                  agentReasoning: `${category}: ${rationaleResult.rationale}`,
                  score: rationaleResult.normalized_score || 5.0
                };
              }
            } catch (apiError) {
              console.error('Error with rationale API:', apiError);
              // Fall back to local agent on error
              agentEval = null;
            }
          }
          
          // Fall back to local agent if API call failed or model not available
          if (!agentEval) {
            console.log('Using local agent for analysis (fallback)');
            agentEval = localRLAgent.evaluatePlayerMove(
              move.boardStateBeforeMove, 
              move.tilesOnBoardSnapshot, 
              move.collectionSlotBeforeMove, 
              playerClickedTileInfo,
              currentDifficultySettings
            );
          }
          
          return {
            moveNumber: move.moveNumber,
            playerTileClickedSymbol: move.tileClicked.symbol,
            playerTileClickedId: move.tileClicked.id,
            evaluation: agentEval.evaluation,
            agentReasoning: agentEval.agentReasoning,
            missedOpportunity: agentEval.missedOpportunity,
            simulatedPlayerMovePolicyConfidence: agentEval.simulatedPlayerMovePolicyConfidence,
            simulatedBoardValueAfterPlayerMove: agentEval.simulatedBoardValueAfterPlayerMove,
          };
        })
      );
      
      setLocalAnalysisResults(analysisResults); // Set new results
    } catch (error) {
      console.error("Error during local analysis:", error);
      setLocalAnalysisError(`Failed to get analysis: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsPerformingLocalAnalysis(false);
      setShowAnalysisModal(true);
    }
  }, [moveHistory, currentDifficulty, gameState]);

  const handleStartTraining = async () => {
    if (isTraining) return;
    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingMessage("Calibrating simulated NN model...");
    
    if (!localRLAgent.isInitialized) {
        await localRLAgent.initialize();
    }

    try {
        const resultMessage = await localRLAgent.simulateTraining(100, (progress) => {
            setTrainingProgress(progress);
        });
        
        if (moveHistory.length > 0 && (gameState === 'won' || gameState === 'lost')) { // Check if game has ended
            setTrainingMessage(resultMessage + " Model ready. Opening review for the last game...");
            await handleRequestLocalAnalysis(); 
        } else {
            setTrainingMessage(resultMessage + " Model ready. Play a game to see its review!");
        }
    } catch (error) {
        console.error("Error during training simulation:", error);
        const errorMessage = error instanceof Error ? error.message : String(error);
        setTrainingMessage(`Simulated training failed: ${errorMessage}`);
    } finally {
        setIsTraining(false);
    }
  };

  const handleSaveCurrentGame = useCallback(() => {
    if (isCurrentGameSaved || (gameState !== 'won' && gameState !== 'lost')) return;

    const gameToSave: SavedGameRecord = {
      gameId: generateId(),
      timestamp: Date.now(),
      difficulty: currentDifficulty,
      outcome: gameState, // 'won' or 'lost'
      moveHistory: [...moveHistory],
      analysisResults: localAnalysisResults ? [...localAnalysisResults] : null,
    };

    setSavedGames(prevSavedGames => {
      const updatedGames = [gameToSave, ...prevSavedGames];
      if (updatedGames.length > MAX_SAVED_GAMES) {
        updatedGames.splice(MAX_SAVED_GAMES); // Keep only the most recent MAX_SAVED_GAMES
      }
      try {
        localStorage.setItem(LOCAL_STORAGE_KEY_SAVED_GAMES, JSON.stringify(updatedGames));
      } catch (error) {
        console.error("Error saving games to localStorage:", error);
        // Potentially show a user-facing error if localStorage is full or disabled
      }
      return updatedGames;
    });
    setIsCurrentGameSaved(true);
  }, [gameState, moveHistory, localAnalysisResults, currentDifficulty, isCurrentGameSaved]);

  const handleShuffleBoard = useCallback(() => {
    if (!canShuffle || gameState !== 'playing' || isTraining) return;

    const accessibleBoardTiles = allTiles.filter(tile => 
        !tile.isCollected && 
        !tile.isMatched && 
        checkTileAccessibilityGlobal(tile.id, allTiles)
    );

    if (accessibleBoardTiles.length < 2) {
        console.log("Not enough accessible tiles to shuffle.");
        // Optionally provide user feedback here
        return;
    }

    const symbolsToShuffle = accessibleBoardTiles.map(tile => tile.symbol);
    const shuffledSymbols = shuffleArray(symbolsToShuffle);

    const newTiles = allTiles.map(tile => {
        const accessibleIndex = accessibleBoardTiles.findIndex(accTile => accTile.id === tile.id);
        if (accessibleIndex !== -1) {
            return { ...tile, symbol: shuffledSymbols[accessibleIndex] };
        }
        return tile;
    });

    setAllTiles(newTiles);
    setCanShuffle(false);
  }, [allTiles, canShuffle, gameState, isTraining]);

  const handleRemoveFromSlot = useCallback(() => {
    if (!canRemove || gameState !== 'playing' || isTraining || collectedTileIds.length === 0) return;

    const numToRemove = Math.min(3, collectedTileIds.length);
    const idsToRemove = collectedTileIds.slice(0, numToRemove);

    const newTiles = allTiles.map(tile => {
        if (idsToRemove.includes(tile.id)) {
            return { ...tile, isCollected: false }; // Effectively returns tile to board
        }
        return tile;
    });
    
    const newCollectedIds = collectedTileIds.slice(numToRemove);

    setAllTiles(newTiles);
    setCollectedTileIds(newCollectedIds);
    setCanRemove(false);

  }, [allTiles, collectedTileIds, canRemove, gameState, isTraining]);


  const collectedTilesData = collectedTileIds.map(id => allTiles.find(t => t.id === id && t.isCollected)).filter(Boolean) as TileData[];
  const currentActiveSettings = DIFFICULTY_LEVELS[currentDifficulty];

  if (gameState === 'loading' || allTiles.length === 0) {
    return <div className="flex items-center justify-center min-h-screen bg-slate-800 text-white text-2xl">Loading Game...</div>;
  }
  
  const nonMatchedTilesOnBoard = allTiles.filter(tile => !tile.isCollected && !tile.isMatched);

  return (
    <div className="flex flex-col items-center justify-between min-h-screen p-4 bg-gradient-to-br from-slate-900 to-slate-800 select-none">
      <header className="w-full flex flex-col sm:flex-row justify-between items-center p-4 space-y-2 sm:space-y-0">
        <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-500">Triple Tile Fun</h1>
        <div className="flex items-center space-x-2">
            <span className="text-sm text-slate-400">Difficulty:</span>
            {(Object.keys(DIFFICULTY_LEVELS) as DifficultyLevel[]).map((level) => (
            <button
                key={level}
                onClick={() => resetGame(level)}
                disabled={currentDifficulty === level || isTraining}
                className={`px-3 py-1 text-sm font-semibold rounded-md transition-colors duration-150
                ${currentDifficulty === level ? 'bg-pink-600 text-white cursor-default' : 'bg-slate-600 hover:bg-slate-500 text-slate-200'}
                ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
                {DIFFICULTY_LEVELS[level].NAME}
            </button>
            ))}
            <button
                onClick={() => resetGame()} 
                disabled={isTraining}
                className={`px-4 py-1.5 bg-yellow-500 hover:bg-yellow-600 text-slate-900 font-semibold rounded-lg shadow-md transition-colors duration-150 text-sm ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
                New Game
            </button>
        </div>
      </header>

      <div className="my-2 p-3 bg-slate-700/50 rounded-lg shadow-md w-full max-w-2xl">
        <h3 className="text-md font-semibold text-sky-400 mb-2 text-center">Local Agent Status</h3>
        {isTraining ? (
          <div>
            <p className="text-slate-300 text-sm text-center">Calibrating simulated NN: {localRLAgent.simulatedNNModelName}... {trainingProgress.toFixed(0)}%</p>
            <div className="w-full bg-slate-600 rounded-full h-2.5 mt-1">
              <div className="bg-sky-500 h-2.5 rounded-full transition-all duration-150" style={{ width: `${trainingProgress}%` }}></div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center">
            <button
              onClick={handleStartTraining}
              className="px-4 py-2 bg-sky-600 hover:bg-sky-700 text-white font-semibold rounded-lg shadow-md transition-colors duration-150 text-sm"
            >
              Calibrate Agent (Simulated NN)
            </button>
            {trainingMessage && <p className="text-slate-400 mt-2 text-xs text-center">{trainingMessage}</p>}
          </div>
        )}
      </div>
      
      {/* Helper Abilities Section */}
      <div className="my-2 p-3 bg-slate-700/60 rounded-lg shadow-md w-full max-w-2xl flex justify-center space-x-4">
        <button
            onClick={handleShuffleBoard}
            disabled={!canShuffle || gameState !== 'playing' || isTraining}
            className={`px-4 py-2 text-sm font-semibold rounded-lg shadow-md transition-colors duration-150
                ${!canShuffle || gameState !== 'playing' || isTraining ? 'bg-gray-500 text-gray-300 cursor-not-allowed opacity-60' : 'bg-teal-500 hover:bg-teal-600 text-white'}`}
            title={canShuffle ? "Shuffle symbols of accessible tiles (1 use)" : "Shuffle used or unavailable"}
        >
            {canShuffle ? "Shuffle Board (1)" : "Shuffle Used"}
        </button>
        <button
            onClick={handleRemoveFromSlot}
            disabled={!canRemove || gameState !== 'playing' || isTraining || collectedTileIds.length === 0}
            className={`px-4 py-2 text-sm font-semibold rounded-lg shadow-md transition-colors duration-150
                ${!canRemove || gameState !== 'playing' || isTraining || collectedTileIds.length === 0 ? 'bg-gray-500 text-gray-300 cursor-not-allowed opacity-60' : 'bg-orange-500 hover:bg-orange-600 text-white'}`}
            title={canRemove ? "Remove up to 3 tiles from collection slot (1 use)" : "Remove used or slot empty/unavailable"}
        >
            {canRemove ? "Remove 3 From Slot (1)" : "Remove Used"}
        </button>
      </div>
      
      <main className="flex-grow flex items-center justify-center w-full overflow-hidden py-2">
         <Board 
            tiles={nonMatchedTilesOnBoard} 
            onTileClick={handleTileClick} 
            allTilesForAccessibilityCheck={allTiles} 
            boardDimensions={boardDimensions}
          />
      </main>

      <CollectionBar tiles={collectedTilesData} capacity={currentActiveSettings.COLLECTION_SLOT_CAPACITY} />

      { (gameState === 'won' || gameState === 'lost') && (
        <GameStatusModal 
            gameState={gameState} 
            onReset={() => resetGame()} 
            onShowLocalAnalysis={handleRequestLocalAnalysis}
            onShowPostMatchReview={handleShowPostMatchReview}
            onSaveMatch={handleSaveCurrentGame}
            isMatchSaved={isCurrentGameSaved}
        />
      )}
      { showAnalysisModal && localAnalysisResults && moveHistory.length > 0 && ( 
        <GameAnalysisModal
            analysisResults={localAnalysisResults}
            moveHistory={moveHistory} 
            isLoading={isPerformingLocalAnalysis}
            error={localAnalysisError}
            onClose={() => setShowAnalysisModal(false)}
            collectionSlotCapacity={currentActiveSettings.COLLECTION_SLOT_CAPACITY}
        />
      )}
      
      {showPostMatchReviewModal && (
        <PostMatchReviewModal
          moveHistory={moveHistory}
          analysisResults={localAnalysisResults}
          gameResult={gameState as 'won' | 'lost'}
          onClose={() => setShowPostMatchReviewModal(false)}
          difficulty={currentDifficulty}
        />
      )}
      
      <footer className="text-center text-xs text-slate-500 py-2">
          Tiles on board: {nonMatchedTilesOnBoard.length} | Tiles in slot: {collectedTileIds.length}/{currentActiveSettings.COLLECTION_SLOT_CAPACITY}
          | Difficulty: {currentActiveSettings.NAME}
          | Agent: {localRLAgent.simulatedNNModelName} (Acc: ~{(localRLAgent.simulatedNNAccuracy * 100).toFixed(0)}%)
      </footer>
    </div>
  );
};

export default App;
