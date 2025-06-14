
import React, { useState, useEffect } from 'react';
import { LocalAgentMoveEvaluation, MoveEvaluationCategory, MoveRecord, PredictedMove } from '../types';
import { MiniBoard } from './MiniBoard'; 

interface GameAnalysisModalProps {
  analysisResults: LocalAgentMoveEvaluation[] | null;
  moveHistory: MoveRecord[]; 
  isLoading: boolean;
  error: string | null;
  onClose: () => void;
  collectionSlotCapacity: number;
}

const getEvaluationStyles = (evaluation: MoveEvaluationCategory): { text: string; bg: string; border: string } => {
  switch (evaluation) {
    case "Genius":
      return { text: "text-purple-300", bg: "bg-purple-700/50", border: "border-purple-500" };
    case "Good":
      return { text: "text-green-300", bg: "bg-green-700/50", border: "border-green-500" };
    case "Average":
      return { text: "text-yellow-300", bg: "bg-yellow-700/50", border: "border-yellow-500" };
    case "Bad":
      return { text: "text-orange-300", bg: "bg-orange-700/50", border: "border-orange-500" };
    case "Stupid":
      return { text: "text-red-300", bg: "bg-red-700/50", border: "border-red-500" };
    case "Info":
       return { text: "text-sky-300", bg: "bg-sky-700/50", border: "border-sky-500" };
    default:
      return { text: "text-slate-300", bg: "bg-slate-600/50", border: "border-slate-500" };
  }
};

export const GameAnalysisModal: React.FC<GameAnalysisModalProps> = ({ 
    analysisResults, 
    moveHistory, 
    isLoading, 
    error, 
    onClose, 
    collectionSlotCapacity 
}) => {
  const [activeLookaheadSequence, setActiveLookaheadSequence] = useState<PredictedMove[] | null>(null);
  const [currentLookaheadStepIndex, setCurrentLookaheadStepIndex] = useState<number>(-1);
  const [baseMoveRecordForLookahead, setBaseMoveRecordForLookahead] = useState<MoveRecord | null>(null);
  const [displayLookaheadForMoveNumber, setDisplayLookaheadForMoveNumber] = useState<number | null>(null);

  useEffect(() => {
    // Reset lookahead state if modal is reopened or analysisResults change
    setActiveLookaheadSequence(null);
    setCurrentLookaheadStepIndex(-1);
    setBaseMoveRecordForLookahead(null);
    setDisplayLookaheadForMoveNumber(null);
  }, [analysisResults, onClose]);


  const handleVisualizeLine = (analyzedMove: LocalAgentMoveEvaluation) => {
    const moveRecord = moveHistory.find(mh => mh.moveNumber === analyzedMove.moveNumber);
    if (moveRecord && analyzedMove.missedOpportunity?.predictedSequence) {
      setBaseMoveRecordForLookahead(moveRecord);
      setActiveLookaheadSequence(analyzedMove.missedOpportunity.predictedSequence);
      setCurrentLookaheadStepIndex(0);
      setDisplayLookaheadForMoveNumber(analyzedMove.moveNumber);
    }
  };

  const handleLookaheadNav = (direction: 'next' | 'prev') => {
    if (!activeLookaheadSequence) return;
    if (direction === 'next' && currentLookaheadStepIndex < activeLookaheadSequence.length - 1) {
      setCurrentLookaheadStepIndex(currentLookaheadStepIndex + 1);
    } else if (direction === 'prev' && currentLookaheadStepIndex > 0) {
      setCurrentLookaheadStepIndex(currentLookaheadStepIndex - 1);
    }
  };

  const handleExitVisualization = () => {
    setActiveLookaheadSequence(null);
    setCurrentLookaheadStepIndex(-1);
    // baseMoveRecordForLookahead remains to show the player's original move for that list item
    // setDisplayLookaheadForMoveNumber(null); // Keep this to know which item's context we are in
  };
  
  const getMiniBoardProps = (analyzedMove: LocalAgentMoveEvaluation) => {
    const moveRecord = moveHistory.find(mh => mh.moveNumber === analyzedMove.moveNumber);
    if (!moveRecord) return null;

    // If visualizing this specific move's lookahead sequence
    if (displayLookaheadForMoveNumber === analyzedMove.moveNumber && activeLookaheadSequence && currentLookaheadStepIndex >= 0) {
        const currentStep = activeLookaheadSequence[currentLookaheadStepIndex];
        return {
            tilesOnBoard: currentStep.newBoardSnapshot,
            collectionSlot: currentStep.newCollectionSlotSnapshot,
            playerClickedTileId: currentStep.tileIdPicked, // Highlight the tile picked in this step of the sequence
            agentSuggestedTileId: undefined, // Not showing agent's original suggestion during its own line playback
            keySuffix: `lookahead-${currentLookaheadStepIndex}` // Force re-render of MiniBoard
        };
    }
    // Default: Show player's actual move from history
    return {
        tilesOnBoard: moveRecord.tilesOnBoardSnapshot,
        collectionSlot: moveRecord.collectionSlotSnapshot,
        playerClickedTileId: analyzedMove.playerTileClickedId,
        agentSuggestedTileId: analyzedMove.missedOpportunity?.tileId,
        keySuffix: `player-${analyzedMove.moveNumber}`
    };
  };


  return (
    <div className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-[60] p-2 sm:p-4" onClick={onClose}>
      <div 
        className="bg-slate-800 p-3 sm:p-6 rounded-xl shadow-2xl border border-slate-700 max-w-md sm:max-w-lg md:max-w-2xl lg:max-w-3xl w-full max-h-[90vh] flex flex-col"
        onClick={(e) => e.stopPropagation()} 
      >
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl sm:text-2xl font-bold text-purple-400">Game Review (Simulated NN)</h2>
          <button 
            onClick={onClose} 
            className="text-slate-400 hover:text-slate-200 text-2xl sm:text-3xl"
            aria-label="Close analysis"
          >
            &times;
          </button>
        </div>

        <div className="overflow-y-auto flex-grow pr-2 custom-scrollbar"> 
          {isLoading && (
            <div className="flex flex-col items-center justify-center h-full">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-400"></div>
              <p className="text-slate-300 mt-4">Agent is reviewing the game...</p>
            </div>
          )}
          {error && (
            <div className="text-center p-4 bg-red-900/50 border border-red-700 rounded-md">
              <h3 className="text-red-400 font-semibold text-lg">Analysis Error</h3>
              <p className="text-red-300 text-sm">{error}</p>
            </div>
          )}
          {!isLoading && !error && analysisResults && analysisResults.length > 0 && moveHistory && moveHistory.length > 0 && (
            <ul className="space-y-4">
              {analysisResults.map((analyzedMove) => {
                const miniBoardDynamicProps = getMiniBoardProps(analyzedMove);
                if (!miniBoardDynamicProps) return null;

                const evalStyles = getEvaluationStyles(analyzedMove.evaluation);
                const isVisualizingThisMovesLine = displayLookaheadForMoveNumber === analyzedMove.moveNumber && activeLookaheadSequence && currentLookaheadStepIndex >= 0;

                return (
                  <li key={analyzedMove.moveNumber} className={`p-3 sm:p-4 rounded-lg border ${evalStyles.bg} ${evalStyles.border} flex flex-col md:flex-row md:space-x-4`}>
                    <div className="md:w-2/5 mb-3 md:mb-0 flex-shrink-0">
                       <MiniBoard
                          key={miniBoardDynamicProps.keySuffix} // Add key to force re-render if snapshots change
                          tilesOnBoard={miniBoardDynamicProps.tilesOnBoard}
                          collectionSlot={miniBoardDynamicProps.collectionSlot}
                          playerClickedTileId={miniBoardDynamicProps.playerClickedTileId}
                          agentSuggestedTileId={miniBoardDynamicProps.agentSuggestedTileId}
                          collectionSlotCapacity={collectionSlotCapacity}
                       />
                       {isVisualizingThisMovesLine && activeLookaheadSequence && (
                         <div className="mt-2 text-center text-xs">
                            <p className="text-sky-300 mb-1">
                                Agent's Line: Step {currentLookaheadStepIndex + 1} of {activeLookaheadSequence.length}
                                {activeLookaheadSequence[currentLookaheadStepIndex].matchMade && <span className="ml-1 text-green-400">(Match!)</span>}
                            </p>
                            <div className="flex justify-center space-x-1">
                                <button onClick={() => handleLookaheadNav('prev')} disabled={currentLookaheadStepIndex === 0} className="px-2 py-0.5 bg-slate-600 hover:bg-slate-500 rounded text-xs disabled:opacity-50">Prev</button>
                                <button onClick={() => handleLookaheadNav('next')} disabled={currentLookaheadStepIndex === activeLookaheadSequence.length - 1} className="px-2 py-0.5 bg-slate-600 hover:bg-slate-500 rounded text-xs disabled:opacity-50">Next</button>
                                <button onClick={handleExitVisualization} className="px-2 py-0.5 bg-yellow-600 hover:bg-yellow-500 rounded text-xs text-slate-800">Player's Move</button>
                            </div>
                         </div>
                       )}
                    </div>
                    <div className="flex-grow">
                        <div className="flex flex-col sm:flex-row justify-between items-start mb-2">
                            <h3 className="text-md sm:text-lg font-semibold text-slate-100 order-2 sm:order-1">
                            Move {analyzedMove.moveNumber}: <span className="text-yellow-400">{analyzedMove.playerTileClickedSymbol}</span>
                            <span className="text-xs text-slate-400"> (ID: {analyzedMove.playerTileClickedId.substring(0,4)})</span>
                            </h3>
                            <span className={`px-2 py-0.5 sm:px-3 sm:py-1 text-xs sm:text-sm font-semibold rounded-full ${evalStyles.bg} ${evalStyles.text} border ${evalStyles.border} order-1 sm:order-2 mb-1 sm:mb-0`}>
                                {analyzedMove.evaluation}
                            </span>
                        </div>
                        
                        <p className={`text-xs sm:text-sm ${evalStyles.text} italic`}>{analyzedMove.agentReasoning}</p>
                        
                        {analyzedMove.missedOpportunity && (
                            <div className="mt-2 pt-2 border-t border-slate-600">
                                <div className="flex justify-between items-center">
                                    <div>
                                        <p className="text-xs sm:text-sm text-sky-400 font-semibold">Missed Opportunity:</p>
                                        <p className="text-xs sm:text-sm text-sky-300">
                                            Agent suggests <span className="text-md text-teal-300">{analyzedMove.missedOpportunity.tileSymbol}</span>
                                            <span className="text-xs text-slate-400"> (ID: {analyzedMove.missedOpportunity.tileId.substring(0,4)})</span>
                                            {analyzedMove.missedOpportunity.simulatedPolicyConfidence && 
                                                <span className="text-2xs text-sky-500"> (Conf: {(analyzedMove.missedOpportunity.simulatedPolicyConfidence * 100).toFixed(0)}%)</span>}
                                        </p>
                                    </div>
                                    {analyzedMove.missedOpportunity.predictedSequence && analyzedMove.missedOpportunity.predictedSequence.length > 0 && !isVisualizingThisMovesLine && (
                                        <button 
                                            onClick={() => handleVisualizeLine(analyzedMove)}
                                            className="px-2 py-1 bg-teal-600 hover:bg-teal-500 text-white text-xs rounded shadow"
                                        >
                                            Visualize Line
                                        </button>
                                    )}
                                </div>
                                <p className="text-2xs sm:text-xs text-slate-300 italic mt-0.5">Reason: {analyzedMove.missedOpportunity.reasoning}</p>
                            </div>
                        )}
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
          {!isLoading && !error && (!analysisResults || analysisResults.length === 0) && (
             <p className="text-slate-400 text-center py-4">No moves were analyzed or the analysis returned empty.</p>
          )}
        </div>
        <button
            onClick={onClose}
            className="mt-4 sm:mt-6 w-full px-6 py-2 bg-yellow-500 hover:bg-yellow-600 text-slate-900 font-semibold rounded-lg shadow-md transition-colors duration-150"
        >
            Close Review
        </button>
      </div>
    </div>
  );
};
