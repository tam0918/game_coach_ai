import React, { useState, useEffect, useMemo } from 'react';
import { LocalAgentMoveEvaluation, MoveEvaluationCategory, MoveRecord } from '../types';
import ApiClient from '../apiClient';

interface PostMatchReviewModalProps {
  moveHistory: MoveRecord[];
  analysisResults: LocalAgentMoveEvaluation[] | null;
  gameResult: 'won' | 'lost';
  onClose: () => void;
  difficulty: string;
}

interface GameMetrics {
  totalMoves: number;
  goodMoves: number;
  averageMoves: number;
  badMoves: number;
  playerRating: number;
  playerImprovementTips: string[];
  overallAssessment: string;
}

// New interface for detailed move analysis
interface DetailedMoveAnalysis {
  moveNumber: number;
  symbol: string;
  evaluation: MoveEvaluationCategory;
  reasoning: string;
  impact: string; // How this move affected the game
  hasMissedOpportunity: boolean;
  missedOpportunityDescription?: string;
  boardValueChange?: number; // Numeric change in board value
  moveContext?: string; // Additional context about game state at this point
  visualRecommendation?: string; // Visual description of what would have been better
}

export const PostMatchReviewModal: React.FC<PostMatchReviewModalProps> = ({
  moveHistory,
  analysisResults,
  gameResult,
  onClose,
  difficulty
}) => {
  // Generate a local rationale when API is unavailable
  const generateLocalRationale = (result: string, rating: number, difficultyLevel: string): string => {
    if (result === 'won') {
      if (rating >= 85) return `You showed excellent strategic thinking in this ${difficultyLevel} game, making consistently strong moves.`;
      if (rating >= 70) return `You played well throughout this ${difficultyLevel} game, with good understanding of the game mechanics.`;
      return `Your perseverance paid off in this ${difficultyLevel} game, though there's room to improve your tile selection strategy.`;
    } else {
      if (rating >= 70) return `Despite the loss, you made many good moves in this ${difficultyLevel} game - bad luck may have been a factor.`;
      if (rating >= 50) return `Your ${difficultyLevel} game showed potential but was undermined by a few critical mistakes in collection management.`;
      return `This ${difficultyLevel} game revealed some fundamental issues in your approach to tile selection and matching.`;
    }
  };

  const [metrics, setMetrics] = useState<GameMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [aiRationaleResponse, setAiRationaleResponse] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [detailedMoveAnalyses, setDetailedMoveAnalyses] = useState<DetailedMoveAnalysis[]>([]);
  const [selectedTab, setSelectedTab] = useState<'summary' | 'moves'>('summary');
  const [selectedMove, setSelectedMove] = useState<number | null>(null);
  
  // Debug logging when component mounts  // State for pattern insights
  const [patternInsights, setPatternInsights] = useState<string[]>([]);
  
  useEffect(() => {
    console.log('PostMatchReviewModal mounted with props:', {
      moveHistoryLength: moveHistory?.length,
      analysisResultsLength: analysisResults?.length,
      gameResult,
      difficulty
    });

    async function generatePostMatchAnalysis() {
      setIsLoading(true);
      setError(null);
      
      try {
        // If we don't have analysis results but we do have move history, try to generate simple metrics
        if (!analysisResults || analysisResults.length === 0) {
          console.log("No analysis results available, generating basic metrics from move history");
          
          if (moveHistory && moveHistory.length > 0) {
            const totalMoves = moveHistory.length;
            // Default to average moves when we don't have analysis
            const avgRating = gameResult === 'won' ? 70 : 40;
            
            // Generate a simple assessment
            const overallAssessment = gameResult === 'won' 
              ? `You completed the ${difficulty} game in ${totalMoves} moves.`
              : `You played ${totalMoves} moves in this ${difficulty} game before losing.`;
            
            // Simple improvement tip
            const improvementTips = [
              gameResult === 'won' 
                ? "Try to complete the game in fewer moves by planning ahead." 
                : "Focus on keeping your collection slots open and match tiles efficiently."
            ];
            
            setMetrics({
              totalMoves,
              goodMoves: Math.round(totalMoves * 0.4),  // Rough estimates
              averageMoves: Math.round(totalMoves * 0.4),
              badMoves: Math.round(totalMoves * 0.2),
              playerRating: avgRating,
              playerImprovementTips: improvementTips,
              overallAssessment
            });

            // Generate basic detailed move analyses
            const basicMoveAnalyses = moveHistory.map((move, index) => {
              return {
                moveNumber: move.moveNumber,
                symbol: move.tileClicked.symbol,
                evaluation: 'Average' as MoveEvaluationCategory,
                reasoning: `Move ${move.moveNumber}: Selected a ${move.tileClicked.symbol} from layer ${move.tileClicked.layer}.`,
                impact: move.matchMade ? 'Created a match' : 'Added to collection slot',
                hasMissedOpportunity: false
              };
            });
            
            setDetailedMoveAnalyses(basicMoveAnalyses);
          } else {
            setError("No move history available for analysis");
          }
          setIsLoading(false);
          return;
        }
        
        // Calculate basic metrics from analysis results
        if (analysisResults && analysisResults.length > 0) {
          console.log("Using existing analysis results to generate metrics");
          const totalMoves = analysisResults.length;
          const goodMoves = analysisResults.filter(result => result.evaluation === 'Good' || result.evaluation === 'Genius').length;
          const badMoves = analysisResults.filter(result => result.evaluation === 'Bad' || result.evaluation === 'Stupid').length;
          const averageMoves = totalMoves - goodMoves - badMoves;
          
          // Calculate player rating (0-100)
          const moveScores = {
            'Genius': 100,
            'Good': 80,
            'Average': 50,
            'Bad': 30,
            'Stupid': 10,
            'Info': 50
          };
          
          const totalScore = analysisResults.reduce((sum, result) => {
            // TypeScript needs this cast to safely access moveScores with the evaluation
            const evalCategory = result.evaluation as keyof typeof moveScores;
            return sum + moveScores[evalCategory];
          }, 0);
          
          const playerRating = Math.round(totalScore / totalMoves);
          
          // Generate improvement tips based on common issues
          const improvementTips = generateImprovementTips(analysisResults, gameResult);
          
          // Generate overall assessment
          const overallAssessment = generateOverallAssessment(playerRating, gameResult, difficulty);
          
          // Set metrics for summary view
          setMetrics({
            totalMoves,
            goodMoves,
            averageMoves,
            badMoves,
            playerRating,
            playerImprovementTips: improvementTips,
            overallAssessment
          });          // Generate detailed move analyses with enhanced context and visualization cues
          const detailedAnalyses = analysisResults.map((result, index) => {
            // Get the corresponding move record for additional context
            const moveRecord = moveHistory[index];
            
            // Track game progress percentage
            const progressPercent = Math.round((index / analysisResults.length) * 100);
            const gameStage = progressPercent < 30 ? 'early game' : 
                              progressPercent < 70 ? 'mid game' : 'late game';
            
            // Get collection slot fullness for context
            const collectionSlotSize = moveRecord.collectionSlotAfterMove ? moveRecord.collectionSlotAfterMove.length : 0;
            const collectionContext = collectionSlotSize === 0 ? "empty collection" :
                                    collectionSlotSize <= 1 ? "nearly empty collection" :
                                    collectionSlotSize <= 2 ? "partially filled collection" :
                                    "nearly full collection";
            
            // Create move context description
            const moveContext = `${gameStage} | ${collectionContext} | ${moveRecord.boardStateBeforeMove.totalTilesOnBoard} tiles remaining`;
            
            // Calculate board value change for visualization
            const boardValueChange = index > 0 && result.simulatedBoardValueAfterPlayerMove && analysisResults[index-1].simulatedBoardValueAfterPlayerMove ?
              result.simulatedBoardValueAfterPlayerMove - analysisResults[index-1].simulatedBoardValueAfterPlayerMove : 0;
            
            // Enhanced impact analysis
            let impact = "No significant effect";
            
            if (moveRecord.matchMade) {
              impact = "Created a match, clearing space in the collection slot";
              if (collectionSlotSize >= 2) {
                impact = "Created a critical match, preventing potential collection overflow";
              }
            } else if (index > 0 && index < analysisResults.length - 1) {
              // Compare board state before and after
              const nextAnalysis = analysisResults[index + 1];
              
              if (nextAnalysis && nextAnalysis.simulatedBoardValueAfterPlayerMove > result.simulatedBoardValueAfterPlayerMove) {
                impact = "Set up a future match opportunity";
                if (boardValueChange > 2) {
                  impact = "Significantly improved board state, creating multiple future opportunities";
                }
              } else if (nextAnalysis && nextAnalysis.simulatedBoardValueAfterPlayerMove < result.simulatedBoardValueAfterPlayerMove) {
                impact = "Potentially limited future options";
                if (boardValueChange < -2) {
                  impact = "Substantially restricted future options, putting you at a disadvantage";
                }
              }
            } else if (index === analysisResults.length - 1) {
              // Last move
              impact = gameResult === 'won' ? 
                "The final move that sealed your victory" : 
                "The final move before the game ended in defeat";
            }
            
            // Visual recommendation for missed opportunities
            let visualRecommendation = undefined;
            if (result.missedOpportunity) {
              visualRecommendation = result.missedOpportunity.tileSymbol === moveRecord.tileClicked.symbol ?
                `Better to choose a different ${result.missedOpportunity.tileSymbol} tile (one that would unblock more tiles)` :
                `Taking a ${result.missedOpportunity.tileSymbol} tile would have been better than the ${moveRecord.tileClicked.symbol} you selected`;
            }
            
            return {
              moveNumber: result.moveNumber,
              symbol: result.playerTileClickedSymbol,
              evaluation: result.evaluation,
              reasoning: result.agentReasoning,
              impact,
              hasMissedOpportunity: !!result.missedOpportunity,
              missedOpportunityDescription: result.missedOpportunity ? 
                `Better option: ${result.missedOpportunity.tileSymbol} (${result.missedOpportunity.reasoning})` : 
                undefined,
              boardValueChange: boardValueChange,
              moveContext: moveContext,
              visualRecommendation: visualRecommendation
            };
          });
            setDetailedMoveAnalyses(detailedAnalyses);
          
          // Generate pattern insights from the detailed analyses
          const insights = identifyMovePatterns(detailedAnalyses);
          setPatternInsights(insights);
          
          // Try to get rationale from backend
          try {
            // Package game data for AI analysis
            const gameData = {
              difficulty: difficulty,
              result: gameResult,
              totalMoves: totalMoves,
              goodMovePercentage: Math.round((goodMoves / totalMoves) * 100),
              badMovePercentage: Math.round((badMoves / totalMoves) * 100)
            };
            
            console.log("Requesting AI rationale with data:", gameData);
            const rationaleResponse = await ApiClient.getRationalePrediction(gameData);
            console.log("Received rationale response:", rationaleResponse);
            
            if (rationaleResponse && !rationaleResponse.error) {
              setAiRationaleResponse(rationaleResponse.rationale);
            } else {
              // Create a local fallback rationale
              setAiRationaleResponse(generateLocalRationale(gameResult, playerRating, difficulty));
            }
          } catch (error) {
            console.error("Failed to get AI rationale for game review:", error);
            // In case of any error, generate a local rationale
            setAiRationaleResponse(generateLocalRationale(gameResult, playerRating, difficulty));
          }
        }
      } catch (error) {
        console.error("Error generating post-match analysis:", error);
        setError(`Error generating analysis: ${error instanceof Error ? error.message : String(error)}`);
      } finally {
        setIsLoading(false);
      }
    }
    
    generatePostMatchAnalysis();
  }, [analysisResults, gameResult, difficulty, moveHistory]);
    const generateImprovementTips = (analysisResults: LocalAgentMoveEvaluation[], gameResult: string): string[] => {
    const tips: string[] = [];
    
    // Count missed opportunities
    const missedOpportunities = analysisResults.filter(result => result.missedOpportunity).length;
    if (missedOpportunities > 0) {
      tips.push(`Look for better moves - you missed ${missedOpportunities} key opportunities.`);
    }
    
    // Check for pattern of bad moves toward the end (game management)
    const lastFiveMoves = analysisResults.slice(-5);
    const badMovesAtEnd = lastFiveMoves.filter(result => 
      result.evaluation === 'Bad' || result.evaluation === 'Stupid'
    ).length;
    
    if (badMovesAtEnd >= 2 && gameResult === 'lost') {
      tips.push("Pay more attention to your endgame - several mistakes in the final moves cost you the game.");
    }
    
    // Check if player tends to make more mistakes as the game progresses
    const firstHalf = analysisResults.slice(0, Math.floor(analysisResults.length / 2));
    const secondHalf = analysisResults.slice(Math.floor(analysisResults.length / 2));
    
    const badMovesFirstHalf = firstHalf.filter(result => 
      result.evaluation === 'Bad' || result.evaluation === 'Stupid'
    ).length;
    
    const badMovesSecondHalf = secondHalf.filter(result => 
      result.evaluation === 'Bad' || result.evaluation === 'Stupid'
    ).length;
    
    if (badMovesSecondHalf > badMovesFirstHalf * 1.5) {
      tips.push("Your performance declines as games progress. Try to maintain focus throughout the game.");
    }
    
    // Look for preventable collection slot overflow
    const collectionOverflowMistakes = analysisResults.filter(result => 
      result.agentReasoning && result.agentReasoning.toLowerCase().includes('collection') && 
      result.agentReasoning.toLowerCase().includes('overflow') && 
      (result.evaluation === 'Bad' || result.evaluation === 'Stupid')
    ).length;
    
    if (collectionOverflowMistakes > 0) {
      tips.push(`Watch your collection slot usage - you made ${collectionOverflowMistakes} moves that risked or caused collection overflow.`);
    }
    
    // Check for symbol stacking
    const symbolCounts: {[key: string]: number} = {};
    analysisResults.forEach(result => {
      const symbol = result.playerTileClickedSymbol;
      symbolCounts[symbol] = (symbolCounts[symbol] || 0) + 1;
    });
    
    const mostPickedSymbol = Object.keys(symbolCounts).sort((a, b) => symbolCounts[b] - symbolCounts[a])[0];
    const mostPickedCount = symbolCounts[mostPickedSymbol];
    
    if (mostPickedCount > analysisResults.length / 3) {
      tips.push(`You selected the ${mostPickedSymbol} symbol ${mostPickedCount} times, which may indicate over-reliance on certain tiles. Try to keep your options diverse.`);
    }
    
    // Add general tips
    if (tips.length === 0) {
      if (gameResult === 'won') {
        tips.push("Keep maintaining your consistent performance throughout the game.");
      } else {
        tips.push("Try to plan a few moves ahead to avoid collection slot overflow.");
      }
    }
    
    return tips;
  };
  
  const generateOverallAssessment = (rating: number, gameResult: string, difficulty: string): string => {
    if (gameResult === 'won') {
      if (rating >= 85) return `Excellent performance! You mastered the ${difficulty} difficulty with strategic brilliance.`;
      if (rating >= 70) return `Good job! You successfully completed the ${difficulty} difficulty game with solid moves.`;
      return `You won the ${difficulty} game, but there's room for improvement in your strategy.`;
    } else {
      if (rating >= 70) return `Despite losing this ${difficulty} game, your moves were mostly strong. Bad luck may have played a role.`;
      if (rating >= 50) return `Your strategy in this ${difficulty} game had good moments, but some critical mistakes led to the loss.`;
      return `This ${difficulty} game revealed several strategic weaknesses that need attention to improve your results.`;
    }
  };
  
  const getRatingColor = (rating: number): string => {
    if (rating >= 85) return 'text-purple-400';
    if (rating >= 70) return 'text-green-400';
    if (rating >= 50) return 'text-yellow-400';
    if (rating >= 30) return 'text-orange-400';
    return 'text-red-400';
  };

  const getMoveEvaluationColor = (evaluation: MoveEvaluationCategory): string => {
    switch(evaluation) {
      case 'Genius': return 'text-purple-400';
      case 'Good': return 'text-green-400';
      case 'Average': return 'text-yellow-400';
      case 'Bad': return 'text-orange-400';
      case 'Stupid': return 'text-red-400';
      default: return 'text-slate-300';
    }
  };

  // Function to identify patterns in the player's moves for deeper insights
  const identifyMovePatterns = (moves: DetailedMoveAnalysis[]): string[] => {
    if (!moves || moves.length === 0) return [];
    
    const insights: string[] = [];
    
    // Check for consecutive mistakes
    let maxConsecutivePoorMoves = 0;
    let currentStreak = 0;
    
    for (let i = 0; i < moves.length; i++) {
      if (moves[i].evaluation === 'Bad' || moves[i].evaluation === 'Stupid') {
        currentStreak++;
        maxConsecutivePoorMoves = Math.max(maxConsecutivePoorMoves, currentStreak);
      } else {
        currentStreak = 0;
      }
    }
    
    if (maxConsecutivePoorMoves >= 3) {
      insights.push(`You had a streak of ${maxConsecutivePoorMoves} consecutive poor moves. Focus on maintaining consistency even when the board situation becomes challenging.`);
    }
    
    // Check if player tends to do better at certain stages of the game
    const earlyMoves = moves.slice(0, Math.floor(moves.length / 3));
    const midMoves = moves.slice(Math.floor(moves.length / 3), Math.floor(2 * moves.length / 3));
    const lateMoves = moves.slice(Math.floor(2 * moves.length / 3));
    
    const getAverageQuality = (moveset: DetailedMoveAnalysis[]) => {
      const scoreMap = {
        'Genius': 4,
        'Good': 3,
        'Average': 2,
        'Bad': 1,
        'Stupid': 0,
        'Info': 2
      };
      
      return moveset.reduce((sum, move) => {
        return sum + scoreMap[move.evaluation as keyof typeof scoreMap];
      }, 0) / moveset.length;
    };
    
    const earlyQuality = getAverageQuality(earlyMoves);
    const midQuality = getAverageQuality(midMoves);
    const lateQuality = getAverageQuality(lateMoves);
    
    const stageScores = [
      { stage: 'early game', score: earlyQuality },
      { stage: 'mid game', score: midQuality },
      { stage: 'late game', score: lateQuality }
    ];
    
    stageScores.sort((a, b) => b.score - a.score);
    
    insights.push(`Your strongest performance was in the ${stageScores[0].stage}, while you struggled most in the ${stageScores[2].stage}.`);
    
    // Check for missed matches
    const missedMatches = moves.filter(move => 
      move.hasMissedOpportunity && 
      move.missedOpportunityDescription?.includes('match')
    ).length;
    
    if (missedMatches > 0) {
      insights.push(`You missed ${missedMatches} potential matching opportunities. Try to scan the board for immediate matches before selecting tiles.`);
    }
    
    // Count matches created
    const matchesMade = moves.filter(move => 
      move.impact.toLowerCase().includes('match')
    ).length;
    
    if (matchesMade > 0 && moves.length > 0) {
      const matchRate = (matchesMade / moves.length * 100).toFixed(1);
      insights.push(`Your match creation rate was ${matchRate}% (${matchesMade} matches in ${moves.length} moves).`);
    }
    
    return insights;
  };
  
  // Component to visualize move quality over the course of the game
  const MoveTimelineChart: React.FC<{moves: DetailedMoveAnalysis[], onSelectMove: (index: number) => void}> = ({ 
    moves, 
    onSelectMove 
  }) => {
    // Convert evaluation categories to numeric values for visualization
    const getEvalScore = (evaluation: MoveEvaluationCategory): number => {
      switch(evaluation) {
        case 'Genius': return 10;
        case 'Good': return 7.5;
        case 'Average': return 5;
        case 'Bad': return 2.5;
        case 'Stupid': return 0;
        default: return 5;
      }
    };
    
    // Map moves to data points
    const dataPoints = useMemo(() => {
      return moves.map((move, index) => ({
        index,
        score: getEvalScore(move.evaluation),
        evaluation: move.evaluation,
        moveNumber: move.moveNumber,
        symbol: move.symbol
      }));
    }, [moves]);
    
    // Calculate overall trend
    const calculateTrend = () => {
      if (dataPoints.length < 3) return "steady";
      
      const firstHalf = dataPoints.slice(0, Math.floor(dataPoints.length / 2));
      const secondHalf = dataPoints.slice(Math.floor(dataPoints.length / 2));
      
      const firstHalfAvg = firstHalf.reduce((sum, point) => sum + point.score, 0) / firstHalf.length;
      const secondHalfAvg = secondHalf.reduce((sum, point) => sum + point.score, 0) / secondHalf.length;
      
      const difference = secondHalfAvg - firstHalfAvg;
      if (difference > 0.5) return "improving";
      if (difference < -0.5) return "declining";
      return "steady";
    };
    
    const trend = useMemo(() => calculateTrend(), [dataPoints]);
    
    const chartHeight = 80;
    const chartWidth = "100%";
    const pointRadius = 4;
    
    return (
      <div className="mt-4 bg-slate-800/40 rounded-lg p-3 relative">
        <h4 className="text-sm font-semibold text-slate-300 mb-1">Move Quality Timeline</h4>
        <div className="text-xs text-slate-400 mb-3">
          {trend === "improving" && "Your move quality improved as the game progressed"}
          {trend === "declining" && "Your move quality declined over the course of the game"}
          {trend === "steady" && "Your move quality remained consistent throughout the game"}
        </div>
        
        <div className="w-full h-24 relative">
          {/* Background grid lines */}
          <div className="absolute inset-0 flex flex-col justify-between">
            <div className="border-t border-slate-700/50 h-0" style={{top: "20%"}}></div>
            <div className="border-t border-slate-700/50 h-0" style={{top: "40%"}}></div>
            <div className="border-t border-slate-700/50 h-0" style={{top: "60%"}}></div>
            <div className="border-t border-slate-700/50 h-0" style={{top: "80%"}}></div>
          </div>
          
          {/* Y-axis labels */}
          <div className="absolute left-0 inset-y-0 flex flex-col justify-between text-xs text-slate-500 pr-1">
            <span>Great</span>
            <span>Good</span>
            <span>Avg</span>
            <span>Poor</span>
            <span>Bad</span>
          </div>
          
          {/* The chart itself */}
          <div className="absolute inset-0 pl-8">
            <svg width={chartWidth} height={chartHeight} className="overflow-visible">
              {/* Connect the dots with lines */}
              <polyline
                points={dataPoints.map((point, i) => {
                  const x = (i / (dataPoints.length - 1 || 1)) * 100 + '%';
                  const y = chartHeight - (point.score / 10) * chartHeight;
                  return `${x},${y}`;
                }).join(' ')}
                fill="none"
                stroke="#a855f7"
                strokeWidth="2"
                strokeLinejoin="round"
              />
              
              {/* Plot points */}
              {dataPoints.map((point, i) => {
                const x = (i / (dataPoints.length - 1 || 1)) * 100 + '%';
                const y = chartHeight - (point.score / 10) * chartHeight;
                
                // Determine color based on evaluation
                let color = "#a855f7"; // Default purple
                if (point.evaluation === "Genius") color = "#a855f7"; // Purple
                if (point.evaluation === "Good") color = "#4ade80"; // Green
                if (point.evaluation === "Average") color = "#facc15"; // Yellow
                if (point.evaluation === "Bad") color = "#fb923c"; // Orange
                if (point.evaluation === "Stupid") color = "#ef4444"; // Red
                
                return (
                  <g key={i} onClick={() => onSelectMove(i)} style={{cursor: "pointer"}}>
                    <circle 
                      cx={x} 
                      cy={y} 
                      r={pointRadius}
                      fill={color}
                      stroke="#1e293b"
                      strokeWidth="1"
                    />
                    <text 
                      x={x} 
                      y={y - 8} 
                      textAnchor="middle" 
                      fontSize="9" 
                      fill={color}
                    >
                      {point.moveNumber}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        </div>
        
        <div className="text-xs text-slate-400 mt-1 text-center">
          Click on any point to see move details
        </div>
      </div>
    );
  };
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4 overflow-auto">
      <div className="bg-slate-800 rounded-xl shadow-2xl border border-slate-700 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        <div className="px-6 py-4 border-b border-slate-700 flex justify-between items-center sticky top-0 bg-slate-800 z-10">
          <h2 className="text-2xl font-bold text-white">Post-Match Review</h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white focus:outline-none"
          >
            <span className="text-2xl">&times;</span>
          </button>
        </div>
        
        <div className="border-b border-slate-700">
          <div className="flex">
            <button
              onClick={() => setSelectedTab('summary')}
              className={`px-4 py-3 font-semibold ${selectedTab === 'summary' 
                ? 'text-purple-400 border-b-2 border-purple-400' 
                : 'text-slate-400 hover:text-white'}`}
            >
              Game Summary
            </button>
            <button
              onClick={() => setSelectedTab('moves')}
              className={`px-4 py-3 font-semibold ${selectedTab === 'moves' 
                ? 'text-purple-400 border-b-2 border-purple-400' 
                : 'text-slate-400 hover:text-white'}`}
            >
              Move Analysis
            </button>
          </div>
        </div>
        
        <div className="p-6">
          {isLoading ? (
            <div className="text-center py-8">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-purple-500"></div>
              <p className="mt-2 text-slate-300">Analyzing your game...</p>
            </div>
          ) : error ? (
            <div className="text-center py-8 text-red-400">
              <p>{error}</p>
              <button
                onClick={onClose}
                className="mt-4 px-6 py-2 bg-slate-600 hover:bg-slate-500 text-white font-semibold rounded-lg transition-colors duration-150"
              >
                Close
              </button>
            </div>
          ) : metrics ? (
            selectedTab === 'summary' ? (
              <div className="space-y-6">
                <div className="text-center">
                  <h3 className="text-xl font-semibold text-slate-300 mb-2">Game Summary</h3>
                  <div className="inline-block rounded-lg px-6 py-4 bg-slate-700">
                    <div className="text-4xl font-bold mb-1 flex justify-center items-center">
                      <span className={getRatingColor(metrics.playerRating)}>
                        {metrics.playerRating}
                      </span>
                      <span className="text-slate-400 text-lg ml-2">/ 100</span>
                    </div>
                    <div className="text-slate-300">{gameResult === 'won' ? 'Victory' : 'Defeat'} on {difficulty}</div>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-green-900/30 rounded-lg p-4 text-center">
                    <div className="text-3xl font-bold text-green-400 mb-1">{metrics.goodMoves}</div>
                    <div className="text-slate-300 text-sm">Good Moves</div>
                  </div>
                  <div className="bg-yellow-900/30 rounded-lg p-4 text-center">
                    <div className="text-3xl font-bold text-yellow-400 mb-1">{metrics.averageMoves}</div>
                    <div className="text-slate-300 text-sm">Average Moves</div>
                  </div>
                  <div className="bg-red-900/30 rounded-lg p-4 text-center">
                    <div className="text-3xl font-bold text-red-400 mb-1">{metrics.badMoves}</div>
                    <div className="text-slate-300 text-sm">Poor Moves</div>
                  </div>
                </div>
                
                <div className="bg-slate-700/50 rounded-lg p-4">
                  <h3 className="text-xl font-semibold text-slate-300 mb-3">Overall Assessment</h3>
                  <p className="text-slate-300">{metrics.overallAssessment}</p>
                  
                  {aiRationaleResponse && (
                    <div className="mt-3 text-sky-300 italic">
                      "{aiRationaleResponse}"
                    </div>
                  )}
                </div>
                  <div className="bg-slate-700/50 rounded-lg p-4">
                  <h3 className="text-xl font-semibold text-slate-300 mb-3">Game Analysis</h3>
                  
                  {/* Add move timeline chart */}
                  {detailedMoveAnalyses.length > 0 && (
                    <MoveTimelineChart 
                      moves={detailedMoveAnalyses} 
                      onSelectMove={(index) => {
                        setSelectedTab('moves');
                        setSelectedMove(index);
                      }} 
                    />
                  )}
                    {patternInsights.length > 0 && (
                    <>
                      <h4 className="text-lg font-semibold text-slate-300 mt-5 mb-2">Pattern Analysis</h4>
                      <div className="bg-indigo-900/30 border border-indigo-800/40 rounded p-3 mb-4">
                        <ul className="space-y-2 text-indigo-200">
                          {patternInsights.map((insight, idx) => (
                            <li key={idx} className="flex">
                              <span className="text-indigo-300 mr-2">•</span>
                              <span>{insight}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </>
                  )}
                  
                  <h4 className="text-lg font-semibold text-slate-300 mt-5 mb-2">Areas for Improvement</h4>
                  <ul className="list-disc pl-5 text-slate-300 space-y-1">
                    {metrics.playerImprovementTips.map((tip, index) => (
                      <li key={index}>{tip}</li>
                    ))}
                  </ul>
                </div>
              </div>
            ) : (
              <div>
                <h3 className="text-xl font-semibold text-slate-300 mb-4">Move-by-Move Analysis</h3>
                
                {detailedMoveAnalyses.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">                    {/* List of all moves */}
                    <div className="bg-slate-700/50 rounded-lg p-2 overflow-y-auto max-h-[70vh]">
                      <div className="text-sm text-slate-300 mb-3 p-2 border-b border-slate-600">
                        <h4 className="font-semibold text-center">Move Timeline</h4>
                        <p className="text-xs text-slate-400 text-center mt-1">Select a move to view detailed analysis</p>
                      </div>
                      <div className="space-y-1">
                        {detailedMoveAnalyses.map((moveAnalysis, index) => {
                          // Determine if the move is part of a pattern or a critical turning point
                          const isMatchMove = moveAnalysis.impact && moveAnalysis.impact.toLowerCase().includes('match');
                          const isBadMove = moveAnalysis.evaluation === 'Bad' || moveAnalysis.evaluation === 'Stupid';
                          const isGoodMove = moveAnalysis.evaluation === 'Good' || moveAnalysis.evaluation === 'Genius';
                          const hasMissed = moveAnalysis.hasMissedOpportunity;
                          
                          return (
                            <button
                              key={index}
                              onClick={() => setSelectedMove(index)}
                              className={`w-full p-2 text-left rounded relative 
                                ${selectedMove === index ? 'bg-slate-600 shadow-lg' : 'bg-slate-800'} 
                                hover:bg-slate-600 transition-colors duration-150
                                ${isBadMove ? 'border-l-4 border-orange-500' : ''}
                                ${isGoodMove ? 'border-l-4 border-green-500' : ''}
                              `}
                            >
                              <div className="flex justify-between items-center">
                                <div className="flex items-center">
                                  <span className="text-slate-300 font-medium">{moveAnalysis.moveNumber}</span>
                                  <span className="text-2xl mx-2">{moveAnalysis.symbol}</span>
                                  <div className="text-xs">
                                    {isMatchMove && <span className="inline-block px-1.5 py-0.5 bg-green-900/50 text-green-400 rounded mr-1">Match</span>}
                                    {hasMissed && <span className="inline-block px-1.5 py-0.5 bg-orange-900/50 text-orange-400 rounded">Missed</span>}
                                  </div>
                                </div>
                                <span className={`${getMoveEvaluationColor(moveAnalysis.evaluation)} text-sm font-medium`}>
                                  {moveAnalysis.evaluation}
                                </span>
                              </div>
                              {moveAnalysis.moveContext && (
                                <div className="mt-1 text-xs text-slate-400 truncate">
                                  {moveAnalysis.moveContext}
                                </div>
                              )}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                      {/* Selected move details */}
                    <div className="md:col-span-2 bg-slate-700/50 rounded-lg p-4">
                      {selectedMove !== null && detailedMoveAnalyses[selectedMove] ? (
                        <div>
                          <div className="flex justify-between items-center border-b border-slate-600 pb-3 mb-4">
                            <h4 className="text-lg font-semibold">
                              <span>Move {detailedMoveAnalyses[selectedMove].moveNumber}: {detailedMoveAnalyses[selectedMove].symbol}</span>
                            </h4>
                            <div className="flex items-center">
                              <div className="px-2 py-1 rounded-lg bg-slate-800">
                                <span className="text-slate-400 text-xs font-medium mr-2">Rating:</span>
                                <span className={`${getMoveEvaluationColor(detailedMoveAnalyses[selectedMove].evaluation)} font-bold`}>
                                  {detailedMoveAnalyses[selectedMove].evaluation}
                                </span>
                              </div>
                            </div>
                          </div>
                          
                          {detailedMoveAnalyses[selectedMove].moveContext && (
                            <div className="bg-slate-800/50 rounded mb-4 p-2 text-xs text-slate-400 font-medium">
                              {detailedMoveAnalyses[selectedMove].moveContext}
                            </div>
                          )}
                          
                          <div className="mt-4 text-slate-300">
                            <h5 className="font-semibold text-purple-300 mb-1">Agent Analysis</h5>
                            <div className="bg-slate-800/30 rounded p-3">
                              <p>{detailedMoveAnalyses[selectedMove].reasoning}</p>
                            </div>
                          </div>
                          
                          <div className="mt-4 text-slate-300">
                            <h5 className="font-semibold text-purple-300 mb-1">Impact on Game</h5>
                            <div className="bg-slate-800/30 rounded p-3 flex items-start">
                              {detailedMoveAnalyses[selectedMove].boardValueChange && 
                                detailedMoveAnalyses[selectedMove].boardValueChange !== 0 && (
                                <div className={`mr-2 font-bold ${
                                  detailedMoveAnalyses[selectedMove].boardValueChange > 0 
                                    ? 'text-green-400' 
                                    : 'text-red-400'
                                }`}>
                                  {detailedMoveAnalyses[selectedMove].boardValueChange > 0 ? '+' : ''}
                                  {detailedMoveAnalyses[selectedMove].boardValueChange.toFixed(1)}
                                </div>
                              )}
                              <p>{detailedMoveAnalyses[selectedMove].impact}</p>
                            </div>
                          </div>
                          
                          {detailedMoveAnalyses[selectedMove].hasMissedOpportunity && (
                            <div className="mt-4">
                              <h5 className="font-semibold text-orange-300 mb-1">Missed Opportunity</h5>
                              <div className="bg-orange-900/20 border border-orange-800/40 rounded p-3 text-orange-200">
                                <p>{detailedMoveAnalyses[selectedMove].missedOpportunityDescription}</p>
                                {detailedMoveAnalyses[selectedMove].visualRecommendation && (
                                  <p className="mt-2 text-sm italic text-orange-300">
                                    {detailedMoveAnalyses[selectedMove].visualRecommendation}
                                  </p>
                                )}
                              </div>
                            </div>
                          )}
                          
                          <div className="mt-6 pt-3 border-t border-slate-600 flex justify-between">
                            <button 
                              onClick={() => setSelectedMove(selectedMove - 1)}
                              className={`px-3 py-1 rounded transition-colors duration-150 ${
                                selectedMove > 0 
                                  ? 'bg-slate-600 hover:bg-slate-500 text-white' 
                                  : 'bg-slate-700 text-slate-500 cursor-not-allowed'
                              }`}
                              disabled={selectedMove <= 0}
                            >
                              ← Previous Move
                            </button>
                            <button 
                              onClick={() => setSelectedMove(selectedMove + 1)}
                              className={`px-3 py-1 rounded transition-colors duration-150 ${
                                selectedMove < detailedMoveAnalyses.length - 1 
                                  ? 'bg-slate-600 hover:bg-slate-500 text-white' 
                                  : 'bg-slate-700 text-slate-500 cursor-not-allowed'
                              }`}
                              disabled={selectedMove >= detailedMoveAnalyses.length - 1}
                            >
                              Next Move →
                            </button>
                          </div>
                        </div>
                      ) : (
                        <div className="text-center py-8 text-slate-400">
                          Select a move from the list to see detailed analysis
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-slate-400">
                    No move analysis data available
                  </div>
                )}
              </div>
            )
          ) : (
            <div className="text-center py-8 text-slate-300">
              <p>No analysis data available.</p>
              <p className="text-sm text-slate-400 mt-2">Try playing another game or recalibrating the agent.</p>
              <button
                onClick={onClose}
                className="mt-4 px-6 py-2 bg-slate-600 hover:bg-slate-500 text-white font-semibold rounded-lg transition-colors duration-150"
              >
                Close
              </button>
            </div>
          )}
        </div>
        
        <div className="px-6 py-4 border-t border-slate-700 flex justify-center">
          <button
            onClick={onClose}
            className="px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white font-semibold rounded-lg transition-colors duration-150"
          >
            Close Review
          </button>        </div>
      </div>
    </div>
  );
};
