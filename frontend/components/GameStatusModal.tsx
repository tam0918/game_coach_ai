
import React from 'react';
import { GameState } from '../types';

interface GameStatusModalProps {
  gameState: Extract<GameState, 'won' | 'lost'>;
  onReset: () => void;
  onShowLocalAnalysis: () => void;
  onShowPostMatchReview: () => void; // New prop for showing post-match review
  onSaveMatch: () => void; // Prop for saving the match
  isMatchSaved: boolean;   // Prop to indicate if the match is already saved
}

export const GameStatusModal: React.FC<GameStatusModalProps> = ({ 
    gameState, 
    onReset, 
    onShowLocalAnalysis,
    onShowPostMatchReview,
    onSaveMatch,
    isMatchSaved
}) => {
  const messages = {
    won: { title: "Congratulations!", text: "You've cleared all tiles!", color: "text-green-400" },
    lost: { title: "Game Over!", text: "The collection slot is full.", color: "text-red-400" },
  };

  const message = messages[gameState];

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
      <div className="bg-slate-800 p-8 rounded-xl shadow-2xl text-center border border-slate-700 max-w-sm w-full">
        <h2 className={`text-4xl font-bold mb-4 ${message.color}`}>{message.title}</h2>
        <p className="text-slate-300 text-lg mb-6">{message.text}</p>
        <div className="space-y-3"> {/* Reduced space-y slightly */}
            <button
              onClick={onReset}
              className="w-full px-8 py-3 bg-yellow-500 hover:bg-yellow-600 text-slate-900 font-bold text-lg rounded-lg shadow-md transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-yellow-400 focus:ring-opacity-75"
            >
              Play Again
            </button>
            <button
              onClick={onShowLocalAnalysis}
              className={`w-full px-8 py-3 font-bold text-lg rounded-lg shadow-md transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-opacity-75
                 bg-purple-500 hover:bg-purple-600 text-white focus:ring-purple-400`}
            >
              Review with Local Agent
            </button>
            <button
              onClick={onShowPostMatchReview}
              className={`w-full px-8 py-3 font-bold text-lg rounded-lg shadow-md transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-opacity-75
                 bg-indigo-500 hover:bg-indigo-600 text-white focus:ring-indigo-400`}
            >
              Post-Match Review
            </button>
            <button
              onClick={onSaveMatch}
              disabled={isMatchSaved}
              className={`w-full px-8 py-3 font-bold text-lg rounded-lg shadow-md transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-opacity-75
                 ${isMatchSaved ? 'bg-green-700 text-green-200 cursor-default' : 'bg-sky-500 hover:bg-sky-600 text-white focus:ring-sky-400'}`}
            >
              {isMatchSaved ? 'Match Saved ✓' : 'Save Match to History'}
            </button>
        </div>
      </div>
    </div>
  );
};
