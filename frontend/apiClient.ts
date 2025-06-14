// API client for interacting with the backend services
import { PredictedMove } from './types';

const API_BASE_URL = 'http://localhost:5000';

// Interface for rationale prediction response
interface RationalePredictionResponse {
  score: number;
  normalized_score: number;
  category: string;
  rationale: string;
  error?: string;
}

// Interface for model health status
interface ModelHealthStatus {
  status: string;
  fpt_model_configured: boolean;
  local_torch_model_loaded: boolean;
  local_sklearn_model_loaded: boolean;
  rationale_model_loaded: boolean;
}

// API client functions
export const ApiClient = {
  /**
   * Check server health and available models
   */
  async checkHealth(): Promise<ModelHealthStatus> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        console.warn(`Health check: Server returned ${response.status}: ${response.statusText}`);
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }
      
      return response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      return {
        status: 'error',
        fpt_model_configured: false,
        local_torch_model_loaded: false,
        local_sklearn_model_loaded: false,
        rationale_model_loaded: false
      };
    }
  },
  
  /**
   * Get rationale prediction for a move or game summary
   * @param moveData - Game state and move data, or game summary data
   */
  async getRationalePrediction(moveData: any): Promise<RationalePredictionResponse> {
    try {
      console.log('Making rationale prediction request with data:', JSON.stringify(moveData, null, 2));
      
      // Special handling for game summary data (post-match review)
      if (moveData.result !== undefined) {
        console.log('Detected game summary data for post-match review');
        
        // For game summary, we'll just return a fixed response since
        // the backend isn't specially optimized for this use case yet
        const gameResult = moveData.result;
        const goodMovePercentage = moveData.goodMovePercentage || 0;
        const difficulty = moveData.difficulty || 'medium';
        
        // Generate appropriate rationale based on game data
        let rationale = '';
        let category = 'Strategic';
        let score = 5.0;
        
        if (gameResult === 'won') {
          if (goodMovePercentage > 70) {
            category = 'Brilliant';
            score = 9.0;
            rationale = `Excellent game! Your strategic decisions throughout the ${difficulty} difficulty level led to a solid victory.`;
          } else if (goodMovePercentage > 50) {
            category = 'Good';
            score = 7.5;
            rationale = `Good performance on ${difficulty} difficulty. Your consistently solid moves resulted in a well-earned win.`;
          } else {
            category = 'Strategic';
            score = 6.0;
            rationale = `You secured a victory on ${difficulty} difficulty with some smart plays, despite making a few suboptimal moves.`;
          }
        } else {
          // Loss
          if (goodMovePercentage > 50) {
            category = 'Strategic';
            score = 5.0;
            rationale = `Despite the loss on ${difficulty} difficulty, many of your moves showed good understanding of the game mechanics.`;
          } else {
            category = 'Inaccuracy';
            score = 3.5;
            rationale = `This ${difficulty} level game revealed some opportunities to improve your matching strategy and slot management.`;
          }
        }
        
        return {
          score: score,
          normalized_score: score,
          category: category,
          rationale: rationale
        };
      }
      
      // Standard move analysis call to backend
      const response = await fetch(`${API_BASE_URL}/predict_rationale`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(moveData)
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        try {
          const errorData = JSON.parse(errorText);
          throw new Error(errorData.error || `Server returned ${response.status}`);
        } catch (parseError) {
          throw new Error(`Server returned ${response.status}: ${errorText}`);
        }
      }
      
      return response.json();
    } catch (error) {
      console.error('Rationale prediction failed:', error);
      return {
        score: 5.0,
        normalized_score: 5.0,
        category: 'Average',
        rationale: 'Failed to get prediction from server.',
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  },
  
  /**
   * Get standard move prediction (existing endpoint)
   * @param moveData - Game state and move data
   */
  async getMovePrediction(moveData: any): Promise<any> {
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(moveData)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Server returned ${response.status}`);
      }
      
      return response.json();
    } catch (error) {
      console.error('Move prediction failed:', error);
      return {
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }
};

export default ApiClient;
