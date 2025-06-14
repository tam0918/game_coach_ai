from typing import List, Dict, Any
from collections import defaultdict
from .game_logic import GameState, Tile

# Constants, can be tuned
TILES_PER_MATCH_SET = 3

class HeuristicAgent:
    """
    A Python port of the rule-based agent from the frontend.
    It analyzes a game state and scores all possible moves based on a set of heuristics.
    """

    def get_unblocking_potential(self, tile_to_remove: Tile, game_state: GameState) -> int:
        """
        Calculates how many new tiles become accessible if a given tile is removed.
        A port of getUnblockingPotential.
        """
        # Create a hypothetical board state without the tile
        tiles_without_target = [t for t in game_state.board_tiles if t.id != tile_to_remove.id and not t.is_collected and not t.is_matched]
        
        # Create a hypothetical game state to use its methods, passing the required settings
        hypothetical_state = GameState(board_tiles=tiles_without_target, difficulty_settings=game_state.difficulty_settings)
        
        newly_accessible_count = 0
        
        # Find tiles that were previously blocked by the removed tile
        for candidate_tile in game_state.board_tiles:
            if candidate_tile.layer < tile_to_remove.layer and not game_state.is_tile_accessible(candidate_tile.id):
                # Check if it becomes accessible in the new state
                if hypothetical_state.is_tile_accessible(candidate_tile.id):
                    newly_accessible_count += 1
                    
        return newly_accessible_count

    def calculate_raw_agent_moves(self, game_state: GameState) -> List[Dict[str, Any]]:
        """
        Scores all accessible tiles based on a set of rules.
        A port of calculateRawAgentMoves.
        """
        accessible_tiles = game_state.get_accessible_tiles()
        collection_tiles = game_state.collection_tiles
        collection_capacity = game_state.collection_capacity
        
        raw_agent_moves = []

        slot_symbol_counts = defaultdict(list)
        for tile in collection_tiles:
            slot_symbol_counts[tile.symbol].append(tile.id)

        for acc_tile in accessible_tiles:
            score = 0
            desc = ""
            unblocks = self.get_unblocking_potential(acc_tile, game_state)
            
            symbol_count_in_slot = len(slot_symbol_counts.get(acc_tile.symbol, []))

            # Rule 1: Complete an immediate triple
            if symbol_count_in_slot == TILES_PER_MATCH_SET - 1:
                score = 100 + unblocks * 2 + acc_tile.layer
                desc = f"Completes an immediate triple for {acc_tile.symbol}"
            
            # Rule 2: Consolidate in a nearly full slot (risky but sometimes necessary)
            elif len(collection_tiles) >= collection_capacity - 2 and symbol_count_in_slot > 0:
                score = 80 + unblocks + acc_tile.layer
                desc = f"Consolidates existing {acc_tile.symbol} in a nearly full slot"
            
            # Rule 3: Form a pair
            elif symbol_count_in_slot == TILES_PER_MATCH_SET - 2 and len(collection_tiles) < collection_capacity - 1:
                third_tile_available = any(t.symbol == acc_tile.symbol and t.id != acc_tile.id for t in accessible_tiles)
                score = 70 + (10 if third_tile_available else 0) + unblocks + acc_tile.layer
                desc = f"Forms a pair for {acc_tile.symbol}"
            
            # Rule 4: Start a new pair from board tiles
            elif len(collection_tiles) < collection_capacity - 2:
                other_accessible_of_symbol = sum(1 for t in accessible_tiles if t.symbol == acc_tile.symbol and t.id != acc_tile.id)
                if other_accessible_of_symbol >= 1 and symbol_count_in_slot == 0:
                    score = 60 + acc_tile.layer + unblocks + other_accessible_of_symbol * 2
                    desc = f"Starts a new pair for {acc_tile.symbol} from board tiles"
                else:
                    # Generic strategic move
                    score = 40 + acc_tile.layer * 2 + unblocks * 3
                    desc = f"Is a strategic layer {acc_tile.layer} tile ({acc_tile.symbol})"
            
            # Fallback rule
            else:
                score = 30 + acc_tile.layer + unblocks * 2
                desc = f"Is a layer {acc_tile.layer} tile ({acc_tile.symbol})"

            if unblocks > 0:
                desc += f" (unblocks {unblocks})"
            
            raw_agent_moves.append({
                "tile": acc_tile,
                "score": score,
                "description": desc,
                "unblocks": unblocks
            })
            
        # Sort moves by score, descending
        raw_agent_moves.sort(key=lambda x: x['score'], reverse=True)
        return raw_agent_moves 