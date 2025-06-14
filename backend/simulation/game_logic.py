from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import random
import uuid

# --- Constants Ported from frontend/constants.ts ---

TILE_WIDTH = 64
TILE_HEIGHT = 80
SYMBOLS = ['🍓', '🍌', '🍉', '🍇', '🍊', '🍍', '🥝', '🍒', '🍑', '🥭', '🥥', '🥑', '🍋', '🍐', '🍅', '🍆']

@dataclass
class LayerConfig:
    tileCount: int
    cols: int
    rows: int
    xOffset: float
    yOffset: float
    layerZ: int

@dataclass
class DifficultySettings:
    NAME: str
    NUM_UNIQUE_SYMBOLS_TO_USE: int
    NUM_SETS_PER_SYMBOL: int
    TILES_PER_MATCH_SET: int
    COLLECTION_SLOT_CAPACITY: int
    LAYERS_CONFIG: List[LayerConfig]
    TOTAL_TILES_TO_GENERATE: int

DIFFICULTY_LEVELS: Dict[str, DifficultySettings] = {
    'easy': DifficultySettings(
        NAME="Easy", NUM_UNIQUE_SYMBOLS_TO_USE=6, NUM_SETS_PER_SYMBOL=2, TILES_PER_MATCH_SET=3, COLLECTION_SLOT_CAPACITY=7,
        LAYERS_CONFIG=[
            LayerConfig(tileCount=20, cols=5, rows=4, xOffset=TILE_WIDTH/3, yOffset=TILE_HEIGHT/3, layerZ=0),
            LayerConfig(tileCount=16, cols=4, rows=4, xOffset=TILE_WIDTH/1.5, yOffset=TILE_HEIGHT/1.5, layerZ=1),
        ], TOTAL_TILES_TO_GENERATE=36),
    'medium': DifficultySettings(
        NAME="Medium", NUM_UNIQUE_SYMBOLS_TO_USE=8, NUM_SETS_PER_SYMBOL=2, TILES_PER_MATCH_SET=3, COLLECTION_SLOT_CAPACITY=7,
        LAYERS_CONFIG=[
            LayerConfig(tileCount=24, cols=6, rows=4, xOffset=0, yOffset=0, layerZ=0),
            LayerConfig(tileCount=15, cols=5, rows=3, xOffset=TILE_WIDTH/2.5, yOffset=TILE_HEIGHT/2.5, layerZ=1),
            LayerConfig(tileCount=9, cols=3, rows=3, xOffset=TILE_WIDTH/1.5, yOffset=TILE_HEIGHT/1.5, layerZ=2),
        ], TOTAL_TILES_TO_GENERATE=48),
    'hard': DifficultySettings(
        NAME="Hard", NUM_UNIQUE_SYMBOLS_TO_USE=10, NUM_SETS_PER_SYMBOL=2, TILES_PER_MATCH_SET=3, COLLECTION_SLOT_CAPACITY=7,
        LAYERS_CONFIG=[
            LayerConfig(tileCount=25, cols=5, rows=5, xOffset=0, yOffset=0, layerZ=0),
            LayerConfig(tileCount=20, cols=5, rows=4, xOffset=TILE_WIDTH/3, yOffset=TILE_HEIGHT/3, layerZ=1),
            LayerConfig(tileCount=15, cols=5, rows=3, xOffset=TILE_WIDTH/1.5, yOffset=TILE_HEIGHT/1.5, layerZ=2),
        ], TOTAL_TILES_TO_GENERATE=60),
    'hell': DifficultySettings(
        NAME="Hell", NUM_UNIQUE_SYMBOLS_TO_USE=12, NUM_SETS_PER_SYMBOL=2, TILES_PER_MATCH_SET=3, COLLECTION_SLOT_CAPACITY=7,
        LAYERS_CONFIG=[
            LayerConfig(tileCount=28, cols=7, rows=4, xOffset=0, yOffset=0, layerZ=0),
            LayerConfig(tileCount=20, cols=5, rows=4, xOffset=TILE_WIDTH*0.8, yOffset=TILE_HEIGHT*0.5, layerZ=1),
            LayerConfig(tileCount=12, cols=4, rows=3, xOffset=TILE_WIDTH*0.3, yOffset=TILE_HEIGHT*1.2, layerZ=2),
            LayerConfig(tileCount=8, cols=2, rows=4, xOffset=TILE_WIDTH*2.5, yOffset=TILE_HEIGHT*0.8, layerZ=3),
            LayerConfig(tileCount=4, cols=2, rows=2, xOffset=TILE_WIDTH*1.2, yOffset=TILE_HEIGHT*2.0, layerZ=4),
        ], TOTAL_TILES_TO_GENERATE=72),
}

# --- Core Game Logic Classes ---

@dataclass
class Tile:
    symbol: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    x: int = 0
    y: int = 0
    layer: int = 0
    is_collected: bool = False
    is_matched: bool = False

@dataclass
class GameState:
    board_tiles: List[Tile]
    difficulty_settings: DifficultySettings
    collection_tiles: List[Tile] = field(default_factory=list)
    move_history: List[str] = field(default_factory=list)
    
    @property
    def collection_capacity(self) -> int:
        return self.difficulty_settings.COLLECTION_SLOT_CAPACITY

    def get_accessible_tiles(self) -> List[Tile]:
        """Returns a list of all tiles that can be currently clicked."""
        accessible = []
        for tile in self.board_tiles:
            if not tile.is_collected and not tile.is_matched:
                if self.is_tile_accessible(tile.id):
                    accessible.append(tile)
        return accessible

    def is_tile_accessible(self, tile_id: str) -> bool:
        """
        Checks if a tile is accessible (not covered by any other tile).
        This is a Python port of the checkTileAccessibilityGlobal function.
        """
        try:
            target_tile = next(t for t in self.board_tiles if t.id == tile_id)
        except StopIteration:
            return False # Tile not found

        if target_tile.is_collected or target_tile.is_matched:
            return False

        # Check for obstructing tiles on higher layers
        for other_tile in self.board_tiles:
            if other_tile.id == target_tile.id or other_tile.is_collected or other_tile.is_matched:
                continue

            if other_tile.layer > target_tile.layer:
                # AABB collision check
                t1_left, t1_right = target_tile.x, target_tile.x + TILE_WIDTH
                t1_top, t1_bottom = target_tile.y, target_tile.y + TILE_HEIGHT
                t2_left, t2_right = other_tile.x, other_tile.x + TILE_WIDTH
                t2_top, t2_bottom = other_tile.y, other_tile.y + TILE_HEIGHT

                x_overlap = t1_left < t2_right and t1_right > t2_left
                y_overlap = t1_top < t2_bottom and t1_bottom > t2_top

                if x_overlap and y_overlap:
                    return False  # Target tile is covered

        return True # No covering tiles found

    def apply_move(self, tile: Tile) -> bool:
        """
        Applies a move to the game state: moves tile to collection and handles matches.
        Returns True if a match was made, False otherwise.
        """
        # 1. Update tile state and move to collection
        tile.is_collected = True
        self.collection_tiles.append(tile)
        self.move_history.append(tile.id)

        # 2. Check for matches
        symbol_counts = {}
        for t in self.collection_tiles:
            symbol_counts[t.symbol] = symbol_counts.get(t.symbol, 0) + 1
        
        match_made = False
        for symbol, count in symbol_counts.items():
            if count >= self.difficulty_settings.TILES_PER_MATCH_SET:
                # Match found, remove matched tiles from collection
                self.collection_tiles = [t for t in self.collection_tiles if t.symbol != symbol]
                # Update board tiles status
                for board_tile in self.board_tiles:
                    if board_tile.symbol == symbol and board_tile.is_collected:
                        board_tile.is_matched = True
                match_made = True
                # Assuming only one match can be made per move for simplicity
                break
        
        return match_made

    def is_game_over(self) -> Tuple[bool, str]:
        """
        Checks if the game is over.
        Returns a tuple: (is_over: bool, result: str ('win', 'loss', 'ongoing')).
        """
        # Win condition: all tiles are matched
        if all(t.is_matched for t in self.board_tiles):
            return True, 'win'
        
        # Loss condition 1: collection is full
        if len(self.collection_tiles) >= self.collection_capacity:
            return True, 'loss'
            
        # Loss condition 2: no more accessible tiles, but board is not clear
        if not self.get_accessible_tiles() and any(not t.is_matched for t in self.board_tiles):
            return True, 'loss'
            
        return False, 'ongoing'

def generate_board_for_difficulty(difficulty: str) -> GameState:
    """
    Generates a new game state based on the specified difficulty level.
    This is a direct port of the frontend's generation logic.
    """
    settings = DIFFICULTY_LEVELS[difficulty]
    
    # 1. Create the required pool of tiles
    symbols_to_use = SYMBOLS[:settings.NUM_UNIQUE_SYMBOLS_TO_USE]
    tile_pool = []
    for symbol in symbols_to_use:
        for _ in range(settings.NUM_SETS_PER_SYMBOL):
            for _ in range(settings.TILES_PER_MATCH_SET):
                tile_pool.append(Tile(symbol=symbol))
    
    random.shuffle(tile_pool)
    
    # 2. Place tiles on the board according to layer configuration
    board_tiles = []
    tiles_placed_count = 0
    for layer_conf in settings.LAYERS_CONFIG:
        for i in range(layer_conf.tileCount):
            if tiles_placed_count < len(tile_pool):
                tile = tile_pool[tiles_placed_count]
                row = i // layer_conf.cols
                col = i % layer_conf.cols
                
                tile.x = int(layer_conf.xOffset + col * TILE_WIDTH)
                tile.y = int(layer_conf.yOffset + row * TILE_HEIGHT)
                tile.layer = layer_conf.layerZ
                
                board_tiles.append(tile)
                tiles_placed_count += 1
                
    # Ensure all tiles from the pool are placed, even if config is slightly off
    while tiles_placed_count < len(tile_pool):
        tile = tile_pool[tiles_placed_count]
        tile.layer = settings.LAYERS_CONFIG[-1].layerZ # Put remaining in top layer
        tile.x = random.randint(0, 400)
        tile.y = random.randint(0, 400)
        board_tiles.append(tile)
        tiles_placed_count += 1
        
    return GameState(board_tiles=board_tiles, difficulty_settings=settings)