#!/usr/bin/env python3
import curses
import random
import time

# --- Constants & Configuration ---
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
FPS = 60
DROP_INTERVAL_BASE = 0.5  # Seconds per drop at level 1

# Tetromino Definitions (Matrix representations)
# 1: I, 2: J, 3: L, 4: O, 5: S, 6: T, 7: Z
SHAPES = {
    1: [[1, 1, 1, 1]],  # I
    2: [[1, 0, 0], [1, 1, 1]],  # J
    3: [[0, 0, 1], [1, 1, 1]],  # L
    4: [[1, 1], [1, 1]],  # O
    5: [[0, 1, 1], [1, 1, 0]],  # S
    6: [[0, 1, 0], [1, 1, 1]],  # T
    7: [[1, 1, 0], [0, 1, 1]]   # Z
}

# Color IDs (1-7)
COLORS = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7
}

# --- Classes ---

class Piece:
    def __init__(self, shape_id):
        self.shape_id = shape_id
        self.matrix = [row[:] for row in SHAPES[shape_id]]
        self.x = 0
        self.y = 0
        # Center the piece horizontally
        self.x = (BOARD_WIDTH - len(self.matrix[0])) // 2
        self.y = 0

    def rotate(self):
        # Transpose and reverse rows
        self.matrix = [list(row) for row in zip(*self.matrix[::-1])]

    def get_shape(self):
        return self.matrix

class Board:
    def __init__(self):
        # 0 = Empty, >0 = Color ID
        self.grid = [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]

    def is_valid_move(self, piece, x_offset, y_offset):
        new_x = piece.x + x_offset
        new_y = piece.y + y_offset
        
        for row_idx, row in enumerate(piece.matrix):
            for col_idx, val in enumerate(row):
                if val != 0:
                    new_col = new_x + col_idx
                    new_row = new_y + row_idx
                    
                    # Check boundaries
                    if new_col < 0 or new_col >= BOARD_WIDTH:
                        return False
                    if new_row >= BOARD_HEIGHT:
                        return False
                    # Check if row is valid (can't be negative as we spawn above)
                    if new_row >= 0:
                        if self.grid[new_row][new_col] != 0:
                            return False
        return True

    def place_piece(self, piece):
        for row_idx, row in enumerate(piece.matrix):
            for col_idx, val in enumerate(row):
                if val != 0:
                    y = piece.y + row_idx
                    x = piece.x + col_idx
                    if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                        self.grid[y][x] = val

    def clear_lines(self):
        lines_cleared = 0
        # Iterate from bottom to top
        for row in range(BOARD_HEIGHT - 1, -1, -1):
            if all(cell != 0 for cell in self.grid[row]):
                # Remove row
                del self.grid[row]
                # Add new empty row at top
                self.grid.insert(0, [0 for _ in range(BOARD_WIDTH)])
                lines_cleared += 1
        return lines_cleared

    def get_top_row(self):
        # Find the highest occupied row index, or -1 if empty
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if self.grid[y][x] != 0:
                    return y
        return -1

    def check_game_over(self, piece):
        # If a piece spawns and immediately collides
        if not self.is_valid_move(piece, 0, 0):
            return True
        return False

    def get_next_piece_spawn_y(self):
        # Spawn just above the highest block, or at top if empty
        top_y = self.get_top_row()
        if top_y == -1:
            return 0
        return top_y - len(piece.matrix) # Approximate, logic handled in Game class

# --- Main Game Controller ---

class TetrisGame:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.board = Board()
        self.score = 0
        self.level = 1
        self.game_over = False
        self.paused = False
        
        # Initialize Colors
        curses.start_color()
        curses.use_default_colors()
        for i in range(1, 8):
            curses.init_pair(i, i, -1) # Foreground color, default background

        # Setup UI
        curses.curs_set(0) # Hide cursor
        self.stdscr.nodelay(True) # Non-blocking input
        self.stdscr.timeout(1000 // FPS) # Refresh rate

        # Game State
        self.current_piece = self.spawn_piece()
        self.next_piece_id = random.randint(1, 7)
        self.last_drop_time = time.time()
        self.drop_interval = DROP_INTERVAL_BASE

        # Input Buffer
        self.input_buffer = []

    def spawn_piece(self):
        piece = Piece(self.next_piece_id)
        self.next_piece_id = random.randint(1, 7)
        
        # Check immediate collision
        if self.board.check_game_over(piece):
            self.game_over = True
        
        return piece

    def handle_input(self, key):
        if self.game_over:
            if key == ord('q') or key == ord('Q'):
                return False # Quit
            return True # Keep loop running for restart logic if needed

        if key == ord('q') or key == ord('Q'):
            return False # Quit

        if key == ord('p') or key == ord('P'):
            self.paused = not self.paused
            return True

        if self.paused:
            return True

        # Movement
        if key == curses.KEY_LEFT:
            if self.board.is_valid_move(self.current_piece, -1, 0):
                self.current_piece.x -= 1
        elif key == curses.KEY_RIGHT:
            if self.board.is_valid_move(self.current_piece, 1, 0):
                self.current_piece.x += 1
        elif key == curses.KEY_DOWN:
            if self.board.is_valid_move(self.current_piece, 0, 1):
                self.current_piece.y += 1
                # Optional: Soft drop score bonus?
        elif key == curses.KEY_UP:
            # Rotate
            old_matrix = [row[:] for row in self.current_piece.matrix]
            self.current_piece.rotate()
            # Wall kick (basic): if rotation fails, try moving left/right
            if not self.board.is_valid_move(self.current_piece, 0, 0):
                if self.board.is_valid_move(self.current_piece, -1, 0):
                    self.current_piece.x -= 1
                elif self.board.is_valid_move(self.current_piece, 1, 0):
                    self.current_piece.x += 1
                else:
                    # Revert
                    self.current_piece.matrix = old_matrix
        elif key == ord(' '): # Space: Hard Drop
            while self.board.is_valid_move(self.current_piece, 0, 1):
                self.current_piece.y += 1
            # Lock immediately after loop
            self.lock_piece()
            return True

        return True

    def lock_piece(self):
        self.board.place_piece(self.current_piece)
        lines = self.board.clear_lines()
        
        # Score calculation (Classic)
        if lines > 0:
            points = [0, 40, 100, 300, 1200]
            self.score += points[lines] * self.level
            # Level up every 1000 points (simplified)
            self.level = 1 + (self.score // 1000)
            self.drop_interval = max(0.05, DROP_INTERVAL_BASE / (1 + (self.level - 1) * 0.1))

        self.current_piece = self.spawn_piece()

    def update_game_logic(self):
        if self.game_over or self.paused:
            return

        current_time = time.time()
        if current_time - self.last_drop_time >= self.drop_interval:
            if self.board.is_valid_move(self.current_piece, 0, 1):
                self.current_piece.y += 1
            else:
                self.lock_piece()
            self.last_drop_time = current_time

    def render(self):
        self.stdscr.clear()
        
        # Draw Board Border
        try:
            # Top/Bottom
            self.stdscr.addstr(0, 0, "+" + "-" * BOARD_WIDTH + "+")
            self.stdscr.addstr(BOARD_HEIGHT + 1, 0, "+" + "-" * BOARD_WIDTH + "+")
            # Sides
            for y in range(1, BOARD_HEIGHT + 1):
                self.stdscr.addstr(y, 0, "|")
                self.stdscr.addstr(y, BOARD_WIDTH, "|")
            
            # Draw Board Content
            for y in range(1, BOARD_HEIGHT + 1):
                for x in range(1, BOARD_WIDTH + 1):
                    # Check if this cell is part of the static board
                    # Board grid is 0-indexed, screen is 1-indexed
                    cell_val = self.board.grid[y-1][x-1]
                    if cell_val != 0:
                        self.stdscr.addstr(y, x, " ", curses.color_pair(cell_val) | curses.A_BOLD)
            
            # Draw Active Piece
            if self.current_piece:
                for y_off, row in enumerate(self.current_piece.matrix):
                    for x_off, val in enumerate(row):
                        if val != 0:
                            screen_y = self.current_piece.y + y_off + 1
                            screen_x = self.current_piece.x + x_off + 1
                            if 0 <= screen_y < BOARD_HEIGHT + 2 and 0 <= screen_x < BOARD_WIDTH + 2:
                                self.stdscr.addstr(screen_y, screen_x, " ", curses.color_pair(val) | curses.A_BOLD)

            # Draw UI (Score, Level, Next Piece)
            ui_y = BOARD_HEIGHT + 3
            self.stdscr.addstr(ui_y, 0, f"Score: {self.score}")
            self.stdscr.addstr(ui_y + 1, 0, f"Level: {self.level}")
            
            # Next Piece Preview
            self.stdscr.addstr(ui_y, 15, "Next:")
            next_piece = Piece(self.next_piece_id)
            # Adjust preview position
            preview_x = 15
            preview_y = ui_y + 2
            for y_off, row in enumerate(next_piece.matrix):
                for x_off, val in enumerate(row):
                    if val != 0:
                        if preview_y + y_off < BOARD_HEIGHT + 10:
                            self.stdscr.addstr(preview_y + y_off, preview_x + x_off, " ", curses.color_pair(val))

            # Game Over Message
            if self.game_over:
                self.stdscr.addstr(BOARD_HEIGHT // 2, BOARD_WIDTH // 2 - 5, "GAME OVER")
                self.stdscr.addstr(BOARD_HEIGHT // 2 + 2, BOARD_WIDTH // 2 - 5, "Press 'q' to Quit")

            # Pause Message
            if self.paused:
                self.stdscr.addstr(BOARD_HEIGHT // 2, BOARD_WIDTH // 2 - 3, "PAUSED")

            self.stdscr.refresh()
        except curses.error:
            pass # Handle resize or overflow gracefully

    def run(self):
        while True:
            try:
                # Input Handling
                key = self.stdscr.getch()
                if key != -1:
                    self.handle_input(key)
                
                # Logic Update
                self.update_game_logic()
                
                # Render
                self.render()
                
                # Sleep to control FPS
                time.sleep(1.0 / FPS)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                # Handle terminal resize or other errors gracefully
                pass

def main(stdscr):
    game = TetrisGame(stdscr)
    game.run()

if __name__ == "__main__":
    curses.wrapper(main)