#!/usr/bin/env python3
import curses
import random
import time

# --- Constants ---
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
INITIAL_SPEED = 0.5
MIN_SPEED = 0.05

# Tetromino definitions (4 rotations each)
# 0: Empty, 1: Block
SHAPES = {
    'I': [
        [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
        [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]
    ],
    'J': [
        [[1, 0, 0], [1, 1, 1], [0, 0, 0]],
        [[0, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 1], [0, 0, 1]],
        [[0, 1, 0], [0, 1, 0], [1, 1, 0]]
    ],
    'L': [
        [[0, 0, 1], [1, 1, 1], [0, 0, 0]],
        [[0, 1, 0], [0, 1, 0], [0, 1, 1]],
        [[0, 0, 0], [1, 1, 1], [1, 0, 0]],
        [[1, 1, 0], [0, 1, 0], [0, 1, 0]]
    ],
    'O': [
        [[1, 1], [1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1], [1, 1]]
    ],
    'S': [
        [[0, 1, 1], [1, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [0, 1, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 1], [1, 1, 0]],
        [[1, 0, 0], [1, 1, 0], [0, 1, 0]]
    ],
    'T': [
        [[0, 1, 0], [1, 1, 1], [0, 0, 0]],
        [[0, 1, 0], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 1, 0], [0, 1, 0]]
    ],
    'Z': [
        [[1, 1, 0], [0, 1, 1], [0, 0, 0]],
        [[0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 0], [0, 1, 1]],
        [[0, 1, 0], [1, 1, 0], [1, 0, 0]]
    ]
}

# Color IDs for curses
COLORS = {
    'I': 1, 'J': 2, 'L': 3, 'O': 4, 'S': 5, 'T': 6, 'Z': 7
}

class Piece:
    def __init__(self, shape_type):
        self.type = shape_type
        self.shape = SHAPES[shape_type]
        self.rotation = 0
        self.x = BOARD_WIDTH // 2 - 2  # Center horizontally
        self.y = 0  # Start at top

    def get_cells(self):
        cells = []
        matrix = self.shape[self.rotation]
        for r in range(len(matrix)):
            for c in range(len(matrix[r])):
                if matrix[r][c]:
                    cells.append((self.x + c, self.y + r))
        return cells

    def rotate(self):
        self.rotation = (self.rotation + 1) % 4

class Board:
    def __init__(self):
        self.grid = [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.cleared_lines = 0
        self.game_over = False

    def is_valid_move(self, piece, dx=0, dy=0):
        new_x = piece.x + dx
        new_y = piece.y + dy
        
        # Check boundaries
        if new_x < 0 or new_x + len(piece.shape[0]) > BOARD_WIDTH:
            return False
        if new_y + len(piece.shape) > BOARD_HEIGHT:
            return False

        # Check collisions with locked pieces
        for r in range(len(piece.shape)):
            for c in range(len(piece.shape[r])):
                if piece.shape[r][c]:
                    cell_x = new_x + c
                    cell_y = new_y + r
                    if cell_y >= BOARD_HEIGHT or self.grid[cell_y][cell_x] != 0:
                        return False
        return True

    def lock_piece(self, piece):
        for r in range(len(piece.shape)):
            for c in range(len(piece.shape[r])):
                if piece.shape[r][c]:
                    y = piece.y + r
                    x = piece.x + c
                    if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                        self.grid[y][x] = COLORS[piece.type]
        
        self.clear_lines()
        
        # Check Game Over
        if self.grid[0][0] != 0: # Simplified check, usually check top row
            # More robust check:
            for x in range(BOARD_WIDTH):
                if self.grid[0][x] != 0:
                    self.game_over = True
                    break

    def clear_lines(self):
        lines_to_clear = []
        for r in range(BOARD_HEIGHT):
            if all(self.grid[r][c] != 0 for c in range(BOARD_WIDTH)):
                lines_to_clear.append(r)
        
        if lines_to_clear:
            self.cleared_lines += len(lines_to_clear)
            for r in lines_to_clear:
                del self.grid[r]
                self.grid.insert(0, [0 for _ in range(BOARD_WIDTH)])

    def get_topmost_row(self):
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                if self.grid[r][c] != 0:
                    return r
        return -1

class Renderer:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.init_colors()
        self.height, self.width = stdscr.getmaxyx()

    def init_colors(self):
        curses.start_color()
        curses.use_default_colors()
        # Define color pairs: (ID, Foreground, Background)
        # Background is -1 (default)
        curses.init_pair(1, curses.COLOR_CYAN, -1)    # I
        curses.init_pair(2, curses.COLOR_BLUE, -1)     # J
        curses.init_pair(3, curses.COLOR_YELLOW, -1)   # L
        curses.init_pair(4, curses.COLOR_YELLOW, -1)   # O (Yellow)
        curses.init_pair(5, curses.COLOR_GREEN, -1)    # S
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # T
        curses.init_pair(7, curses.COLOR_RED, -1)      # Z

    def draw_board(self, board, piece=None):
        # Clear screen
        self.stdscr.clear()
        
        # Draw Board Border
        for r in range(BOARD_HEIGHT + 2):
            self.stdscr.addstr(r, 0, "+")
            self.stdscr.addstr(r, BOARD_WIDTH + 1, "+")
        for c in range(BOARD_WIDTH + 2):
            self.stdscr.addstr(0, c, "+")
            self.stdscr.addstr(BOARD_HEIGHT + 1, c, "+")

        # Draw Grid
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                val = board.grid[r][c]
                if val != 0:
                    color = curses.color_pair(val)
                    self.stdscr.addstr(r + 1, c + 1, " ", color)
                else:
                    self.stdscr.addstr(r + 1, c + 1, " ")

        # Draw Active Piece
        if piece:
            for r in range(len(piece.shape)):
                for c in range(len(piece.shape[r])):
                    if piece.shape[r][c]:
                        y = piece.y + r
                        x = piece.x + c
                        if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                            color = curses.color_pair(COLORS[piece.type])
                            self.stdscr.addstr(y + 1, x + 1, " ", color)

        # Draw UI (Score, Level, Next Piece)
        # We'll draw this on the right side of the screen
        ui_x = BOARD_WIDTH + 3
        ui_y = 2
        
        self.stdscr.addstr(ui_y, ui_x, "TETRIS")
        self.stdscr.addstr(ui_y + 2, ui_x, "Score: {}".format(board.cleared_lines))
        self.stdscr.addstr(ui_y + 4, ui_x, "Level: 1")
        self.stdscr.addstr(ui_y + 6, ui_x, "Controls:")
        self.stdscr.addstr(ui_y + 7, ui_x, "Arrows: Move")
        self.stdscr.addstr(ui_y + 8, ui_x, "Up: Rotate")
        self.stdscr.addstr(ui_y + 9, ui_x, "Q: Quit")

        # Draw Game Over
        if board.game_over:
            self.stdscr.addstr(BOARD_HEIGHT // 2, BOARD_WIDTH // 2 - 5, "GAME OVER")
            self.stdscr.addstr(BOARD_HEIGHT // 2 + 2, BOARD_WIDTH // 2 - 5, "Press Q to Quit")

        self.stdscr.refresh()

class TetrisGame:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.board = Board()
        self.renderer = Renderer(stdscr)
        self.current_piece = None
        self.last_tick = time.time()
        self.tick_rate = INITIAL_SPEED
        self.input_queue = []
        self.running = True
        
        # Setup Curses
        curses.curs_set(0)
        self.stdscr.nodelay(True)
        self.stdscr.keypad(True)
        
        self.spawn_piece()

    def spawn_piece(self):
        shape_type = random.choice(list(SHAPES.keys()))
        self.current_piece = Piece(shape_type)
        
        # Check immediate game over
        if not self.board.is_valid_move(self.current_piece):
            self.board.game_over = True

    def handle_input(self, key):
        if key == ord('q') or key == ord('x'):
            self.running = False
            return
        
        if self.board.game_over:
            return

        if key == curses.KEY_UP:
            if self.board.is_valid_move(self.current_piece, 0, 0):
                self.current_piece.rotate()
        elif key == curses.KEY_LEFT:
            if self.board.is_valid_move(self.current_piece, -1, 0):
                self.current_piece.x -= 1
        elif key == curses.KEY_RIGHT:
            if self.board.is_valid_move(self.current_piece, 1, 0):
                self.current_piece.x += 1
        elif key == curses.KEY_DOWN:
            if self.board.is_valid_move(self.current_piece, 0, 1):
                self.current_piece.y += 1
            else:
                # Hard drop logic could go here, but for now just lock
                self.board.lock_piece(self.current_piece)
                self.spawn_piece()

    def game_tick(self):
        if self.board.game_over:
            return

        if not self.board.is_valid_move(self.current_piece, 0, 1):
            self.board.lock_piece(self.current_piece)
            self.spawn_piece()
        else:
            self.current_piece.y += 1

    def run(self):
        while self.running:
            # Input Handling
            try:
                key = self.stdscr.getch()
                if key != -1:
                    self.handle_input(key)
            except:
                pass

            # Game Logic Tick
            current_time = time.time()
            if current_time - self.last_tick >= self.tick_rate:
                self.game_tick()
                self.last_tick = current_time

            # Rendering
            self.renderer.draw_board(self.board, self.current_piece)

            # Check Game Over loop condition
            if self.board.game_over:
                self.renderer.draw_board(self.board, self.current_piece)
                time.sleep(1)
                self.running = False

def main(stdscr):
    game = TetrisGame(stdscr)
    game.run()

if __name__ == "__main__":
    curses.wrapper(main)