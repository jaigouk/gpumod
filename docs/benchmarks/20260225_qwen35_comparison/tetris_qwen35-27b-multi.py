#!/usr/bin/env python3
import curses
import random
import time

# --- Constants ---
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
EMPTY = 0

# Tetromino definitions: (Shape Matrix, Color ID)
# Colors: 1=Blue(I), 2=Orange(L), 3=Yellow(O), 4=Green(S), 5=Cyan(T), 6=Purple(Z), 7=Red(J)
TETROMINOS = {
    'I': ([[1, 1, 1, 1]], curses.COLOR_CYAN),
    'O': ([[1, 1], [1, 1]], curses.COLOR_YELLOW),
    'T': ([[0, 1, 0], [1, 1, 1]], curses.COLOR_MAGENTA),
    'S': ([[0, 1, 1], [1, 1, 0]], curses.COLOR_GREEN),
    'Z': ([[1, 1, 0], [0, 1, 1]], curses.COLOR_RED),
    'J': ([[1, 0, 0], [1, 1, 1]], curses.COLOR_BLUE),
    'L': ([[0, 0, 1], [1, 1, 1]], curses.COLOR_ORANGE)
}

# --- Classes ---

class Tetromino:
    """Represents a single piece type with its shape and color."""
    def __init__(self, shape, color_id):
        self.shape = shape  # List of lists (2D matrix)
        self.color_id = color_id

    def rotate(self):
        """Returns a new shape matrix rotated 90 degrees clockwise."""
        # Transpose and reverse rows for clockwise rotation
        return [list(row) for row in zip(*self.shape[::-1])]

class Piece:
    """Represents an active piece currently falling in the game."""
    def __init__(self, piece_type):
        self.type = piece_type
        self.shape = [row[:] for row in TETROMINOS[piece_type][0]]
        self.color_id = TETROMINOS[piece_type][1]
        self.x = BOARD_WIDTH // 2 - 2
        self.y = 0

    def get_rotated_shape(self):
        """Returns the shape rotated 90 degrees clockwise."""
        return [list(row) for row in zip(*self.shape[::-1])]

class Board:
    """Manages the game grid and static state."""
    def __init__(self):
        self.grid = [[EMPTY for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]

    def check_collision(self, piece, offset_x, offset_y, shape=None):
        """
        Checks if the piece at the proposed position/shape collides with walls or locked blocks.
        Returns True if collision occurs, False otherwise.
        """
        if shape is None:
            shape = piece.shape

        for py, row in enumerate(shape):
            for px, cell in enumerate(row):
                if cell:
                    new_x = piece.x + px + offset_x
                    new_y = piece.y + py + offset_y

                    # Wall collisions
                    if new_x < 0 or new_x >= BOARD_WIDTH or new_y >= BOARD_HEIGHT:
                        return True
                    
                    # Block collisions (only check if within grid bounds vertically)
                    if new_y >= 0 and self.grid[new_y][new_x] != EMPTY:
                        return True
        return False

    def lock_piece(self, piece):
        """Writes the Piece's shape into the Board's grid."""
        for py, row in enumerate(piece.shape):
            for px, cell in enumerate(row):
                if cell:
                    if 0 <= piece.y + py < BOARD_HEIGHT and 0 <= piece.x + px < BOARD_WIDTH:
                        self.grid[piece.y + py][piece.x + px] = piece.color_id

    def clear_lines(self):
        """
        Identifies full rows, removes them, shifts rows down.
        Returns the count of lines cleared.
        """
        lines_cleared = 0
        new_grid = []
        
        for row in self.grid:
            if all(cell != EMPTY for cell in row):
                lines_cleared += 1
            else:
                new_grid.append(row)
        
        # Add new empty rows at the top
        while len(new_grid) < BOARD_HEIGHT:
            new_grid.insert(0, [EMPTY for _ in range(BOARD_WIDTH)])
        
        self.grid = new_grid
        return lines_cleared

class Game:
    """The main controller orchestrating the flow."""
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.board = Board()
        self.current_piece = None
        self.next_piece_type = None
        self.score = 0
        self.level = 1
        self.game_over = False
        self.paused = False
        self.last_fall_time = 0
        self.drop_interval = 0.5  # Seconds

        # Initialize curses settings
        curses.curs_set(0)
        self.stdscr.nodelay(True)
        self.stdscr.timeout(10)  # 10ms refresh rate for input
        curses.start_color()
        curses.use_default_colors()
        
        # Initialize colors
        for i in range(1, 8):
            curses.init_pair(i, i, -1)

        self.spawn_piece()

    def spawn_piece(self):
        """Generates a new piece."""
        if self.next_piece_type is None:
            self.next_piece_type = random.choice(list(TETROMINOS.keys()))
        
        self.current_piece = Piece(self.next_piece_type)
        self.next_piece_type = random.choice(list(TETROMINOS.keys()))

        # Check for immediate game over
        if self.board.check_collision(self.current_piece, 0, 0):
            self.game_over = True

    def rotate_piece(self):
        """Attempts to rotate the current piece."""
        new_shape = self.current_piece.get_rotated_shape()
        if not self.board.check_collision(self.current_piece, 0, 0, new_shape):
            self.current_piece.shape = new_shape
            return True
        return False

    def move_piece(self, dx, dy):
        """Attempts to move the piece. Returns True if successful."""
        if not self.board.check_collision(self.current_piece, dx, dy):
            self.current_piece.x += dx
            self.current_piece.y += dy
            return True
        return False

    def lock_and_clear(self):
        """Locks the piece, clears lines, and spawns a new one."""
        self.board.lock_piece(self.current_piece)
        lines = self.board.clear_lines()
        
        if lines > 0:
            # Scoring: 100, 300, 500, 800 for 1, 2, 3, 4 lines
            self.score += [0, 100, 300, 500, 800][lines] * self.level
            self.level = self.score // 1000 + 1
            self.drop_interval = max(0.1, 0.5 - (self.level - 1) * 0.05)

        self.spawn_piece()

    def run(self):
        """Main game loop."""
        while not self.game_over:
            start_time = time.time()
            
            # Input Handling
            try:
                key = self.stdscr.getch()
            except:
                key = -1

            if key == ord('q') or key == ord('Q'):
                break
            if key == ord('p') or key == ord('P'):
                self.paused = not self.paused
            
            if not self.paused:
                if key == ord('h') or key == curses.KEY_LEFT:
                    self.move_piece(-1, 0)
                elif key == ord('l') or key == curses.KEY_RIGHT:
                    self.move_piece(1, 0)
                elif key == ord('j') or key == curses.KEY_DOWN:
                    if self.move_piece(0, 1):
                        self.score += 1 # Soft drop points
                elif key == ord('k') or key == curses.KEY_UP:
                    self.rotate_piece()
                elif key == 27: # ESC
                    break

            # Gravity Logic
            if not self.paused:
                current_time = time.time()
                if current_time - self.last_fall_time > self.drop_interval:
                    if not self.move_piece(0, 1):
                        self.lock_and_clear()
                    self.last_fall_time = current_time

            # Rendering
            self.render()

            # Sleep to control loop speed (prevent CPU overuse)
            elapsed = time.time() - start_time
            if elapsed < 0.01:
                time.sleep(0.01 - elapsed)

        # Cleanup
        self.game_over_screen()

    def render(self):
        """Renders the current state to the terminal."""
        self.stdscr.clear()
        height, width = self.stdscr.getmaxyx()
        
        # Draw Board
        # Center the board horizontally
        start_x = (width - (BOARD_WIDTH * 2 + 2)) // 2
        start_y = (height - BOARD_HEIGHT - 4) // 2

        # Draw Top Border
        self.stdscr.addstr(start_y, start_x, "┌" + "─" * (BOARD_WIDTH * 2) + "┐")
        
        # Draw Grid
        for y in range(BOARD_HEIGHT):
            row_str = "│"
            for x in range(BOARD_WIDTH):
                cell = self.board.grid[y][x]
                if cell == EMPTY:
                    row_str += "  "
                else:
                    # Draw filled block
                    row_str += f"██"
            row_str += "│"
            self.stdscr.addstr(start_y + y + 1, start_x, row_str)
        
        # Draw Bottom Border
        self.stdscr.addstr(start_y + BOARD_HEIGHT + 1, start_x, "└" + "─" * (BOARD_WIDTH * 2) + "┘")

        # Draw Active Piece
        if self.current_piece:
            shape = self.current_piece.shape
            color = self.current_piece.color_id
            for py, row in enumerate(shape):
                for px, cell in enumerate(row):
                    if cell:
                        draw_y = start_y + 1 + self.current_piece.y + py
                        draw_x = start_x + 1 + (self.current_piece.x + px) * 2
                        try:
                            self.stdscr.addstr(draw_y, draw_x, "██", curses.color_pair(color) | curses.A_BOLD)
                        except curses.error:
                            pass # Ignore drawing errors if off-screen

        # Draw Sidebar (Score, Level, Next)
        sidebar_x = start_x + BOARD_WIDTH * 2 + 5
        self.stdscr.addstr(start_y + 2, sidebar_x, "SCORE:")
        self.stdscr.addstr(start_y + 3, sidebar_x, str(self.score))
        
        self.stdscr.addstr(start_y + 5, sidebar_x, "LEVEL:")
        self.stdscr.addstr(start_y + 6, sidebar_x, str(self.level))
        
        self.stdscr.addstr(start_y + 8, sidebar_x, "NEXT:")
        if self.next_piece_type:
            next_shape = TETROMINOS[self.next_piece_type][0]
            next_color = TETROMINOS[self.next_piece_type][1]
            for py, row in enumerate(next_shape):
                for px, cell in enumerate(row):
                    if cell:
                        try:
                            self.stdscr.addstr(start_y + 9 + py, sidebar_x + 1 + px * 2, "██", curses.color_pair(next_color))
                        except curses.error:
                            pass

        # Draw Controls
        self.stdscr.addstr(height - 3, 2, "Controls: Arrows/WASD | P: Pause | Q: Quit")
        
        if self.paused:
            self.stdscr.addstr(height // 2, width // 2 - 3, "PAUSED", curses.A_REVERSE)

        self.stdscr.refresh()

    def game_over_screen(self):
        """Displays the Game Over screen."""
        height, width = self.stdscr.getmaxyx()
        self.stdscr.clear()
        self.stdscr.addstr(height // 2 - 1, (width - 15) // 2, "GAME OVER", curses.A_REVERSE)
        self.stdscr.addstr(height // 2, (width - 20) // 2, f"Final Score: {self.score}")
        self.stdscr.addstr(height // 2 + 2, (width - 20) // 2, "Press Q to Quit")
        self.stdscr.refresh()
        
        while True:
            key = self.stdscr.getch()
            if key == ord('q') or key == ord('Q'):
                break

def main(stdscr):
    game = Game(stdscr)
    game.run()

if __name__ == "__main__":
    curses.wrapper(main)