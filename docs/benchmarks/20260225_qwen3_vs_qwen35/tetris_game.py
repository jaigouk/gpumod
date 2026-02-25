#!/usr/bin/env python3

import curses
import random
import time

class Piece:
    def __init__(self, shape, x=0, y=0, rotation=0):
        self.shape = shape
        self.x = x
        self.y = y
        self.rotation = rotation

    def get_rotated_shape(self, rotation_offset):
        current_rotation = (self.rotation + rotation_offset) % 4
        return self.shape[current_rotation]

    def get_cells(self):
        cells = []
        shape = self.get_rotated_shape(0)
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    cells.append((self.x + x, self.y + y))
        return cells

    def copy(self):
        return Piece(self.shape, self.x, self.y, self.rotation)

class PieceFactory:
    PIECES = {
        'I': [
            [[0,0,0,0], [1,1,1,1], [0,0,0,0], [0,0,0,0]],
            [[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0]],
            [[0,0,0,0], [0,0,0,0], [1,1,1,1], [0,0,0,0]],
            [[0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0]]
        ],
        'O': [
            [[1,1], [1,1]],
            [[1,1], [1,1]],
            [[1,1], [1,1]],
            [[1,1], [1,1]]
        ],
        'T': [
            [[0,1,0], [1,1,1], [0,0,0]],
            [[0,1,0], [0,1,1], [0,1,0]],
            [[0,0,0], [1,1,1], [0,1,0]],
            [[0,1,0], [1,1,0], [0,1,0]]
        ],
        'S': [
            [[0,1,1], [1,1,0], [0,0,0]],
            [[0,1,0], [0,1,1], [0,0,1]],
            [[0,0,0], [0,1,1], [1,1,0]],
            [[1,0,0], [1,1,0], [0,1,0]]
        ],
        'Z': [
            [[1,1,0], [0,1,1], [0,0,0]],
            [[0,0,1], [0,1,1], [0,1,0]],
            [[0,0,0], [1,1,0], [1,1,0]],
            [[0,1,0], [1,1,0], [1,0,0]]
        ],
        'J': [
            [[1,0,0], [1,1,1], [0,0,0]],
            [[0,1,1], [0,1,0], [0,1,0]],
            [[0,0,0], [1,1,1], [0,0,1]],
            [[0,1,0], [0,1,0], [1,1,0]]
        ],
        'L': [
            [[0,0,1], [1,1,1], [0,0,0]],
            [[0,1,0], [0,1,0], [0,1,1]],
            [[0,0,0], [1,1,1], [1,0,0]],
            [[1,1,0], [0,1,0], [0,1,0]]
        ]
    }

    @staticmethod
    def create_random_piece():
        piece_type = random.choice(list(PieceFactory.PIECES.keys()))
        return Piece(PieceFactory.PIECES[piece_type])

class Board:
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]

    def is_valid_position(self, piece, dx=0, dy=0):
        for x, y in piece.get_cells():
            new_x, new_y = x + dx, y + dy
            if new_x < 0 or new_x >= self.width or new_y >= self.height:
                return False
            if new_y >= 0 and self.grid[new_y][new_x]:
                return False
        return True

    def place_piece(self, piece):
        for x, y in piece.get_cells():
            if y >= 0:
                self.grid[y][x] = 1

    def clear_lines(self):
        lines_to_clear = []
        for y in range(self.height):
            if all(self.grid[y]):
                lines_to_clear.append(y)
        
        for line in lines_to_clear:
            del self.grid[line]
            self.grid.insert(0, [0 for _ in range(self.width)])
        
        return len(lines_to_clear)

    def is_game_over(self):
        for x in range(self.width):
            if self.grid[0][x]:
                return True
        return False

class TetrisGame:
    def __init__(self):
        self.board = Board()
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.game_over = False
        self.fall_speed = 0.5
        self.last_fall_time = time.time()
        self.piece_factory = PieceFactory()
        self.spawn_piece()

    def spawn_piece(self):
        if self.next_piece is None:
            self.next_piece = self.piece_factory.create_random_piece()
        self.current_piece = self.next_piece
        self.next_piece = self.piece_factory.create_random_piece()
        if not self.board.is_valid_position(self.current_piece):
            self.game_over = True

    def move_piece(self, dx, dy):
        if self.current_piece and not self.game_over:
            if self.board.is_valid_position(self.current_piece, dx, dy):
                self.current_piece.x += dx
                self.current_piece.y += dy
                return True
        return False

    def rotate_piece(self):
        if self.current_piece and not self.game_over:
            original_rotation = self.current_piece.rotation
            self.current_piece.rotation = (self.current_piece.rotation + 1) % 4
            if not self.board.is_valid_position(self.current_piece):
                self.current_piece.rotation = original_rotation
                return False
            return True
        return False

    def hard_drop(self):
        if self.current_piece and not self.game_over:
            while self.move_piece(0, 1):
                pass
            self.lock_piece()

    def lock_piece(self):
        self.board.place_piece(self.current_piece)
        lines_cleared = self.board.clear_lines()
        if lines_cleared > 0:
            self.lines_cleared += lines_cleared
            self.score += [0, 40, 100, 300, 1200][lines_cleared] * self.level
            self.level = self.lines_cleared // 10 + 1
            self.fall_speed = max(0.05, 0.5 - (self.level - 1) * 0.05)
        self.spawn_piece()

    def update(self):
        if self.game_over:
            return

        current_time = time.time()
        if current_time - self.last_fall_time > self.fall_speed:
            if not self.move_piece(0, 1):
                self.lock_piece()
            self.last_fall_time = current_time

    def render(self, stdscr):
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        
        # Draw board
        board_height = self.board.height
        board_width = self.board.width
        start_y = (height - board_height) // 2
        start_x = (width - board_width * 2) // 2
        
        # Draw board border
        for y in range(board_height + 2):
            for x in range(board_width * 2 + 2):
                if y == 0 and x == 0:
                    stdscr.addch(start_y + y, start_x + x, '+')
                elif y == 0 and x == board_width * 2 + 1:
                    stdscr.addch(start_y + y, start_x + x, '+')
                elif y == board_height + 1 and x == 0:
                    stdscr.addch(start_y + y, start_x + x, '+')
                elif y == board_height + 1 and x == board_width * 2 + 1:
                    stdscr.addch(start_y + y, start_x + x, '+')
                elif y == 0 or y == board_height + 1:
                    stdscr.addch(start_y + y, start_x + x, '-')
                elif x == 0 or x == board_width * 2 + 1:
                    stdscr.addch(start_y + y, start_x + x, '|')
        
        # Draw board content
        for y in range(board_height):
            for x in range(board_width):
                if self.board.grid[y][x]:
                    stdscr.addstr(start_y + y + 1, start_x + x * 2 + 1, '[]')
        
        # Draw current piece
        if self.current_piece:
            for x, y in self.current_piece.get_cells():
                if y >= 0:
                    stdscr.addstr(start_y + y + 1, start_x + x * 2 + 1, '[]')
        
        # Draw next piece preview
        next_y = start_y + board_height + 3
        stdscr.addstr(next_y, start_x, "Next:")
        if self.next_piece:
            shape = self.next_piece.get_rotated_shape(0)
            for y, row in enumerate(shape):
                for x, cell in enumerate(row):
                    if cell:
                        stdscr.addstr(next_y + y + 1, start_x + x * 2 + 1, '[]')
        
        # Draw score
        score_y = start_y + board_height + 6
        stdscr.addstr(score_y, start_x, f"Score: {self.score}")
        stdscr.addstr(score_y + 1, start_x, f"Level: {self.level}")
        stdscr.addstr(score_y + 2, start_x, f"Lines: {self.lines_cleared}")
        
        # Draw controls
        controls_y = start_y + board_height + 9
        stdscr.addstr(controls_y, start_x, "Controls:")
        stdscr.addstr(controls_y + 1, start_x, "← → : Move")
        stdscr.addstr(controls_y + 2, start_x, "↑ : Rotate")
        stdscr.addstr(controls_y + 3, start_x, "↓ : Soft Drop")
        stdscr.addstr(controls_y + 4, start_x, "Space : Hard Drop")
        stdscr.addstr(controls_y + 5, start_x, "q : Quit")
        
        # Draw game over
        if self.game_over:
            game_over_y = start_y + board_height // 2
            stdscr.addstr(game_over_y, start_x, "GAME OVER")
            stdscr.addstr(game_over_y + 1, start_x, "Press 'q' to quit")
        
        stdscr.refresh()

    def run(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(100)
        
        while True:
            key = stdscr.getch()
            if key == ord('q'):
                break
            elif key == curses.KEY_LEFT:
                self.move_piece(-1, 0)
            elif key == curses.KEY_RIGHT:
                self.move_piece(1, 0)
            elif key == curses.KEY_DOWN:
                self.move_piece(0, 1)
            elif key == curses.KEY_UP:
                self.rotate_piece()
            elif key == ord(' '):
                self.hard_drop()
            
            self.update()
            self.render(stdscr)

def main(stdscr):
    game = TetrisGame()
    game.run(stdscr)

if __name__ == "__main__":
    curses.wrapper(main)