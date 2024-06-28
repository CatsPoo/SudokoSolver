
import curses

class BoardEditor:

    def __init__(self,board) -> None:
        self.board = board

    def print_board(self):
        print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n")
        for i,row in enumerate(self.board):
            line = ""
            if(i!=0 and i%3==0):
                print("--------------------------------")
            for j,col in enumerate(row):
                if(self.board[i][j] == 0):
                    line = line + " _ "
                else:
                    line = line + " " + str(self.board[i][j]) + " "
                if(j!=8 and (j+1)%3==0):
                    line = line + " | "
            print(line)

    
    def draw_board(self, stdscr, current_row=-1, current_col=-1):
        stdscr.clear()
        max_y, max_x = stdscr.getmaxyx()
        if max_y < 18 or max_x < 36:
            stdscr.addstr(0, 0, "Window too small")
        else:
            for row in range(9):
                if row != 0 and row % 3 == 0:
                    stdscr.addstr(row * 2 - 1, 0, '-' * 37)
                for col in range(9):
                    if col != 0 and col % 3 == 0:
                        stdscr.addstr(row * 2, col * 4 - 1, '|')
                    if row == current_row and col == current_col:
                        stdscr.attron(curses.A_REVERSE)
                    stdscr.addstr(row * 2, col * 4, str(self.board[row][col]) if self.board[row][col] != 0 else '.')
                    stdscr.attroff(curses.A_REVERSE)
        stdscr.refresh()

    def _edit_board(self, stdscr):
        curses.curs_set(0)
        current_row, current_col = 0, 0

        while True:
            self.draw_board(stdscr, current_row, current_col)
            key = stdscr.getch()

            if key == curses.KEY_UP:
                current_row = (current_row - 1) % 9
            elif key == curses.KEY_DOWN:
                current_row = (current_row + 1) % 9
            elif key == curses.KEY_LEFT:
                current_col = (current_col - 1) % 9
            elif key == curses.KEY_RIGHT:
                current_col = (current_col + 1) % 9
            elif key in range(ord('0'), ord('9') + 1):
                self.board[current_row][current_col] = int(chr(key))
            elif key == ord('e'):
                return self.board

    def edit_board(self):
        curses.wrapper(self._edit_board)