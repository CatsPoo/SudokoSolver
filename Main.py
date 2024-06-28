from SudokoSolver.Solver import SudokuSolver
from SudokoSolver.Scanner import SudokoScanner
from UserInterface.BoardEditor import BoardEditor
import os

            

def main():
    scanner = SudokoScanner('./ExampleBoardsImages/IMG_9723.HEIC')
    board = scanner.get_board_from_image()
    editor = BoardEditor(board)

    board = editor.edit_board()
    editor.print_board()
    # board = GetBoard()
    #PrintBoard(board)
    # slv = SudokuSolver(board)
    # slv.Solve()
    # PrintBoard(board)

if __name__ == "__main__":
    main()