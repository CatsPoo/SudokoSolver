from SudokoSolver.Solver import SudokuSolver
from SudokoSolver.Scanner import SudokoScanner
from UserInterface.BoardEditor import BoardEditor
import os
from SudokoSolver.ExampleBoards import Boards

            

def main():
    scanner = SudokoScanner('./ExampleBoardsImages/IMG_9723.HEIC')
    board = scanner.get_board_from_image()
    
    #board=Boards[0]
    
    editor = BoardEditor(board)

    #edited_board = editor.edit_board()
    #edited_board = board


    #slv = SudokuSolver(edited_board)
    #solved_board = slv.Solve()
    #editor.setboard(edited_board)
    editor.print_board()

if __name__ == "__main__":
    main()