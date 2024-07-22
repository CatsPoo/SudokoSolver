from SudokoSolver.Solver import SudokuSolver
from SudokoSolver.Scanner import SudokoScanner
from UserInterface.BoardEditor import BoardEditor
import cv2
from SudokoSolver.ExampleBoards import Boards
from UserInterface.BoardImageEditor import BoardImageEditor
            

def main():
    scanner = SudokoScanner('./ExampleBoardsImages/IMG_9723.HEIC')
    scanner1 = SudokoScanner('./ExampleBoardsImages/images.jpeg')
    board = scanner.get_board_from_image()
    #board = scanner1.get_board_from_image()
    #board=Boards[0]
    editor = BoardEditor(board)

    #edited_board = editor.edit_board()
    editor.print_board()
    #edited_board = board

    slv = SudokuSolver(edited_board)
    solved_board = slv.Solve()

    editor.setboard(solved_board)
    editor.print_board()

    imageEditor = BoardImageEditor(scanner.get_image())
    solved_board_image = imageEditor.get_solved_board(edited_board,solved_board)
    cv2.imshow('solved',solved_board_image)
    cv2.waitKey(0)



if __name__ == "__main__":
    main()