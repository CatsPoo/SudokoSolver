from ExampleBoards import Boards
from Solver import SudokuSolver

def GetBoard():
    return Boards[0]

def PrintBoard(board):
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n")
    for i,row in enumerate(board):
        line = ""
        if(i!=0 and i%3==0):
            print("--------------------------------")
        for j,col in enumerate(row):
            if(board[i][j] == 0):
                line = line + " _ "
            else:
                line = line + " " + str(board[i][j]) + " "
            if(j!=8 and (j+1)%3==0):
                line = line + " | "
        print(line)
            

def main():
    board = GetBoard()
    PrintBoard(board)
    slv = SudokuSolver(board)
    slv.Solve()
    PrintBoard(board)

if __name__ == "__main__":
    main()