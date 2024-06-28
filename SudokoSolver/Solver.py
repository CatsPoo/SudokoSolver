import copy
class SudokuSolver:
    def __init__(self,board):
        self.board=copy.deepcopy(board)
        self.constantMap=[]
        self.InitConstantMap()          

    def InitConstantMap(self):
        for i,row in enumerate(self.board):
            self.constantMap.append([])
            for j,col in enumerate(row):
                self.constantMap[i].append(self.board[i][j]!=0)


    def IsRowAvailable(self,row,val):
        return not (val in self.board[row])

    def IsColAvailable(self,col,val):
        for row in self.board:
            if(row[col]==val):
                return False
        return True

    def isSquareAvailavle(self,row,col,val):
        startRow = (int)(row/3)*3
        startCol = (int)(col/3)*3

        for i in range(startRow,startRow+3):
            for j in range(startCol, startCol+3):
                if(self.board[i][j]==val):
                    return False
        return True

    def IsAvailable(self,row,col,val):
        return (self.IsRowAvailable(row,val) and self.IsColAvailable(col,val) and self.isSquareAvailavle(row,col,val))

    def RecursiveSolve(self,row,col,val):

        if(row > 8 ):
            return True
        nextRow= row + (int)((col+1)/9)
        nextCol= (col+1)%9

        if(self.constantMap[row][col]):
            return self.RecursiveSolve(nextRow,nextCol,val)

        else:
            f=0 
            for i in range(val,10):
                if(self.IsAvailable(row,col,i)):
                    self.board[row][col] = i
                    f=1
                    break
            if(f==0):
                val =0
                self.board[row][col]=val
                return False
        if(row==8 and col ==8):
            return True
        if(self.RecursiveSolve(nextRow,nextCol,1)):
            return True
    
        val = val + 1
        if(val>9):
            val = 0
            self.board[row][col] = val
            return False
        else:
            return self.RecursiveSolve(row,col,val)
    

    def Solve(self):
        self.RecursiveSolve(0,0,1)
        return self.board