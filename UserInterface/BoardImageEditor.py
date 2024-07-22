from SudokoSolver.Scanner import SudokoScanner
import cv2
class BoardImageEditor:
    
    def __init__(self,boardImage) -> None:
        self.boardImage = boardImage
    
    def get_solved_board(self,static_digits,solution):
        scanner = SudokoScanner(self.boardImage)
        centroids_list = scanner._get_board_crosses_centroids()

        for i in range(9):
            for j in range(9):
                if(static_digits[i][j]!=0):
                    continue

                cellImageCorners = [
                    centroids_list[i][j],
                    centroids_list[i][j+1],
                    centroids_list[i+1][j],
                    centroids_list[i+1][j+1],
                ]

                cell_width = cellImageCorners[3][0] - cellImageCorners[2][0]
                test_location = cellImageCorners[2]
                test_location = (test_location[0]+ (cell_width//3) ,test_location[1])


                cv2.putText(self.boardImage,str(solution[i][j]), test_location, cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        return self.boardImage