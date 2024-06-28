from SudokoSolver.Scanner import SudokoScanner
class BoardImageEditor:
    
    def __init__(self,boardImage) -> None:
        self.boardImage = boardImage
    
    def print_solution(self,static_digits,solution):
        scanner = SudokoScanner(self.boardImage)
        cells_centroids = scanner.get_board_crosses_centroids()
        for i in range(9):
            for j in range(9):
                pass