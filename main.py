
from pathlib import Path

from cchess import ChessBoard
from cchess_board import ChessboardDetector

def main():
    detector = ChessboardDetector('models\\detector1\\')
    folder = Path('tests\\boards\\')
    files = list(folder.glob('*.png')) + list(folder.glob('*.jpg'))
    for file_name in files:
        mark_file = Path(file_name.parent, f'mark_{file_name.name}')
        ok = detector.img_board_mark_to(file_name, mark_file)
        print(ok, file_name)    
        
    return        
    
    
if __name__ == "__main__":
    main()