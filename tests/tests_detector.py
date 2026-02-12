
from pathlib import Path

from cchess import ChessBoard
from cchess_detector import ChessboardDetector

def main():
    detector = ChessboardDetector('models\\detector_0\\')
    folder = Path('tests\\boards\\')
    files = list(folder.glob('*.png')) + list(folder.glob('*.jpg'))
    for file_name in files:
        mark_file = Path(file_name.parent, f'mark_{file_name.name}')
        yes = detector.img_board_mark_to(file_name, mark_file)
        print(yes)
    return        
    #org_img, trans_img, labels, scores, time_info = detector.img_to_board('tests\\boards\\demo003.png')
    org_img, trans_img, labels, scores, time_info = detector.img_to_board('tests\\boards\\tiantian5.png')
    #org_img, trans_img, labels, scores, time_info = detector.img_to_board('tests\\boards\\p2.png')
    
    print(len(labels))
    print(labels)
    '''
    board = ChessBoard()
    label_line = labels.split('\n')
    for row, line in enumerate(label_line):
        for col, fench in enumerate(line): 
            if fench == '.':
                continue
            else:
                board.put_fench(fench, (8-col, row))    
    
    board.print_board()        
    '''
    
if __name__ == "__main__":
    main()
