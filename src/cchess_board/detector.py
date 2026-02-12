
import time
from pathlib import Path

import numpy as np
import cv2

from typing import List, Tuple, Union

from .rtmpose import RTMPOSE_ONNX
from .classifier import CLASSIFIER_ONNX

from cchess import ChessBoard

#-----------------------------------------------------#
def labels_to_fen(cell_labels):
        board = ChessBoard()
        is_flip = False
        label_line = cell_labels.split('\n')
        for row, line in enumerate(label_line):
            for col, fench in enumerate(line): 
                if fench == '.':
                    continue
                else:    
                    board.put_fench(fench, (8-col, row))
                    if (fench == 'K') and (row >= 6):
                        is_flip = True    
        if is_flip:
            b = board.flip()
            board = b.mirror()
        
        return board.to_fen()
        
    
#-----------------------------------------------------#
BONE_NAMES = [
    "A0", "A8",
    "J0", "J8",
]

#
#-----------------------------------------------------#
def check_keypoints(keypoints: np.ndarray):
    """
    检查关键点坐标是否正确
    @param keypoints: 关键点坐标, shape 为 (N, 2)
    """
    if keypoints.shape != (len(BONE_NAMES), 2):
        raise Exception(f"keypoints shape error: {keypoints.shape}")

#-----------------------------------------------------#
def perspective_transform(
        image: cv2.UMat, 
        src_points: np.ndarray, 
        keypoints: np.ndarray,
        dst_size=(450, 500)) -> Tuple[cv2.UMat, np.ndarray, np.ndarray]:
    """
    透视变换
    @param image: 图片
    @param src_points: 源点坐标
    @param keypoints: 关键点坐标
    @param dst_size: 目标尺寸 (width, height) 10 行 9 列

    @return:
        result: 透视变换后的图片
        transformed_keypoints: 透视变换后的关键点坐标
        corner_points: 棋盘的 corner 点坐标, shape 为 (4, 2) A0, A8, J0, J8
    """

    check_keypoints(keypoints)


    # 源点和目标点
    src = np.float32(src_points)
    padding = 50
    corner_points = np.float32([
        # 左上角
        [padding, padding], 
        # 右上角
        [dst_size[0]-padding, padding], 
        # 左下角
        [padding, dst_size[1]-padding], 
        # 右下角
        [dst_size[0]-padding, dst_size[1]-padding]])

    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(src, corner_points)

    # 执行透视变换
    result = cv2.warpPerspective(image, matrix, dst_size)

    # 重塑数组为要求的格式 (N,1,2)
    keypoints_reshaped = keypoints.reshape(-1, 1, 2).astype(np.float32)
    transformed_keypoints = cv2.perspectiveTransform(keypoints_reshaped, matrix)
    # 转回原来的形状
    transformed_keypoints = transformed_keypoints.reshape(-1, 2)

    return result, transformed_keypoints, corner_points

#-----------------------------------------------------#
def get_board_corner_points(keypoints: np.ndarray) -> np.ndarray:
    """
    计算棋局四个边角的 points
    @param keypoints: 关键点坐标, shape 为 (N, 2)
    @return: 边角的坐标, shape 为 (4, 2)
    """
    check_keypoints(keypoints)

    # 找到 A0 A8 J0 J8 的坐标 以及 A4 和 J4 的坐标
    a0_index = BONE_NAMES.index("A0")
    a8_index = BONE_NAMES.index("A8")
    j0_index = BONE_NAMES.index("J0")
    j8_index = BONE_NAMES.index("J8")

    a0_xy = keypoints[a0_index]
    a8_xy = keypoints[a8_index]
    j0_xy = keypoints[j0_index]
    j8_xy = keypoints[j8_index]
    
    '''
    #print(a0_xy[0], j0_xy[0])
    #print(a8_xy[0], j8_xy[0])
    '''
    min_x = min(a0_xy[0], a8_xy[0], j0_xy[0], j8_xy[0])
    min_y = min(a0_xy[1], a8_xy[1], j0_xy[1], j8_xy[1])
    
    max_x = max(a0_xy[0], a8_xy[0], j0_xy[0], j8_xy[0])
    max_y = max(a0_xy[1], a8_xy[1], j0_xy[1], j8_xy[1])
    
    a0_xy=(min_x, min_y)
    a8_xy=(max_x, min_y)
    j0_xy=(min_x, max_y)
    j8_xy=(max_x, max_y)
    
    # 计算新的四个角点坐标
    dst_points = np.array([
        a0_xy,
        a8_xy,
        j0_xy,
        j8_xy
    ], dtype=np.float32)

    return dst_points

def corner_points_to_rect(corner_points):
    
    a0_xy = corner_points[0]
    a8_xy = corner_points[1]
    j0_xy = corner_points[2]
    j8_xy = corner_points[3]
    
    '''
    #print(a0_xy[0], j0_xy[0])
    #print(a8_xy[0], j8_xy[0])
    '''
    min_x = min(a0_xy[0], a8_xy[0], j0_xy[0], j8_xy[0])
    min_y = min(a0_xy[1], a8_xy[1], j0_xy[1], j8_xy[1])
    
    max_x = max(a0_xy[0], a8_xy[0], j0_xy[0], j8_xy[0])
    max_y = max(a0_xy[1], a8_xy[1], j0_xy[1], j8_xy[1])
    
    grid_x_half = (max_x - min_x) /8
    min_x = int(max(0, min_x - grid_x_half))
    max_x = int(max(0, max_x + grid_x_half))
    
    grid_y_half = (max_y - min_y) /9
    min_y = int(max(0, min_y - grid_y_half))
    max_y = int(max(0, max_y + grid_y_half))
    
    return ((min_x, min_y), (max_x, max_y))
   
#-----------------------------------------------------#
def extract_chessboard(img_bgr: cv2.UMat, keypoints: np.ndarray) -> Tuple[cv2.UMat, np.ndarray, np.ndarray]:
    """
    提取棋盘信息
    @param img: 图片
    @param keypoints: 关键点坐标, shape 为 (N, 2)
    @return:
        transformed_image: 透视变换后的图片
        transformed_keypoints: 透视变换后的关键点坐标
        transformed_corner_points: 棋盘的 corner 点坐标, shape 为 (4, 2) A0, A8, J0, J8
    """
   
    check_keypoints(keypoints)
    source_corner_points = get_board_corner_points(keypoints)
    transformed_image, transformed_keypoints, transformed_corner_points = perspective_transform(img_bgr, source_corner_points, keypoints)
 
    return transformed_image, transformed_keypoints, transformed_corner_points
   
   
#-----------------------------------------------------#
class ChessboardDetector:
    def __init__(self, model_path: str):
        
        self.pose = RTMPOSE_ONNX(Path(model_path, 'pose.onnx'))
        self.classifier = CLASSIFIER_ONNX(Path(model_path, 'layout.onnx'))

        self.board_positions = []  # 存储棋盘位置坐标
        self.current_image = None
        self.current_filename = None
    

    # 检测中国象棋棋盘
    def pred_keypoints(self, image_bgr: Union[np.ndarray, None] = None) -> Tuple[List[List[int]], List[float]]:

        # 预测关键点, 绘制关键点

        width, height = image_bgr.shape[:2]
        bbox = [0, 0, width, height]

        keypoints, scores = self.pose.pred(image=image_bgr, bbox=bbox)
        return keypoints, scores

    def get_board_rect(self, img: cv2.UMat):
        """
        提取棋盘方框位置
        """
        keypoints, _scores = self.pred_keypoints(img)
        check_keypoints(keypoints)
        corner_points = get_board_corner_points(keypoints) 
        board_rect = corner_points_to_rect(corner_points)
        cropped_image = img[board_rect[0][1]:board_rect[1][1], board_rect[0][0]:board_rect[1][0]]
        
        return board_rect, cropped_image
                        
    # 在裁剪的棋盘图片上预测
    def img_to_labels(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, List[List[str]], List[List[float]]]:
        
        # 预测每个位置的 棋子类别
        _, _, scores, cell_labels = self.classifier.pred(image_bgr)
        return cell_labels
        
        #fen = labels_to_fen(cell_labels)    
        #return cell_labels, fen    
        
    '''    
    def draw_pred_with_keypoints(self, image_rgb: Union[np.ndarray, None] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if image_rgb is None:
            return None, None, None
        
        image_rgb = image_rgb.copy()

        original_image = image_rgb.copy()

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        keypoints, scores = self.pred_keypoints(image_bgr)

        # 绘制棋盘框架
        draw_image = self.pose.draw_pred(img=image_rgb, keypoints=keypoints, scores=scores)

        # 融合 self.pose.bone_names 与 keypoints, 再转换成 DataFrame
        keypoint_list = []
        for bone_name, keypoint in zip(self.pose.bone_names, keypoints):
            keypoint_list.append({"name": bone_name, "x": keypoint[0], "y": keypoint[1]})

        keypoint_df = DataFrame(keypoint_list)

        return draw_image, original_image, keypoint_df
    '''

    # 检测棋盘 detect board
    def pred_detect_board_and_classifier(self, image_bgr: np.ndarray, 
                                        ) -> Tuple[np.ndarray, np.ndarray, str, List[List[float]], str]:

        """
        @param image_bgr: 输入的 BGR 图像
        @return: 
            - transformed_image_layout  # 拉伸棋盘
            - original_image_with_keypoints  # 原图关键点
            - layout_pred_info  # 每个位置的 棋子类别
            - scores  # 每个位置的 置信度
            - time_info  # 推理用时
        """

        keypoints, scores = self.pred_keypoints(image_bgr)

        # 提取棋盘, 绘制 每个位置的 范围信息
        transformed_image, _transformed_keypoints, _corner_points = extract_chessboard(image_bgr, keypoints)

        # 预测每个位置的 棋子类别
        _, _, scores, cell_labels = self.classifier.pred(transformed_image)
        
        return transformed_image, cell_labels
    
    def cv_image_to_fen(self, img_bgr):
        #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed_image, cell_labels = self.pred_detect_board_and_classifier(img_bgr)
        if not cell_labels:
            return '', ''
        
        fen = labels_to_fen(cell_labels)    
        return cell_labels, fen
                    
    # 在裁剪拉伸后的棋盘图片上预测
    def transformed_img_to_fen(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, List[List[str]], List[List[float]]]:
        
        # 预测每个位置的 棋子类别
        _, _, scores, cell_labels = self.classifier.pred(image_bgr)

        fen = labels_to_fen(cell_labels)    
        return cell_labels, fen
        
    # 检测棋盘 detect board
    def img_to_board(self, image_file):
        file_path = str(image_file).encode('gbk')
        img = cv2.imread(file_path)
        return self.pred_detect_board_and_classifier(img)
                
    def img_board_mark_to(self, image_file, mark_file):
        file_path = str(image_file).encode('gbk')
        img = cv2.imread(file_path)
        board_rect, cropped_image = self.get_board_rect(img)
       
        if board_rect is None:
            return False
        
        labels = self.img_to_labels(cropped_image)
        print(labels)
        cv2.imwrite(str(mark_file).encode('GBK'), cropped_image)
        return True
        
        #return self.save_img_board_rect(cut_img, rect, mark_file)