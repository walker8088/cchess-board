#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
from pathlib import Path
import json

import requests

from cchess import ChessBoard

# ===================== 配置 =====================
# 根据你的实际服务修改下面这些
BASE_URL = "http://127.0.0.1:8000" 
#BASE_URL = "https://123.56.244.10/board_server"
#BASE_URL = "https://www.wfmrwh.com/board_server"
 
 
API_ENDPOINT = "/recognize"
TOKEN = "valid_token"                       # 替换成你实际有效的 token

# 支持的图片扩展名
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

def is_valid_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS


def upload_image(image_path: str, timeout: int = 120) -> dict:
    """上传单张图片并获取结果"""
    url = f"{BASE_URL}{API_ENDPOINT}"
    headers = {
        #"Authorization": f"Bearer {token}"
    }
    
    if not is_valid_image(Path(image_path)):
        return {"error": f"不支持的文件格式或文件不存在: {image_path}"}

    try:
        start_time = time.time()
        
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(
                url,
                headers=headers,
                files=files,
                #verify=False,
                timeout=timeout
            )
        
        elapsed = time.time() - start_time
        return response.json()
        
    except requests.exceptions.Timeout:
        return {"status": "timeout", "message": f"超过 {timeout} 秒未响应"}
    except requests.exceptions.RequestException as e:
        return {"status": "request_error", "message": str(e)}

test_data = [
    '57p.jpg 1rbakr3/4a4/2n1b2c1/p1p1p3p/1c3np2/2P6/P3P1P1P/2N1C1C1N/1R3R3/2BAKAB2 w',
    '507_82.jpg 1rbakabnr/9/c1n4c1/p3p3p/2p3p2/5R3/P1P1P1P1P/2NC2NC1/9/2BAKAB1R w',
    'demo003.png rnbakabnr/9/1c7/p1p1p1p1p/9/9/P1P1P1P1P/RC5CR/9/1NBAKABc1 w',
    'qizhe.jpg 3R3c1/7C1/4kN3/1rp4N1/2b5n/9/9/9/4p1p2/1c1rnK3 w',
    'wangzhe.jpg 2bakab1r/5r3/2n1c1n2/p3p1p1p/2p6/5NP2/P1P1P2cP/2N1C3C/9/1RBAKAB1R w',
    'tiantian1.jpg 2b1ka1RC/4a4/4c3b/p1p1p3p/5N3/4c4/P1P5P/N2r5/9/4KA3 w',
    'tiantian2.jpg 1r2kab2/4a1c2/1c2b1C1r/pR2C3p/2p6/4p4/P1P3P1P/2N6/9/2BAKABNR w',
    'tiantian3.jpg 2r1kabr1/4a3n/4b4/p5P1p/6n2/1R7/P3P2cP/2N1B1C1N/9/2BAKA1R1 w',
    'tiantian4.jpg 2bakab2/8r/2n1c1c1n/p1p1p1p1p/7R1/1r4P2/P1P1P3P/1CN1C1N1B/9/1RBAKA3 w',
    'tiantian5.png rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w',
    'tiantian6.jpg 2baka3/9/4b1C2/pR2p1p1p/5c3/2r6/P5P1P/3nC4/5K3/2BA2B2 w',
    'tiantian7.jpg 2baka3/9/4b1C2/pR2p1p1p/5c3/2r6/P5P1P/3nC4/5K3/2BA2B2 w',
    'tiantian8.jpg 1rbaka3/9/2n1b4/p4c1rp/1c2p4/2N3R2/P7P/4C3C/3R5/2BAKAB2 w',
    'tiantian9.jpg 2bak1b2/2c1a2c1/8r/p3N2Cp/4p1p2/1r7/Pn1R4P/B3C4/9/3AKABNR w',
    'shuihu1.jpg 5ab2/5k3/3ab4/8p/n3C4/5p3/4P3P/B2A5/3CK4/3r5 w',
    'shuihu2.jpg 3akab2/2r6/2n1b1n2/p1N1p3p/6p2/2P1P4/cR4P1P/4C3N/4A4/4KAB2 w',
    'zhang1.jpg 2b1ka3/C3a4/8R/2p1p1p1p/p2r5/3n5/P3P3P/4B4/9/3AKAB2 w',
]
    
def main():
    
    print(f"服务器: {BASE_URL}")
    
    images_to_process = []
    
    success_count = 0
    busy_count = 0
    error_count = 0
    
    for i, path_fen in enumerate(test_data, 1):
        file, fen, _ = path_fen.split(' ')
        img_path = Path('tests', 'boards', file)
        print(f"\n[{i}/{len(test_data)}] {img_path}")
            
        result = upload_image(img_path)
        status = result.get("status")
        elapsed = result.get("elapsed", 0)
        
        if status == "ok":
            print(f"成功 ({elapsed:.2f}s) → \n{result.get('fen')}")
            b = ChessBoard(result.get('fen'))
            b = b.mirror()
            new_fen =  b.to_fen().split(' ')[0]
            #print(Path(img_path).name, b.to_fen())
            if new_fen == fen:
                success_count += 1
                print('成功。')
            else:
                error_count += 1
                print('失败。')
                b.print_board()
                print(Path(img_path).name, b.to_fen())
                
        elif status == "busy":
            print(f"忙碌 ({elapsed:.2f}s) → {result.get('message')}")
            busy_count += 1
        elif status == "error" or status == "http_error" or status == "request_error":
            print(f"失败 ({elapsed:.2f}s) → {result.get('error') or result.get('message') or result.get('text')}")
            error_count += 1
        else:
            print(f"未知响应 ({elapsed:.2f}s) → {result}")
        
            #time.sleep(2)
    
    #time.sleep(5)
    print("\n" + "=" * 60)
    print("测试完成统计：")
    print(f"  成功: {success_count}")
    print(f"  服务器忙: {busy_count}")
    print(f"  失败/错误: {error_count}")
    print(f"  总计: {success_count + busy_count + error_count}")


if __name__ == "__main__":
    main()