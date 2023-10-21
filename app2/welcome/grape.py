import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import shutil
import time
import concurrent.futures
import sys
import functools
from pathlib import Path

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            clear_folder(file_path)  # 递归清空子文件夹
            os.rmdir(file_path)

def surround(yellow_mask, green_mask):
    kernel = np.ones((5, 5), np.uint8)
    dilated_yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=1)
    yellow_with_surrounding_green = cv2.bitwise_and(dilated_yellow_mask, green_mask)
    non_zero_pixels = np.count_nonzero(yellow_with_surrounding_green)
    return non_zero_pixels


def Harris(yellow_mask):
    max_corners = 100  # 最大角點數量
    quality_level = 0.1  # 角點質量閾值
    min_distance = 10  # 最小像素距離
    # 使用Shi-Tomasi角點檢測
    corners = cv2.goodFeaturesToTrack(yellow_mask, max_corners, quality_level, min_distance)
    num_corners = corners.shape[0]

    # 獲取角點數量
    # 将检测到的角点转换为整数坐标
    corners = np.int0(corners)
    yellow_mask_colored = cv2.cvtColor(yellow_mask * 255, cv2.COLOR_GRAY2BGR)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(yellow_mask_colored, (x, y), 3, (0, 0, 255), -1)
    return num_corners, yellow_mask_colored

def dilation(image):
    kernel = np.ones((5,5), np.uint8)
    # 进行膨胀操作
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return dilated_image

def erosion(image):
    #path = '/home/tzuiii/grape/m3000/20230513_15000_45000.jpg'
    #image = cv2.imread(path, 0)
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(image, kernel, iterations = 1)

    return erosion    
def opening(image):
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening_result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opening_result

def edge(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img,100,200)
    total_edges = cv2.countNonZero(edges)
    return total_edges

def process(region, total_pixels, yellow_mask):
    lower_green = np.array([36, 61, 82]) 
    upper_green = np.array([75, 186, 226])
    green_mask = cv2.inRange(region, lower_green, upper_green)

    green_pixels = cv2.countNonZero(green_mask)
    green_ratio = green_pixels / total_pixels

    ero_img = erosion(yellow_mask)
    ero_pixels = cv2.countNonZero(ero_img)+1
    total_edges = edge(yellow_mask)
    return green_ratio, ero_pixels, total_edges , green_mask

def calculate_yellow_ratio(y, x, img_name,  output_filename):

    # 將圖片轉換為HSV色彩空間
    #hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    """HSV range: [0~180, 0~255, 0~255]"""
    
    # 提取當前區域
    region = hsv_image[y:y+crop_size, x:x+crop_size]            
    
    
    # 定義黃色的HSV範圍
    lower_yellow = (23, 63, 203)
    upper_yellow = (36, 137, 255)

    # 依照HSV範圍篩選黃色像素
    yellow_mask = cv2.inRange(region, lower_yellow, upper_yellow)
    res = cv2.bitwise_and(region, region, mask=yellow_mask) 

    # 計算黃色像素的總數
    total_pixels = region.shape[0] * region.shape[1]
    yellow_pixels = cv2.countNonZero(yellow_mask)
    #print(yellow_pixels)
    # 計算黃色像素所佔比例
    yellow_ratio = yellow_pixels / total_pixels
    filename = f'{img_name}_{yellow_ratio}_{x}_{y}.png'
    #if filename == '20230513_23500_14100.png':
    #    cv2.imwrite('/home/tzuiii/grape/mask/01_03/'+filename, cv2.cvtColor(res, cv2.COLOR_HSV2BGR))
    #green_edges = edge(green_mask)
    if yellow_ratio>threshold:
        if(yellow_ratio > 0.0 and yellow_ratio < 0.01):
            try:
                with open(output_filename + Ptxt_path, 'a') as output_f:
                    output_f.write(str(yellow_ratio) + '_' + str(x) + '_' + str(y) + "\n")
                #print("Writing to file successful!")
            except Exception as e:
                print("Error writing to file:", e)
        elif(yellow_ratio >= 0.01 and yellow_ratio < 0.03):
            try:
                with open(output_filename + Ptxt_path, 'a') as output_f:
                    output_f.write(str(yellow_ratio) + '_' + str(x) + '_' + str(y) + "\n")
                #print("Writing to file successful!")
            except Exception as e:
                print("Error writing to file:", e)
        elif(yellow_ratio >= 0.03 and yellow_ratio < 0.05):
            #cv2.imwrite('/home/tzuiii/grape/new_range_crop4/mask/03_05/'+filename, cv2.cvtColor(res, cv2.COLOR_HSV2BGR))
            #cv2.imwrite('/home/tzuiii/grape/new_range_crop4/ori/03_05/'+filename, cv2.cvtColor(region, cv2.COLOR_HSV2BGR))
            dilated_img = opening(yellow_mask)
            green_ratio, ero_pixels, total_edges, green_mask = process(region, total_pixels, dilated_img)
            
            
            #cv2.imwrite('/home/tzuiii/grape/new_range_crop3/mask/03_05opening/'+filename, cv2.cvtColor(opening_img, cv2.COLOR_HSV2BGR))
            #corner, corner_img = Harris(dilated_img)
            num_labels, labels = cv2.connectedComponents(dilated_img) 
            surround_pixels = surround(yellow_mask, green_mask)
            #cv2.imwrite('/home/tzuiii/grape/new_range_crop3/mask/03_05cor/'+filename, corner_img)
            #print(filename +':yellow:'+str(yellow_ratio)+' green:'+ str(green_ratio) + "yellow edge:"+ str(total_edges)+"\n")   
            if green_ratio>0.3 and surround_pixels/num_labels>40: 
                #cv2.imwrite(output_filename+'positive_image/'+filename, cv2.cvtColor(region, cv2.COLOR_HSV2BGR))
                try:
                    with open(output_filename + Ptxt_path, 'a') as output_f:
                        output_f.write(str(yellow_ratio) + '_' + str(x) + '_' + str(y) + "\n")
                    #print("Writing to file successful!")
                except Exception as e:
                    print("Error writing to file:", e)
            else:
                #cv2.imwrite(output_filename+'negative_image/'+filename, cv2.cvtColor(region, cv2.COLOR_HSV2BGR))
                try:
                    with open(output_filename + Ntxt_path, 'a') as output_f:
                        output_f.write(str(yellow_ratio) + '_' + str(x) + '_' + str(y) + "\n")
                    #print("Writing to file successful!")
                except Exception as e:
                    print("Error writing to file:", e)
        elif(yellow_ratio >= 0.05 and yellow_ratio < 0.1):
            green_ratio, ero_pixels, total_edges , green_mask= process(region, total_pixels, yellow_mask)
            #cv2.imwrite('/home/tzuiii/grape/new_range_crop3/mask/05_10/'+filename, cv2.cvtColor(res, cv2.COLOR_HSV2BGR))
            #cv2.imwrite('/home/tzuiii/grape/new_range_crop3/ori/05_10/'+filename, cv2.cvtColor(region, cv2.COLOR_HSV2BGR))
            if (total_edges/ero_pixels < 200) and green_ratio >0.05:
                #cv2.imwrite(output_filename+'positive_image/'+filename, cv2.cvtColor(region, cv2.COLOR_HSV2BGR))
                try:
                    with open(output_filename + Ptxt_path, 'a') as output_f:
                        output_f.write(str(yellow_ratio) + '_' + str(x) + '_' + str(y) + "\n")
                    #print("Writing to file successful!")
                except Exception as e:
                    print("Error writing to file:", e)
            else:
                #cv2.imwrite(output_filename+'negative_image/'+filename, cv2.cvtColor(region, cv2.COLOR_HSV2BGR))
                try:
                    with open(output_filename + Ntxt_path, 'a') as output_f:
                        output_f.write(str(yellow_ratio) + '_' + str(x) + '_' + str(y) + "\n")
                    #print("Writing to file successful!")
                except Exception as e:
                    print("Error writing to file:", e)
        elif (yellow_ratio >= 0.1 and yellow_ratio < 0.15):
            green_ratio, ero_pixels, total_edges , green_mask= process(region, total_pixels, yellow_mask)
            #cv2.imwrite('/home/tzuiii/grape/new_range_crop3/mask/10_15/'+filename, cv2.cvtColor(res, cv2.COLOR_HSV2BGR))
            #cv2.imwrite('/home/tzuiii/grape/new_range_crop3/ori/10_15/'+filename, cv2.cvtColor(region, cv2.COLOR_HSV2BGR))
            if (total_edges/ero_pixels < 50) and green_ratio >0.05:

                #cv2.imwrite(output_filename+'positive_image/'+filename, cv2.cvtColor(region, cv2.COLOR_HSV2BGR))
                try:
                    with open(output_filename + Ptxt_path, 'a') as output_f:
                        output_f.write(str(yellow_ratio) + '_' + str(x) + '_' + str(y) + "\n")
                    #print("Writing to file successful!")
                except Exception as e:
                    print("Error writing to file:", e)
            else:
                #cv2.imwrite(output_filename+'negative_image/'+filename, cv2.cvtColor(region, cv2.COLOR_HSV2BGR))
                try:
                    with open(output_filename + Ntxt_path, 'a') as output_f:
                       output_f.write(str(yellow_ratio) + '_' + str(x) + '_' + str(y) + "\n")
                    #print("Writing to file successful!")
                except Exception as e:
                    print("Error writing to file:", e)
        elif (yellow_ratio >= 0.15 and yellow_ratio < 0.2):
            green_ratio, ero_pixels, total_edges , green_mask= process(region, total_pixels, yellow_mask)
            #cv2.imwrite('/home/tzuiii/grape/new_range_crop3/mask/15_20/'+filename, cv2.cvtColor(res, cv2.COLOR_HSV2BGR))
            #cv2.imwrite('/home/tzuiii/grape/new_range_crop3/ori/15_20/'+filename, cv2.cvtColor(region, cv2.COLOR_HSV2BGR))
            if (total_edges/ero_pixels < 15) and green_ratio >0.05:
                #cv2.imwrite(output_filename+'positive_image/'+filename, cv2.cvtColor(region, cv2.COLOR_HSV2BGR))
                try:
                    with open(output_filename + Ptxt_path, 'a') as output_f:
                        output_f.write(str(yellow_ratio) + '_' + str(x) + '_' + str(y) + "\n")
                    #print("Writing to file successful!")
                except Exception as e:
                    print("Error writing to file:", e)
            else:
                #cv2.imwrite(output_filename+'negative_image/'+filename, cv2.cvtColor(region, cv2.COLOR_HSV2BGR))
                try:
                    with open(output_filename + Ntxt_path, 'a') as output_f:
                        output_f.write(str(yellow_ratio) + '_' + str(x) + '_' + str(y) + "\n")
                    #print("Writing to file successful!")
                except Exception as e:
                    print("Error writing to file:", e)
        elif (yellow_ratio >= 0.2 and yellow_ratio < 0.25):
            green_ratio, ero_pixels, total_edges , green_mask= process(region, total_pixels, yellow_mask)
            #cv2.imwrite('/home/tzuiii/grape/new_range_crop3/mask/20_25/'+filename, cv2.cvtColor(res, cv2.COLOR_HSV2BGR))
            #cv2.imwrite('/home/tzuiii/grape/new_range_crop3/ori/25_30/'+filename, cv2.cvtColor(region, cv2.COLOR_HSV2BGR))
            if (total_edges/ero_pixels < 5) and green_ratio >0.05:
                #cv2.imwrite(output_filename+'positive_image/'+filename, cv2.cvtColor(region, cv2.COLOR_HSV2BGR))
                try:
                    with open(output_filename + Ptxt_path, 'a') as output_f:
                        output_f.write(str(yellow_ratio) + '_' + str(x) + '_' + str(y) + "\n")
                    #print("Writing to file successful!")
                except Exception as e:
                    print("Error writing to file:", e)
            else:
                #cv2.imwrite(output_filename+'negative_image/'+filename, cv2.cvtColor(region, cv2.COLOR_HSV2BGR))
                try:
                    with open(output_filename + Ntxt_path, 'a') as output_f:
                        output_f.write(str(yellow_ratio) + '_' + str(x) + '_' + str(y) + "\n")
                    #print("Writing to file successful!")
                except Exception as e:
                    print("Error writing to file:", e)
        elif (yellow_ratio >= 0.25):
            green_ratio, ero_pixels, total_edges , green_mask= process(region, total_pixels, yellow_mask)
            #cv2.imwrite('/home/tzuiii/grape/new_range_crop3/mask/30/'+filename, cv2.cvtColor(res, cv2.COLOR_HSV2BGR))
            #cv2.imwrite('/home/tzuiii/grape/new_range_crop3/ori/30/'+filename, cv2.cvtColor(region, cv2.COLOR_HSV2BGR))
            if  (total_edges/ero_pixels < 5) and green_ratio >0.05:
                #cv2.imwrite(output_filename+'positive_image/'+filename, cv2.cvtColor(region, cv2.COLOR_HSV2BGR))
                try:
                    with open(output_filename + Ptxt_path, 'a') as output_f:
                        output_f.write(str(yellow_ratio) + '_' + str(x) + '_' + str(y) + "\n")
                    #print("Writing to file successful!")
                except Exception as e:
                    print("Error writing to file:", e)
            else:
                #cv2.imwrite(output_filename+'negative_image/'+filename, cv2.cvtColor(region, cv2.COLOR_HSV2BGR))
                try:
                    with open(output_filename + Ntxt_path, 'a') as output_f:
                        output_f.write(str(yellow_ratio) + '_' + str(x) + '_' + str(y) + "\n")
                    #print("Writing to file successful!")
                except Exception as e:
                    print("Error writing to file:", e)
        # 將結果寫入對應的txt檔'''
    

    

            
# 參數設置
# 检查命令行参数的数量
if len(sys.argv) != 4:
    print("Usage: python python_script.py input.txt output.txt")
    sys.exit(1)

# 获取输入文件名和输出文件名
img_path = sys.argv[1]
output_filename = sys.argv[2]
threshold = float(sys.argv[3])
#img_path = "/home/tzuiii/grape/20230513.png"
# 取得檔名部分
file_name = os.path.basename(img_path)
# 移除副檔名
img_name = os.path.splitext(file_name)[0] # 20230513

#threshold = 0.25  

green_threshold = 0.2
erosion_threshold = 0.05
 
crop_size = 100
#txt_dir = "/home/tzuiii/grape/new_range_crop1/"

Ptxt_path = "Positive.txt"
Ntxt_path = "Negative.txt"
Gtxt_path = "FP.txt"
NGtxt_path = "FN.txt"


 
open(output_filename+Ptxt_path, 'w').close()
open(output_filename+Ntxt_path, 'w').close()
#open(output_filename+Gtxt_path, 'w').close()
#open(output_filename+NGtxt_path, 'w').close()
'''open(output_filename+Etxt_path, 'w').close()
open(output_filename+NEtxt_path, 'w').close()'''
'''positivefolder_to_clear = output_filename+"positive_image/"
positivefolder_path = Path(positivefolder_to_clear)
negativefolder_to_clear = output_filename+"negative_image/"
negativefolder_path = Path(negativefolder_to_clear)
if not positivefolder_path.exists():
    positivefolder_path.mkdir()  # 创建文件夹
    print(f"資料夹 '{positivefolder_to_clear}' 創建成功")
else:
    print(f"資料夹 '{positivefolder_to_clear}' 已存在")
    # 调用函数来清空文件夹
    clear_folder(positivefolder_to_clear)
if not negativefolder_path.exists():
    negativefolder_path.mkdir()  # 创建文件夹
    print(f"資料夹 '{negativefolder_to_clear}' 創建成功")
else:
    print(f"資料夹 '{negativefolder_to_clear}' 已存在")
    # 调用函数来清空文件夹
    clear_folder(negativefolder_to_clear)'''


# 讀取大圖
print("Reading image ~~~")
start = time.time()
image = cv2.imread(img_path)
end1 = time.time()
print("Reading image time: %f sec" % (end1 - start))


# 將圖片轉換為HSV色彩空間
start2 = time.time()
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
end2 = time.time()
print("BGR2HSV time: %f sec" % (end2 - start2))
    

# 計算圖像的寬度和高度
height, width = hsv_image.shape[:2]

start3 = time.time()
with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:  # 設置max_workers=CPU的Threads(其他值都會變慢)
    I_list = list(range(0, height+1, crop_size))*((width//crop_size)+1)
    J_list = [x for x in list(range(0, width+1, crop_size)) for _ in range((height//crop_size)+1)]
            
    partial_calculate_yellow_ratio = functools.partial(
        calculate_yellow_ratio,
        img_name=img_name,
        output_filename=output_filename
    )
    
    executor.map(partial_calculate_yellow_ratio, I_list, J_list, chunksize=1000)

end3 = time.time()
print("concurrent time: %f sec" % (end3 - start3))
"""[12, 100] = 2.04sec, [12, 200] = 1.99sec, [12, 300] = 1.98sec, [12, 500] = 1.93sec, [12, 1000] = 1.85sec"""
#cv2.imwrite('retan_image.png', image)
end = time.time()
print("==========Processing time: %f sec==========" % (end - start))
import print2
print2.print_image(output_filename, Ptxt_path, height, width, threshold)