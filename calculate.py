import os, random, shutil
from PIL import Image
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import csv
 
def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    m = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        if g >= b:
            h = ((g-b)/m)*60
        else:
            h = ((g-b)/m)*60 + 360
    elif mx == g:
        h = ((b-r)/m)*60 + 120
    elif mx == b:
        h = ((r-g)/m)*60 + 240
    if mx == 0:
        s = 0
    else:
        s = m/mx
    v = mx
    H = h / 2
    S = s * 255.0
    V = v * 255.0
    return H, S, V

def green(H, S, V):
    if(H>=50 and H<=77 and S>=70 and S<=255 and V>=46 and V<=255):
        return True

def main(fileDir,tarDir):
    pathDir = os.listdir(fileDir)
    filenumber = len(pathDir)
    print(filenumber)
    i=0
    ratio=[[0] * 2 for _ in range(filenumber)]
    for filename in pathDir :
        # 打开原始图片
        image_path=os.path.join(fileDir, filename)
        image = Image.open(image_path)
        ratio[i][0]=str(filename)
        cal_path = os.path.join(tarDir, filename)  #filename
        ratio[i][1]=calculate(image,cal_path)
        i+=1
        
    # csv文件保存
    # crop_path =os.path.join(tarDir, 'score.csv')
    # with open(crop_path,'w',encoding='utf8',newline='') as f :
    #     writer = csv.writer(f)
    #     writer.writerows(ratio)

def calculate(image,cal_path):
        

        # 转换为RGB模式
        # img=image.convert("RGB")
        dataset_re=np.asarray(image)
        fig = plt.figure()
        # ax = fig.add_subplot(1,3,1)
        # plt.imshow(dataset_re)

        # dst=cv2.blur(dataset_re, (5, 5))  #均值滤波
        dst=cv2.bilateralFilter(dataset_re, 0, 100, 5)  #双边滤波
        # dst=cv2.medianBlur(dataset_re, 5)  #中值滤波
        # spatial_radius = 10  # 空间半径
        # color_radius = 30    # 颜色半径
        # dst = cv2.pyrMeanShiftFiltering(dataset_re, spatial_radius, color_radius)  #均值漂移
        dataset=np.array(dst)
        # edges = cv2.Canny(dataset, 100, 200)
        # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # segmented_image = np.zeros_like(image)
        # cv2.drawContours(segmented_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        ax = fig.add_subplot(1,2,1)
        plt.imshow(dataset)

        hsv_image = cv2.cvtColor(dataset, cv2.COLOR_BGR2HSV)
        # HSV 的下界限
        lower_green = np.array([50,50,46])
        # HSV 的上界限
        upper_green = np.array([90,255,255])  #[50,50,46] [90,255,255] Index1
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        # cv2.imwrite("/root/cabbage/image_cut/mask.jpg", mask)
       
        # 获取像素点数据
        img_dst=Image.fromarray(dst)
        pixels = img_dst.load()
        width, height = img_dst.size
        
        # 初始化计数器和总像素数
        red_count = 0
        green_count = 0
        blue_count = 0
        total_pixels = width * height

        # 遍历每个像素点，统计红色、绿色和蓝色像素的数量
        # for x in range(width):
        #     for y in range(height):
        #         r, g, b = pixels[x, y]
        #         temp0,temp1,temp2 = rgb2hsv(r, g, b)#颜色转化
        #         if(green(temp0,temp1,temp2)):
        #         # # 排除黑色素
        #         # if r+g+b>0:
        #         #     total=r+g+b
        #         #     red=r/total
        #         #     green=g/total
        #         #     blue=b/total
        #         # if r > 200 and g > 200 and b > 200:
        #             # red_count += 1
        #         # if red<0.4 and g > 0.6 and b < 0.4:
        #             # green_count += 1
        #             dataset[y,x,0] = 255
        #             dataset[y,x,1] = 255
        #             dataset[y,x,2] = 255
        #         # elif r > 100 and g > 100 and b > 100:
        #         #     blue_count += 1
        #         else:
        #             dataset[y,x,0] = 0
        #             dataset[y,x,1] = 0
        #             dataset[y,x,2] = 0
                     
        # 先闭运算，后开运算
        kernel = np.ones((5, 5), dtype=np.uint8)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        # 通过调整这里的*255的地方可以调整掩膜颜色
        # opening1 = cv2.merge([opening, opening, opening])
        
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(opening, contours, -1, (255,255,255), thickness=cv2.FILLED)  #注释  cv2.FILLED
        # and_area = cv2.bitwise_and(segmented_image,opening)
        img_open=Image.fromarray(opening)
        pixels = img_open.load()
        for x in range(width):
            for y in range(height):
                if opening[y,x]==255:
                    green_count += 1
        # 计算每种颜色在总像素数中所占的比例
        # red_ratio = red_count / total_pixels
        green_ratio = green_count / total_pixels
        # blue_ratio = blue_count / total_pixels
        
        # 输出结果
        print("Green ratio:", green_ratio)
        area = cv2.bitwise_and(dataset_re,dataset_re,mask=opening)
        ax = fig.add_subplot(1,2,2)
        plt.imshow(area)
        plt.savefig(cal_path)
        plt.show()
        # cv2.imwrite('/root/cabbage/image_cut/2/closing1.jpg',closing)
        # cv2.imwrite('/root/cabbage/image_cut/2/contours1.jpg',opening)



        return green_ratio

if __name__ == '__main__':
    fileDir = r"/root/cabbage/image_cut/1" + "/"  # 源图片文件夹路径
    tarDir = r'/root/cabbage/image_cut/2'  # 图片移动到新的文件夹路径
    main(fileDir,tarDir)
