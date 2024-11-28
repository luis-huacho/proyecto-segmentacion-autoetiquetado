##本文件是为了由instance segmentation的label转换为onject detection的label
###This file is intended to be converted from the instance segmentation label to the onject detection label
import os

def convert_coords_to_bbox(input_folder_labels, output_folder_labels):
        
    for filename_train in os.listdir(input_folder_labels):
        input_folder = os.path.join(input_folder_labels, filename_train)
    # 遍历输入文件夹中的所有文件
    ##Iterate over all files in the input folder
        for filename in os.listdir(input_folder):
            if filename.endswith(".txt"):
                # 读取txt文件中的每一行内容
                ##Read the content of each line in the txt file
                with open(os.path.join(input_folder, filename), 'r') as file:
                    lines = file.readlines()
                output_folder = (os.path.join(output_folder_labels, filename_train))
                os.makedirs(output_folder, exist_ok=True)
                # 创建或清空输出文件
                ##Create or clear the output file
                with open(os.path.join(output_folder, filename), 'w') as outfile:
                    # 处理每一行
                    ##Process each line
                    for line in lines:
                        parts = line.strip().split()
                        # 提取类别和坐标点
                        ##Extract the category and coordinate points
                        category = parts[0]
                        coords = list(map(float, parts[1:]))

                        # 将坐标点转换成x_center, y_center, w, h
                        ##Convert the coordinate points to x_center, y_center, w, h
                        x_coords = coords[0::2]  # 获取所有x坐标 ##Get all x coordinates
                        y_coords = coords[1::2]  # 获取所有y坐标 ##Get all y coordinates

                        x_min = min(x_coords)
                        x_max = max(x_coords)
                        y_min = min(y_coords)
                        y_max = max(y_coords)

                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        w = x_max - x_min
                        h = y_max - y_min

                        # 写入转换后的数据  ##Write the converted data
                        outline = f"{category} {x_center} {y_center} {w} {h}\n"
                        outfile.write(outline)

    print("转换完成。")  ##Conversion completed.

# 定义.txt输入和输出文件夹路径  ##Define the input and output folder paths for .txt files
input_folder_labels = '/home/nya/code/apple/out/labels_text_new'
output_folder_labels = '/home/nya/code/apple/data_det'

# 调用函数开始转换  ##Call the function to start the conversion
convert_coords_to_bbox(input_folder_labels, output_folder_labels)