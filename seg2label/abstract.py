#此文件用于将txt标签中的0和1提取出来，并保存为新的txt标签。
##This file is used to extract 0 and 1 from the txt label and save it as a new txt label.

import os

def extract_elements_and_save_new_txt(directory, out_directory):
    os.makedirs(out_directory, exist_ok=True)

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            new_lines = []
            with open(os.path.join(directory, filename), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    #根据自己数据集的标签调整类别  ##Adjust the category according to your own data set
                    if line.strip().startswith(('0')) or line.strip().startswith(('1')):
                        elements = line.split()
                        elements[1:] = [format(float(e), '.6f') for e in elements[1:]]
                        new_line = ' '.join(elements) + '\n'
                        new_lines.append(new_line)

            with open(os.path.join(out_directory, filename), 'w') as out_file:
                out_file.writelines(new_lines)

input_label = '../output/strawberry/labels'
output_label = '../output/strawberry/labels_0_1'

files = os.listdir(input_label)
for file in files:
    label_path = os.path.join(input_label, file)
    output_path = os.path.join(output_label, file)

    extract_elements_and_save_new_txt(label_path, output_path)
