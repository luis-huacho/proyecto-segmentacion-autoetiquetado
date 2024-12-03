# This file is used to extract SDM generated label to YOLO mask for YOLO training
# e.g. python extract_label_needed.py --input_folder ./folder1 --output_folder ./folder2 --labels 0 1
import os
import argparse

# Parse arguments
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required = True, help='Path to the SDM generated label folder')
    parser.add_argument('--output_folder', type=str, required = True,help='Path to save mask outputs for YOLO traning')
    parser.add_argument('--labels',  type=int,  nargs='+',  required=True,  help='List of integers, labels you want to keep')    
    return parser.parse_args()


def extract_elements_and_save_new_txt(directory, out_directory, labels_keep):
    labels_keep_str = [str(num) for num in labels_keep]

    os.makedirs(out_directory, exist_ok=True)
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            new_lines = []
            with open(os.path.join(directory, filename), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if any(line.strip().startswith(num_str) for num_str in labels_keep_str):
                        elements = line.split()
                        elements[1:] = [format(float(e), '.6f') for e in elements[1:]]
                        new_line = ' '.join(elements) + '\n'
                        new_lines.append(new_line)

            with open(os.path.join(out_directory, filename), 'w') as out_file:
                out_file.writelines(new_lines)

def main():
    opt = parse_opt()
    input_label_folder = opt.input_folder
    output_label_folder = opt.output_folder
    labels_keep = opt.labels
    print("Input_label_folder: ", input_label_folder)
    print("Output_label_folder: ", output_label_folder)
    print("Labels_keep: ", labels_keep)

    
    folders = os.listdir(input_label_folder)
    print("Folders to process: ", folders)
    for folder in folders:
        label_path = os.path.join(input_label_folder, folder)
        output_path = os.path.join(output_label_folder, folder)
        extract_elements_and_save_new_txt(label_path, output_path, labels_keep)

if __name__ == '__main__':
    main()