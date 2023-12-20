# folder 안의 class별로 분류된 하위 folder들을 순회하며
# 안에 있는 이미지의 "경로 + 파일명 | class"을 데이터프레임에 저장
# 완성된 data frame을 json 파일로 저장

import os
import pandas as pd

num_match = {
    'background' : 0,
    'brick'      : 1,
    'concrete'   : 2,
    'ground'     : 3,
    'wood'       : 4
}

def make_list(input_folder, output_file):
    columns = ['image_path', 'target']
    df = pd.DataFrame(columns=columns)
    for target, num in num_match.items():
        image_path = os.path.join(input_folder, target)
        image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        for image_file in image_files:
            file_path = target + '/' + image_file
            new_row = pd.DataFrame([{'image_path': file_path, 'target': num}])
            df = pd.concat([df, new_row], ignore_index = True)
    print(df)
    print("\n")
    output_file = output_file + '.json'
    print("Saving... "+output_file+"\n\n")
    df.to_json(output_file, orient='records', lines=True)

def main():
    make_list(r'..\part6\monday', r'..\part1\monday')
    make_list(r'..\part6\tuesday', r'..\part1\tuesday')

if __name__ == "__main__":
    main()