"""test.pyの使い方

画像の操作ステップをtest_outputのフォルダに出力する
出力するステップ：
* 灰色の線バイナリ化
* オレンジ色のバイナリ化
* 引き分けたヘッダー部
* 歩数ラベル

テストしたいファイルをtest_graph_extraction()に入れる
フォルダーを使用したい場合はget_sample_files_fixtureを使って、
    フォルダー内の写真ファイル名を全部取得

"""

import os
import glob
from getsteps_iOS16 import StepsImage, extract_details
import cv2
import traceback

#処理対象ファイルリスト
types = ('/*.png', '/*.jpg', '/*.jpeg')
_sample_files = []


def get_sample_files_fixture(folder_path: str):
    if not _sample_files:
        for type_ in types:
            _sample_files.extend(glob.glob(folder_path + type_))
    return _sample_files


def test_graph_extraction(files):

    for index, file_ in enumerate(files):
        filename = os.path.basename(file_)
        print(f'---{index+1}/{len(files)}---')
        print(filename)
        try:
            print(file_)
            if not os.path.exists(file_):
                raise FileNotFoundError()
            image = StepsImage(file_)

            if not cv2.imwrite(f'test_output/lines-{filename}', image.debug_image_lines):
                print("Failed to write image.")
            
            if not cv2.imwrite(f'test_output/orange-{filename}', image.img_orange):
                print("Failed to write image.")

            output_file = f'test_output/{filename}'
            if not cv2.imwrite(output_file, image.img_bin_graph):
                print("Failed to write image.")
            
            img_period, period, img_label, label, max_height_pixel, heights = extract_details(image)
            print(heights)
            
            if not cv2.imwrite(f'test_output/label-{filename}', img_label):
                print("Failed to write image.")

        except Exception as e:
            traceback.print_exc()
            print(e)


if __name__ == '__main__':
    test_graph_extraction(get_sample_files_fixture('data_sample/white'))
