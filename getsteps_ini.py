# %%
import sys, os, time, csv, re, shutil
import linecache
import math
import scipy.stats as stats
import glob
import configparser
import argparse
import traceback
import openpyxl

import numpy as np
import cv2
import pyocr, pyocr.builders
import matplotlib.pyplot as plt
from PIL import Image
plt.gray()
# %%
#OCRで使うソフトウェアを設定
#tesseract win binary(64bit)を導入した
#pyocrに対応したOCRソフトとしてtesseractだけがインストールされていることを想定する
pyocr.tesseract.TESSERACT_CMD = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
TOOLS = pyocr.get_available_tools()
TOOL = TOOLS[0]

# %%

#コマンドライン引数
parser = argparse.ArgumentParser()
parser.add_argument('config_file', help='設定ファイルのパス')
args = parser.parse_args()

#設定読取
config = configparser.ConfigParser()
config.read(args.config_file, encoding='utf-8')
_DATA_DIR_PATH = config['PATH']['Data_dir']
_RESULT_CSV_PATH = config['PATH']['Result_csv_file']
_ERROR_CSV_PATH = config['PATH']['Error_csv_file']
_OUTPUT_EXCEL_PATH = config['PATH']['Output_excel_file']
_IMG_PERIOD_DIR = config['PATH']['Img_period_dir']
_IMG_LABEL_DIR = config['PATH']['Img_label_dir']
_IMG_SUCCESS_DIR = config['PATH']['Img_success_dir']
_IMG_ERROR_DIR = config['PATH']['Img_error_dir']
_IMG_OLDLAYOUT_DIR = config['PATH']['Img_oldlayout_dir']
# %%

class StepsImage():
    """歩数の画像クラス

    Attributes:
        filepath (str): 画像ファイルのパス
        img_org (numpy.array): OpenCVで読み込んだ画像ファイルのデータ(カラー)
        img_org_gray (numpy.array): img_orgをグレースケール変換した
        is_dark_mode (boolean): 画像ファイルがダークモードか?
        top_region (list): 画像ファイルのグラフより上の部分(平均歩数や期間が書いているところ)
        のy座標From, To
        graph_region (list): 画像ファイルのグラフ部分のy座標From, To
        img_top (numpy.array): グラフより上の部分の画像(カラー)
        img_graph (numpy.array): グラフ部分の画像(カラー)
        img_bin_top (numpy.array): グラフより上の部分の画像(白黒２値)
        img_bin_graph (numpy.array): グラフ部分の画像(白黒２値)
    """

    BIN_BLACK = 0
    BIN_WHITE = 255
    #グラフにつき31個のbinがある
    BIN_NUMBER = 31

    def __init__(self, filepath):
        """コンストラクタ

        Arguments:
            filepath (str): ファイルパス
        """

        #何度も呼び出し、さほどデータサイズも大きくないデータはクラス変数とする
        self.filepath = filepath
        #self.filename = os.path.basename(self.filepath)
        self.fileext = os.path.splitext(self.filepath)[1].lower()
        #cv2.imreadは日本語ファイル名を読めない
        #self.img_org = cv2.imread(self.filepath)
        self.img_org = cv2.imdecode(np.fromfile(self.filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.img_org_gray = cv2.cvtColor(self.img_org, cv2.COLOR_BGR2GRAY)
        #最初のイメージの背景での上がどこにあるか検索している
        self.is_old_layout = self.__is_old_layout()
        self.is_dark_mode = self.__is_dark_mode()
        #最初のイメージの背景での下がどこにあるか検索している
        self.top_region, self.graph_region = self.__get_region_y_range()
        self.img_top = self.img_org[self.top_region[0]:self.top_region[1], :]
        self.img_graph = self.img_org[self.graph_region[0]:self.graph_region[1], :]
        #OCRの精度を上げるためグラフ解析用とOCR用のイメージは別の条件でバイナリイメージを作る
        self.img_bin_top, self.img_bin_top_ocr = self.__get_binary_image(
            self.img_org_gray[self.top_region[0]:self.top_region[1], :]
        )
        self.img_bin_graph, self.img_bin_graph_ocr = self.__get_binary_image(
            self.img_org_gray[self.graph_region[0]:self.graph_region[1], :],
        )
        self.bar_x_ranges, self.graph_x_range, self.graph_y_range = self.__graph_info()

    def __is_old_layout(self):
        """旧レイアウト（iOS 12以前）か？

        Returns
            bool: True 旧レイアウト, False 新レイアウト
        """

        #ピンク系の色が存在している場合、旧レイアウトとみなす
        img_hsv = cv2.cvtColor(self.img_org, cv2.COLOR_BGR2HSV)
        img_pink = np.where((img_hsv[:,:,0] > 120) & (img_hsv[:,:,1] > 160) & (img_hsv[:,:,2] > 220) , 0, 255)
        cnt = np.count_nonzero(img_pink == 0)
        if cnt > 500:
            return True
        return False

    def __is_dark_mode(self):
        """画像はダークモードか?

        Returns
            bool: True ダークモード, False 通常モード
        """
        #↓これは前のやつのとき白がダークと思われたかも？色みを替えたコード
        #_, img = cv2.threshold(self.img_org_gray, 254, 255, cv2.THRESH_BINARY)
        _, img = cv2.threshold(self.img_org_gray, 128, 255, cv2.THRESH_BINARY)
        #0.1から0.3に変更：@11/27
        return (cv2.countNonZero(img)/img.size < 0.3)

    def __get_binary_image(self, img):
        """白黒2値データを取得する

        Arguments:
            img (numpy.array): 画像データ(カラー)

        Returns:
            numpy.array -- 白黒2値データ
            numpy.array -- OCR用の白黒2値データ
        """

        th_min, th_max, bin_mode = (250, 255, cv2.THRESH_BINARY) if not self.is_dark_mode else (0, 255, cv2.THRESH_BINARY_INV)
        th_min_ocr, th_max_ocr, bin_mode_ocr = (230, 255, cv2.THRESH_BINARY) if not self.is_dark_mode else (35, 255, cv2.THRESH_BINARY_INV)

        _, img_bin = cv2.threshold(img, th_min, th_max, bin_mode)
        _, img_bin_ocr = cv2.threshold(img, th_min_ocr, th_max_ocr, bin_mode_ocr)

        #２値化の結果黒が多い場合（Thresholdの条件が厳しい場合）は大津の２値化をベースにThresholdを決める
        mode, _ = stats.mode(np.ravel(img_bin))
        if mode == 0:
            th, _ = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
            _, img_bin = cv2.threshold(img, min(254, th+60), 255, bin_mode)
            _, img_bin_ocr = cv2.threshold(img, th, 255, bin_mode_ocr)

        return img_bin, img_bin_ocr

    def __get_region_y_range(self):
        """画像中のグラフより上の部分とグラフ部分それぞれのy軸From, Toを取得する

        Returns:
            (list, list) -- ([グラフより上の部分のy軸From, To], [グラフ部分のy軸From, To])
        """

        #どこからスタートしているかの縦の線（どこから色を探しているか）
        center = self.img_org.shape[1]//2
        right_edge_ver_line = self.img_org[:, center-10:center+10]

        cv2.imwrite('img_org.png', self.img_org)

        #画像のほとんどが黒い場合は２値化の条件を緩める
        if stats.mode(right_edge_ver_line) == 0:
            th_min, th_max, bin_mode = (250, 255, cv2.THRESH_BINARY) if not self.is_dark_mode else (20, 255, cv2.THRESH_BINARY_INV)
            _, img_bin = cv2.threshold(self.img_org_gray, th_min, th_max, bin_mode)

        #黒線（１ピクセル）が入っている１番目～２番目の位置
        #7,11→2,7に変更した@11/20
        #top_index, bottom_index = (2, 7) if not self.is_dark_mode else (3, -2)
        #top_index, bottom_index = 7, 11
        #light_lines = np.where((right_edge_ver_line == [242, 241, 241]) |
                    #(right_edge_ver_line == [246, 242, 242]))[0]
        #白ではないけど白っぽい
        light_lines = np.where((right_edge_ver_line <self.BIN_WHITE) &
                    (right_edge_ver_line > [230, 230, 230]))[0]
        dark_lines = np.where((right_edge_ver_line > self.BIN_BLACK) &
                    (right_edge_ver_line < [30, 30, 30]))[0]
                    
    

        def get_rep_points(point_):
            #境界線のピクセルが連続する場合を考慮した上で分割する
            begin_idx, pre_idx, end_idx = point_[0], point_[-1], 0
            rep_points = []
            for idx in point_:
                if (idx - pre_idx) <= 1:
                    pre_idx = idx
                    continue
                end_idx = pre_idx
                rep_points.append([begin_idx, end_idx])
                begin_idx, pre_idx = idx, idx
            else:
                rep_points.append([begin_idx, idx])
            #重要:10を引いてる意味：右の数字を探そうとしてる
            #他の画像で上手くいかなかったらここをいじる（例：-10⇒-20とか）
            #携帯の高さによって変わるかもしれない？
            return rep_points

        rep_points = get_rep_points(light_lines if not self.is_dark_mode else dark_lines)
        img_orange = orange_other_binarization(self.img_org)
        orange_rep_points = get_rep_points(np.where(img_orange == self.BIN_BLACK)[0])
        max_range = max(rep[1] - rep[0] for rep in orange_rep_points)
        top_y_orange, bottom_y = [rep for rep in orange_rep_points if rep[1] - rep[0] == max_range][0]
        top_y = max(rep[1] for rep in rep_points if rep[1] < top_y_orange) - 20

        #平均歩数や期間の領域, グラフの領域
        return [0, top_y - 1], [top_y + 1, bottom_y + 10 - 1]

    def __graph_info(self):
        """棒グラフ部分の情報を取得する

        Returns:
            (
            list,: [[棒のx軸From, To, 棒の高さ],[...],...]
            list,: [グラフ領域のx軸From, To]
            list,: [グラフ領域のy軸From, To]
            )

        ToDo:
            短くする
            やっていることが多すぎる
        """

        #グラフの底辺部分
        bin_ = orange_other_binarization(self.img_graph)
        bin_black = np.where(bin_ == self.BIN_BLACK)
        #グラフの一番下にある黒い色を探してる
        #オレンジが一番多い行の中で一番下の行
        #棒が短すぎると、オレンジ色が下の線とぶつけ合う
        #なので、max(bin_black[0])はバグってしまう（OSが新しくなったからではなく、もともとかも？）
        bottom_y, _ = sorted(zip(*np.unique(bin_black[0], return_counts=True)), key=lambda c:(c[1],c[0]), reverse=True)[0]
        bottom = np.where(bin_[bottom_y, :] == self.BIN_BLACK)[0]

        #棒の高さを取得する
        def height(begin_idx, end_idx):
            center = begin_idx + int((end_idx -begin_idx)/2)
            return bottom_y - min(bin_black[0][bin_black[1]==center]) + 1

        #各棒のX軸範囲と高さ
        begin_idx, pre_idx, end_idx = bottom[0], bottom[-1], 0
        bar_x_ranges = []
        for idx in bottom:
            if (idx - pre_idx) <= 1:
                pre_idx = idx
                continue
            end_idx = pre_idx
            bar_x_ranges.append([begin_idx, end_idx, height(begin_idx, end_idx)])
            begin_idx, pre_idx = idx, idx
        else:
            bar_x_ranges.append([begin_idx, idx, height(begin_idx, idx)])

        #グラフ領域のx軸範囲
        graph_bottom = np.where(self.img_bin_graph[bottom_y-2:bottom_y+2, :].transpose() == self.BIN_BLACK)[0]
        #
        pre_idx = graph_bottom[0]
        for idx in graph_bottom:
            if 1 < (idx - pre_idx):
                graph_x_range = [graph_bottom[0], pre_idx]
                break
            pre_idx = idx
        else:
            graph_x_range = [graph_bottom[0], pre_idx]
        #グラフ領域のy軸範囲
        non_bar_center = bar_x_ranges[0][1] + int((bar_x_ranges[1][0] - bar_x_ranges[0][1])/2)
        center_space_black = np.where(self.img_bin_graph[:, non_bar_center] == self.BIN_BLACK)[0]
        #棒と棒の中心に縦点線が当たる場合がある. その場合は一つ右隣の棒と棒の中心を使う
        #点線がない場合は, 黒ドットは高々30個程度だが, 点線がある場合は黒ドットは200個近く現れる
        #かなり少なめに見積もって100個を判定基準とする
        if 100 < len(center_space_black):
            non_bar_center = bar_x_ranges[1][1] + int((bar_x_ranges[2][0] - bar_x_ranges[1][1])/2)
            center_space_black = np.where(self.img_bin_graph[:, non_bar_center] == self.BIN_BLACK)[0]
        top_y = min(center_space_black)

        graph_y_range = [top_y, bottom_y]

        return bar_x_ranges, graph_x_range, graph_y_range

    def __repr__(self):
        info = [
            f'filepath : {self.filepath}',
        ]
        return str(info)

def get_period_y_region(img_top):
    """期間のy軸From, Toを取得する

    Arguments:
        img_top (numpy.array): グラフより上の部分のデータ(白黒2値)

    Returns:
        (int, int) -- (y軸From, To)
    """

    #画像中の黒ドット部分の座標
    search_img = img_top[:, 0:int(img_top.shape[1]*2/3)]
    black = np.unique(np.where(search_img == StepsImage.BIN_BLACK)[0])

    #画像最下部から上方向に走査する
    #y軸方向に走査して初めて検出した黒ドットが続く部分が該当箇所
    pre_idx = black[black.shape[0]-1]
    for idx in black[::-1]:
        if 1 < pre_idx - idx:
            return idx - 1, img_top.shape[0] - 1
        pre_idx = idx

def get_label_y_region(img_graph, start_x):
    """棒グラフの一番上の補助線のラベルのy軸From, Toを取得する

    Arguments:
        img_graph (numpy.array): グラフ部分のデータ(白黒2値)
        start_x (int): 走査を開始するx座標

    Returns:
        (int, int): (y軸From, To)
    """
    #走査する部分を切り出した画像中の黒ドット部分の座標
    search_img = img_graph[:, start_x+2:]
    black = np.unique(np.where(search_img == StepsImage.BIN_BLACK)[0])

    #画像最上部から下方向に走査する
    pre_idx = black[0]
    for idx in black:
        if 4 < idx - pre_idx:
            #多少バッファを見る
            return 0, pre_idx + 7
        pre_idx = idx

def orange_other_binarization(img):
    """画像中のオレンジ系の箇所を黒, その他を白とする白黒2値に変換する

    Arguments:
        img (numpy.array): グラフ部分の画像データ(カラー)

    Returns:
        numpy.array: 白黒2値画像データ
    """

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #オレンジ系の色（Hが0～60度）、これらの数値は結果がうまくいくようにチューニングする
    return np.where((img_hsv[:,:,0] < 20) & (img_hsv[:,:,1] > 10) & (img_hsv[:,:,2] > 60), 0, 255)

def write_csv(path, header, data, encoding='utf8'):
    """CSV出力する

    Args:
        path (str): CSVファイルパス
        header (numpy.array): ヘッダー
        data (numpy.array): レコード
        encoding (str, optional): エンコーディング. Defaults to 'utf8'.
    """
    try:
        if os.path.exists(path):
            os.remove(path)

        with open(path, 'a', encoding=encoding, newline='') as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for d in data:
                w.writerow(d)

    except IOError:
        print(f'[error]{path} is opened.')


def initialize_dir(path):
    """出力先フォルダを空にする

    Args:
        path (str): 出力先フォルダパス
    """

    try:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

    except IOError:
        print(f'[error]{path} is opened.')

def main(image):
    """メイン処理

    Arguments:
        image (StepsImage): StepsImageのインスタンス

    Returns:
        (str,:期間
         str,: 棒グラフの一番上のy軸ラベル
         list,: 棒の高さ
         int,: pixel当たりの棒の高さ(歩数/pixel)
        )
    """

    #グラフの情報を取る
    #bar_info, graph_x_range, graph_y_range = image.graph_info()
    #image.bar_x_ranges, image.graph_x_range, image.graph_y_range = image.graph_info()

    #大津の２値化
    def binarize_otsu(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, o = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)
        if image.is_dark_mode:
            o = cv2.bitwise_not(o)
        return o

    #期間をOCRする
    period_top, period_bottom = get_period_y_region(image.img_bin_top_ocr)
    #period_img = image.img_bin_top_ocr[period_top:period_bottom, 0:int(image.img_bin_top_ocr.shape[1]*2/3)]
    period_img = binarize_otsu(image.img_top[period_top:period_bottom, 0:int(image.img_bin_top_ocr.shape[1]*2/3)])
    period_text = TOOL.image_to_string(Image.fromarray(period_img),
                                       lang='script/Japanese',
                                       builder=pyocr.builders.TextBuilder(tesseract_layout=6))
    period_text = period_text.replace(' ','').replace('~', '～')

    #グラフの上限値をOCRする
    label_top, label_bottom = get_label_y_region(image.img_bin_graph_ocr, image.graph_x_range[1])
    #label_img = image.img_bin_graph_ocr[label_top:label_bottom, graph_x_range[1]+1:]
    label_img = binarize_otsu(image.img_graph[label_top:label_bottom, image.graph_x_range[1]+1:])
    label_text = TOOL.image_to_string(Image.fromarray(label_img),
                                      lang='eng',
                                      builder=pyocr.builders.DigitBuilder(tesseract_layout=7))
    label_text = label_text.replace(' ','').replace(',', '').replace('.', '')
    label_text = re.sub('[^0-9]', '', label_text)

    #heightが31個になるように棒のない箇所の高さをゼロとして補完する
    bar_width = image.bar_x_ranges[0][1] - image.bar_x_ranges[0][0] + 1
    graph_width = image.graph_x_range[1] - image.graph_x_range[0] - 1
    space_width = graph_width- bar_width*StepsImage.BIN_NUMBER
    space_unit = int(space_width/StepsImage.BIN_NUMBER)
    edge_space = int(space_width/StepsImage.BIN_NUMBER/2)
    b_s_width = bar_width + space_unit
    j, heights = 0, np.zeros(StepsImage.BIN_NUMBER)
    for idx, bar in enumerate(image.bar_x_ranges):
        if idx == 0:
            s = bar[0] - image.graph_x_range[0] - 1
            n = int((s - edge_space)/b_s_width)
        else:
            s = bar[0] - image.bar_x_ranges[idx-1][1] - 1
            n = int((s - space_unit)/b_s_width)
        j += n
        heights[j] = bar[2]
        j += 1

    #高さが読めていない場合はバーのピクセルの高さを返す
    height_unit = ''
    max_height_pixel = image.graph_y_range[1] - image.graph_y_range[0]
    #height_est = ['' for i in range(1, StepsImage.BIN_NUMBER+1)]
    if label_text != '':
        #1ピクセル当たりの棒の高さ(歩数)
        height_unit = int(label_text)/max_height_pixel

        #棒グラフの数値を推定する
        #heights_est = heights*height_unit
        #heights_est = np.round(heights).astype(np.int32).astype(np.unicode)

    return period_img, period_text, label_img, label_text, max_height_pixel, heights

if __name__== '__main__':
    start_time = time.time()

    #処理対象ファイルリスト
    types = ('/*.png', '/*.jpg', '/*.jpeg')
    #types = ('/*.jpg', '/*.jpeg')
    files = []
    for type_ in types:
        files.extend(glob.glob(_DATA_DIR_PATH + type_))

    #出力先フォルダの初期化
    for p in [_IMG_LABEL_DIR, _IMG_PERIOD_DIR, _IMG_SUCCESS_DIR, _IMG_ERROR_DIR, _IMG_OLDLAYOUT_DIR ]:
        initialize_dir(p)

    #CSVのヘッダ
    #header = ['filename', 'period', 'max_height', 'height_per_pixel']
    header = ['period_img', 'max_height_img', 'filename', 'period', 'max_height', 'max_height_pixel']
    header.extend([str(i) for i in range(1, StepsImage.BIN_NUMBER + 1)])
    error_header = ['filename', 'exception_type', 'message', 'lineno']

    #Excelファイルの新規作成し、ヘッダを埋め込む
    if os.path.exists(_OUTPUT_EXCEL_PATH):
        os.remove(_OUTPUT_EXCEL_PATH)
    wb = openpyxl.Workbook()
    ws = wb.worksheets[0]
    for i, h in enumerate(header):
        c = ws.cell(row=1, column=i+1)
        c.value = h
    wb.save(_OUTPUT_EXCEL_PATH)

    #Excelを開いておく
    wb = openpyxl.load_workbook(_OUTPUT_EXCEL_PATH)
    ws = wb.worksheets[0]

    #棒グラフを読み取る
    results = []
    error_results = []
    max_width_period, max_width_label = 0, 0
    ni = 1
    for index, file_ in enumerate(files):
        filename = os.path.basename(file_)
        try:
            if not os.path.exists(file_):
                raise FileNotFoundError()

            print(f'---{index+1}/{len(files)}---')
            print(filename)

            image = None
            image = StepsImage(file_)

            if image.is_old_layout:
                raise AttributeError('Old layout.')
            img_period, period, img_label, label, max_height_pixel, heights = main(image)
            #OCRした部分の画像を出力
            img_period_path = _IMG_PERIOD_DIR + '/' + filename
            img_label_path = _IMG_LABEL_DIR + '/' + filename
            cv2.imwrite(img_period_path, img_period)
            cv2.imwrite(img_label_path, img_label)

            #Excelに出力
            img_period_r = openpyxl.drawing.image.Image(img_period_path)
            img_label_r = openpyxl.drawing.image.Image(img_label_path)

            #出力先のセルアドレスを取得
            now_row = ni + 1 #header行を飛ばす
            addr_period = ws.cell(row=now_row, column=1).coordinate
            addr_label = ws.cell(row=now_row, column=2).coordinate

            #セル幅調整
            h = max(img_period.shape[0], img_label.shape[0])
            ws.row_dimensions[now_row].height = h*0.75
            if max_width_period < img_period.shape[1]:
                max_width_period = img_period.shape[1]
            if max_width_label < img_label.shape[1]:
                max_width_label = img_label.shape[1]

            #画像のアンカ設定
            img_period_r.anchor = addr_period
            img_label_r.anchor = addr_label

            #画像出力
            ws.add_image(img_period_r)
            ws.add_image(img_label_r)

            #棒グラフの高さを推測
            heights_xlsf = [f'=E{now_row}/F{now_row}*{i[1]}' for i in enumerate(heights)]
            #record =  [image.filename, period, label, max_height_pixel] + heights.tolist()
            record =  [filename, period, label, max_height_pixel] + heights_xlsf

            #文字データのExcelへの埋め込み
            for i, h in enumerate(record):
                c = ws.cell(row=now_row, column=3+i)
                c.value = h

            #CSV出力用データ
            results.append(dict(zip(header, record)))
            shutil.copy(file_, _IMG_SUCCESS_DIR)

            #Excel Save 100件に1回
            if (index + 1) % 100 == 0:
                wb.save(_OUTPUT_EXCEL_PATH)

            ni += 1

        except FileNotFoundError as fnfe:
            error_results.append(dict(zip(error_header, [filename, 'FileNotFoundError', ''])))
            shutil.copy(file_, _IMG_ERROR_DIR)

        except Exception as e:
            exc_type, exc_obj, tb=sys.exc_info()
            lineno=tb.tb_lineno
            error_results.append(dict(zip(error_header, [filename, type(e), e.args[0], lineno])))
            if image is None:
                shutil.copy(file_, _IMG_ERROR_DIR)
                continue

            if image.is_old_layout:
                shutil.copy(file_, _IMG_OLDLAYOUT_DIR)
                continue
            shutil.copy(file_, _IMG_ERROR_DIR)

    else:
        wb.save(_OUTPUT_EXCEL_PATH)

    #Excelのグラフ貼り付け部分のサイズ変更
    wb = openpyxl.load_workbook(_OUTPUT_EXCEL_PATH)
    ws = wb.worksheets[0]
    ws.column_dimensions['A'].width = max_width_period * 0.14
    ws.column_dimensions['B'].width = max_width_label * 0.14
    wb.save(_OUTPUT_EXCEL_PATH)

    #CSV出力
    encoding_ = 'utf8'
    #encoding_ = 'cp932'
    write_csv(_RESULT_CSV_PATH, header, results, encoding_)
    write_csv(_ERROR_CSV_PATH, error_header, error_results, encoding_)

    print(f'Elapsed time: {time.time() - start_time} second.')
