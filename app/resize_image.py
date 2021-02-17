import os
import cv2
from PIL import Image

class ResizeImage():
    def __init__(self, sakamiti:str, resize_shape=(139, 139)):
        """
        init
        
        Args:
           sakamiti (str): グループ名、出力パス等に利用
           resize_shape (tuple): 学習するネットワークのサイズ 
                                                 デフォルトはファインチューニングする入力のサイズとしている
        Returns:
            None
        """
        self._resize_shape = resize_shape
        self._sakamiti = sakamiti
        self._resize_dir = "pic/" +  self._sakamiti + "/resize/"

    def resize_image(self, input_image_path:str):
        """ 
        入力された画像をopen cvを使って学習するネットワークのサイズにリサイズする
        
        Args:
           input_image_path (str): リサイズ対象の画像
        
        Returns:
           None
        """

        # ファイル名直上ディレクトリ名（＝メンバー名）を取得
        member_name = input_image_path.split('/')[-2:-1] 
        
        # ファイル名を取得
        resize_path = self._resize_dir + member_name[0] 

        #入力画像ファイルの親ディレクトリが存在しない場合、ディレクトリを作成する
        if not os.path.exists(resize_path):
            os.makedirs(resize_path)
            
        # open-cvのリサイズを利用するために、open-cvで画像を読み取る
        img = cv2.imread(input_image_path)
        
        # 対象のネットワークの入力サイズに画像をリサイズする
        resize_img = cv2.resize(img, self._resize_shape)
        
        # open-cvは BGR で画像を保存。RGBに変換。
        im_rgb = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
        resize_output_path = os.path.join(resize_path, os.path.basename(input_image_path))
        
        print(f'resize_output_path = {resize_output_path}')
        
        # リサイズし、RGBに変換した画像を保存する
        Image.fromarray(im_rgb).save(resize_output_path)
        # cv2.imwrite(resize_output_path, im_rgb)
