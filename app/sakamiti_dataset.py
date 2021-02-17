import keras
import cv2
from pathlib import Path
import random
import numpy as np
from resize_image import ResizeImage
from extract_face import ExtractFace

class SakamitiDataset():
    """
    ダウンロードした坂道グループの画像データを学習データ化する
    ダウンロードパス："./pic/" + self._group + "/download/" (self._groupには、各グループ名が入る)
    from_image_create=Trueの場合、ダウンロードした画像ファイルから顔のみを抽出しリサイズする
    その後、numpy.arrayに変換する
    from_image_create=Falseの場合、リサイズされている画像データに対して numpy.arrayの変換をかける
    
    Note:
       from_image_create=True とした場合、画像加工が行われるので、画像データ量によっては、數十分程度かかる場合もある
    """
    
    def __init__(self, from_image_create=False):
        """
        init処理
        
        Args:
           from_image_create(bool): 画像データの作成からデータセットを作成する場合はTrue
                                                          画像データの作成を行わない場合はFalse
        Returns:
        　　　None 
        """

        self._groups = [
            "nogizaka",
            "hinatazaka",
            "sakurazaka",
        ]
        # Inception-v3の入力行列のサイズ
        self._image_shape = (139, 139, 3)
        # 今回の分類対象クラス数（0: 乃木坂、1:日向坂、2:櫻坂）
        self._num_classes = 3

        # from_image_create=Trueの場合、画像の加工が行われる
        if from_image_create:
            self._prepare_image()

    def _prepare_image(self):
        """
        ダウンロードした画像から学習データの素となる顔のみの画像データを作成する
        ExtractFace.extract_face() 、ResizeImage.resize_image()  を利用
        
        Args:
           None
        
        Returns:
           None
        """
        
        # 画像から顔のみを抽出
        for group in self._groups:
           ins = group + "_ef"
           target_dir = "./pic/" + group + "/download/"
           target = Path(target_dir)
           targets = [str(f) for f in target.glob("*/*.jpg")]
           ins = ExtractFace(sakamiti=group)
        
           for target_f in targets:
               ins.extract_face(original_face_image_path=target_f)
            
        # 顔データをリサイズ
        for group in self._groups:
           ins = group + "_ef"
           target_dir = "./pic/" + group + "/extract/"
           target = Path(target_dir)
           targets = [str(f) for f in target.glob("*/*.jpg")]
           ins = ResizeImage(sakamiti=group)
           
           for target_f in targets:
               ins.resize_image(input_image_path=target_f)       
        
        
    def _make_train_data(self, test_per=0.3):
        """
        学習データを作成する
        
        Args:
           test_per(float): テストデータの割合

        Returns:
           X_train, X_test, y_train, y_test
        """
        
        # [image, class]        
        train = []

        # resizeディレクトリの画像データを取得し、cv2を利用してarray化している
        # array化したデータをtrainリストに格納する
        
        for cls, group in enumerate(self._groups):
            target_dir = "./pic/" +  group + "/resize/"
            target_base= Path(target_dir)
            targets = [str(f) for f in target_base.glob("*/*.jpg")]
            
            for target in targets:
                #print(f'target = {target}')
                imarray = cv2.imread(target)
                
                # cls = (0: 乃木坂、1:日向坂、2:櫻坂）
                train.append([imarray, cls])
        
        # train リストの順序をランダムに変更する
        random.shuffle(train)
        
        # 説明変数リスト
        X_train = []

        # 目的変数リスト
        y_train = []
        
        # feature: 画像データ
        # label: 教師データ
        for feature, label in train:
            X_train.append(feature)
            y_train.append(label)
        
        # テストデータのサイズを算出
        test_len = int(len(X_train) * test_per)
        
        X_train = np.array(X_train[test_len:])
        y_train = np.array(y_train[test_len:])
        X_test = np.array(X_train[:test_len])
        y_test = np.array(y_train[:test_len])
         
        return (X_train, X_test), (y_train, y_test)

    def _preprocess(self, data, label_data=False):
        
        if label_data:
            # 1-hot-vectorに変換
            # 0 -> ([1,0,0]), 1 -> ([0,1,0]), 2 -> ([0,0,1])
            data = keras.utils.to_categorical(data, self._num_classes)
        else:
            # float型、 画像データを0~1スケールに変換する
            data = data.astype('float32')
            data /= 255

        return data
    
    def get_batch(self, test_per):
        """
        numpy.arrayデータを取得する
        
        Args:
           test_per(float): テストデータの割合
        
        Returns:
           X_train, X_test, y_train, y_test
        """
        (X_train, X_test), (y_train, y_test) =self._make_train_data(test_per=test_per)
        X_train, X_test = [self._preprocess(data=d)  for d in [X_train, X_test]]
        y_train, y_test = [self._preprocess(data=d, label_data=True)  for d in [y_train, y_test]]
        
        return X_train, X_test, y_train, y_test