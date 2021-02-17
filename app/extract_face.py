from PIL import Image
import face_recognition
from pathlib import Path
import os

class ExtractFace():
    def __init__(self, sakamiti:str):
        self._sakamiti = sakamiti
        self._extract_dir = "./pic/" +  self._sakamiti + "/extract/"

    def extract_face(self, original_face_image_path, model="cnn"):
       """ 
       ダウンロードした画像にて face_recognition を利用し顔のみ抽出する
       実際にダウンロードした画像には、顔部分以外にも体など不要な情報が含まれているため、顔のみ抽出する
    
       Args:
          original_face_image_path (str) : 顔抽出対象の画像
          output_path
    
       """
       # ファイル名直上ディレクトリ名（＝メンバー名）を取得
       member_name = original_face_image_path.split('/')[-2:-1] 
    
       filename = os.path.basename(original_face_image_path)
    
       # ファイル名を取得
       extract_path = self._extract_dir  + member_name[0] 
    
       if not os.path.exists(extract_path):
           os.makedirs(extract_path)
    
       # face recognitionに抽出対象の画像を読み込ませる
       image = face_recognition.load_image_file(original_face_image_path)

       # 顔の位置を識別する
       face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model=model)

       for face_location in face_locations:
           top, right, bottom, left = face_location
           face_image = image[top:bottom, left:right]
           pil_image = Image.fromarray(face_image)
           print(f'extract_path = {extract_path}')
           pil_image.save(os.path.join(extract_path, filename))
        
