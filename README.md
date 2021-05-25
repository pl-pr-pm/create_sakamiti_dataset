# create_sakamiti_dataset
坂道グループの学習データ生成とその学習データを利用したモデル作成を管理

学習データ生成
- sakamiti_dataset.py(a)
   - image_scraping.py(b)
   - resize_image.py(c)
   - extract_face.py(d)

モデル作成
- create-model.py(e)

a:sakamiti_dataset.py
create-model.py で使用する、学習データを生成する
c,dの関数を利用し、画像から顔のみを抽出・リサイズを行い、顔の画像データをnumpy型の配列に変換する

b:image_scraping.py
坂道グループメンバの画像をインターネット経由でダウンロードする

c:resize_image.py
画像を適切なサイズにリサイズする
画像サイズが大きいため、顔抽出に時間がかかったり、モデルに適さない形であるため、リサイズを行う

d:extract_face.py
画像から顔のみを抽出する
学習に余計なデータを省くことで精度向上を狙う


e:create-model.py
モデルを作成する（CNN）
作成したモデルは、spaのバックエンドにて利用される


