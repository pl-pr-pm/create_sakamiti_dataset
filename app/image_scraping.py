
from __future__ import annotations
from icrawler.builtin import BingImageCrawler
import time
import config # configをインポート

class SakamitiImageScraping:
    
    def __init__(self, max_num):
        
        self._max_num = max_num
        # 坂道グループメンバの定義
        self._nogizaka: list[dict[str, str]] = [
            {'akimoto_manatsu':'秋元真夏'},
            {'ikuta_erika':'生田絵梨花'},
            {'ito_junna':'伊藤純奈'},
            {'ito_riria':'伊藤理々杏'},
            {'iwamoto_renka':'岩本蓮加'},
            {'umezawa_minami':'梅澤美波'},
            {'oozono_momoko':'大園桃子'},
            {'kitano_hinako':'北野日奈子'},
            {'kubo_shiori':'久保史緒里'},
            {'saito_asuka':'齋藤飛鳥'},
            {'sakaguti_tamami':'阪口珠美'},
            {'sato_kaede':'佐藤楓'},
            {'shinuti_mai':'新内眞衣'},
            {'suzuki_ayane':'鈴木絢音'},
            {'takayama_kazumi':'高山一実'},
            {'terada_ranze':'寺田蘭世'},
            {'nakamura_reno':'中村麗乃'},
            {'higuti_hina':'樋口日奈'},
            {'hoshino_minami':'星野みなみ'},
            {'hori_miona':'堀未央奈'},
            {'matsumura_sayuri':'松村沙友理'},
            {'mukai_hazuki':'向井葉月'},
            {'yamazaki_rena':'山崎怜奈'},
            {'yamashita_mizuki':'山下美月'},
            {'yoshida_ayanokurisuthi':'吉田綾乃クリスティー'},
            {'yoda_yuuki':'与田祐希'},
            {'watanabe_miria':'渡辺みり愛'},
            {'wada_maaya':'和田まあや'},
            {'endo_sakura':'遠藤さくら'},
            {'kaki_haruka':'賀喜遥香'},
            {'kakehashi_sayaka':'掛橋沙耶香'},
            {'kanagawa_sayaka':'金川紗耶'},
            {'kitagawa_yuri':'北川悠理'},
            {'kuromi_haruka':'黒見明香'},
            {'sato_rika':'佐藤璃果'},
            {'shibata_yuna':'柴田柚菜'},
            {'seimiya_rei':'清宮レイ'},
            {'tamura_mayu':'田村真佑'},
            {'tutui_ayame':'筒井あやめ'},
            {'hayakawa_seira':'早川聖来'},
            {'hayadhi_runa':'林瑠奈'},
            {'matsuo_miyu':'松尾美佑'},
            {'yakubo_mio':'矢久保美緒'},
            {'yumiki_nao':'弓木奈於'},
            ]
       
        self._hinatazaka: list[dict[str, str]] = [
              {'saito_kyoko':'齋藤京子'},
              {'sasaki_kumi':'佐々木久美'},
              {'sasaki_mirei':'佐々木美玲'},
              {'nibu_akari':'丹生明里'},
              {'kosaka_nao':'小坂菜緒'},
              {'matsuda_konoka':'松田好花'},
              {'katou_shiho':'加藤史帆'},
              {'kageyama_yuka':'影山優佳'},
              {'kanemura_miku':'金村美玖'},
              {'kawata_hina':'河田陽奈'},
              {'kamimura_hinano':'上村ひなの'},
              {'watanabe_miho':'渡邊美穂'},
              {'hamagishi_hiyori':'濱岸ひより'},
              {'takamoto_ayaka':'高本彩花'},
              {'tomita_suzuka':'富田鈴花'},
              {'miyata_manamo':'宮田愛萌'},
              {'higashimura_mei':'東村芽衣'},
              {'ushio_sarina':'潮紗理菜'},
              {'kageyama_yuuka':'影山優佳'},
              {'takahashi_mikuni':'高橋未来虹'},
              {'morimoto_mari':'森本茉莉'},
              {'yamaguchi_haruyo':'山口陽世'},
              ]

        self._sakurazaka : list[dict[str, str]] = [
              {'uemura_rina':'上村莉菜'},
              {'ozeki_rika':'尾関梨香'},
              {'koike_minami':'小池美波'},
              {'kobayashi_yui':'小林由依'},
              {'saito_huyuka':'齋藤冬優花'},
              {'sugai_yuka':'菅井友香'},
              {'habu_miduho':'土生瑞穂'},
              {'harada_aoi':'原田葵'},
              {'moriya_akane':'守屋茜'},
              {'watanabe_rika':'渡辺梨加'},
              {'watanabe_risa':'渡邉理佐'},
              {'inoue_rina':'井上梨名'},
              {'endo_hikari':'遠藤光莉'},
              {'ozono_rei':'大園玲'},
              {'onuma_akiho':'大沼晶保'},
              {'kosaka_marino':'幸阪茉里乃'},
              {'seki_yumiko':'関有美子'},
              {'takemoto_yui':'武元唯衣'},
              {'tamura_hono':'田村保乃'},
              {'huziyoshi_karin':'藤吉夏鈴'},
              {'masumoto_kira':'増本綺良'},
              {'matsuda_rina':'松田里奈'},
              {'matsudaira_riko':'松平璃子'},
              {'morita_hikaru':'森田ひかる'},
              {'moriya_rena':'守屋麗奈'},
              {'yamazaki_ten':'山﨑天'},
              ]

    def _image_scraping(self, keyword: str, max_num: int, storage: str) -> None:
       """
       bingより画像をダウンロードしローカルに保存する

       Args:
           keyword (str): ダウンロードしたい画像の検索ワード
           max_num (int): ダウンロード画像数
           storage (str): ローカルの画像ダウンロード先
       """
       crawler: BingImageCrawler = BingImageCrawler(storage={"root_dir": storage})
       crawler.crawl(keyword=keyword, max_num=max_num)
       # 負担軽減のため、3秒スリープする
       time.sleep(1)

    def sakamiti_scraping(self, cmd="all"):
       """
       坂道グループの画像スクレイピングを実施する
       
       Args:
          cmd (str): ダウンロードしたい坂道グループ. デフォルト "all"
                       all : 全グループ（乃木坂/日向坂/櫻坂）
                       nogi : 乃木坂
                       hinata : 日向坂
                       sakura : 櫻坂
       """
       if cmd is None:
        pass

    # 乃木坂メンバ分をダウンロード
       if cmd == "nogi" or cmd == "all":
         for nogizaka_member in self._nogizaka:
             dest: str = list(nogizaka_member.keys())[0]
             key_word: str = list(nogizaka_member.values())[0]
             storage: str = config.PIC_DIR + config.NOGIZAKA_DIR + config.DOWNLOAD_DIR + dest + '/'
             self._image_scraping(keyword=key_word, max_num=self._max_num, storage=storage)

    # 日向坂メンバ分をダウンロード
       if cmd == "hinata" or cmd == "all":
         for hinatazaka_member in self._hinatazaka:
             dest: str = list(hinatazaka_member.keys())[0]
             key_word: str = list(hinatazaka_member.values())[0]
             storage: str = config.PIC_DIR + config.HINATAZAKA_DIR + config.DOWNLOAD_DIR + dest + '/'
             self._image_scraping(keyword=key_word, max_num=self._max_num, storage=storage)

    # 櫻坂メンバ分をダウンロード
       if cmd == "sakura" or cmd == "all":
          for sakurazaka_member in self._sakurazaka:
             dest: str = list(sakurazaka_member.keys())[0]
             key_word: str = list(sakurazaka_member.values())[0]
             storage: str = config.PIC_DIR + config.SAKURAZAKA_DIR + config.DOWNLOAD_DIR + dest +  '/'
             self._image_scraping(keyword=key_word, max_num=self._max_num, storage=storage)
  
