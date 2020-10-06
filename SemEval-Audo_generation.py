#-*-coding:utf-8-*-
from pydub import AudioSegment
import requests
import argparse
import pandas as pd
import numpy as np
import os

url = "https://fanyi.baidu.com/gettts"
PATH = 'Audio/'
# datsets files path
dataset_files = {
    'twitter': {
        'train': './datasets/acl-14-short-data/train.raw',
        'test': './datasets/acl-14-short-data/test.raw'
    },
    'restaurant': {
        'train': './datasets/semeval14/Restaurants_Train.xml.seg',
        'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
    },
    'laptop': {
        'train': './datasets/semeval14/Laptops_Train.xml.seg',
        'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
    }
}


def download_audio(text, num):
    # print('你刚才输入的是：{}'.format(text))
    querystring = {"lan":"uk","text":"fox","spd":"3","source":"wise"}
    querystring["text"] = text
    payload = ""
    headers = {
        'authority': "fanyi.baidu.com",
        'method': "GET",
        'scheme': "https",
        'accept': "*/*",
        'accept-encoding': "identity;q=1, *;q=0",
        'accept-language': "zh-CN,zh;q=0.9",
        'cookie': "PSTM=1563348736; MCITY=340-340%3A; BDUSS=FPQXAyb3Jzcy1uR0YwR2ZES2FBVi1IZEJoZVhDN29KaWd5ZW1nZ0ZUVlUzcDVkSVFBQUFBJCQAAAAAAAAAAAEAAAB6oc0ENDg1Nzk3MwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFRRd11UUXddel; REALTIME_TRANS_SWITCH=1; FANYI_WORD_SWITCH=1; HISTORY_SWITCH=1; SOUND_PREFER_SWITCH=1; SOUND_SPD_SWITCH=1; BIDUPSID=95D15018FEF5999FE032E6B2E68B3ACD; BAIDUID=EB550D6DF62B7C71AC954580BB4E8F97:FG=1; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; H_PS_PSSID=1463_31170_21102_30841_31186_30905_30824_31086_26350_31195; delPer=0; PSINO=5; to_lang_often=%5B%7B%22value%22%3A%22en%22%2C%22text%22%3A%22%u82F1%u8BED%22%7D%2C%7B%22value%22%3A%22zh%22%2C%22text%22%3A%22%u4E2D%u6587%22%7D%5D; __yjsv5_shitong=1.0_7_6c3a76c18fa8e74e2421486170f3c07a3576_300_1585703554526_115.159.40.139_1949f7d4; from_lang_often=%5B%7B%22value%22%3A%22zh%22%2C%22text%22%3A%22%u4E2D%u6587%22%7D%2C%7B%22value%22%3A%22en%22%2C%22text%22%3A%22%u82F1%u8BED%22%7D%5D; Hm_lvt_afd111fa62852d1f37001d1f980b6800=1585703649; Hm_lpvt_afd111fa62852d1f37001d1f980b6800=1585703649; Hm_lvt_64ecd82404c51e03dc91cb9e8c025574=1585703554,1585703649; Hm_lpvt_64ecd82404c51e03dc91cb9e8c025574=1585703649; yjs_js_security_passport=a95028b83df87ca97a8dffbcc83bd1527689310a_1585703655_js",
        'range': "bytes=0-",
        'referer': "https://fanyi.baidu.com/?aldtype=16047",
        'user-agent': "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Mobile Safari/537.36",
        'cache-control': "no-cache",
        'Postman-Token': "013ca83e-bf49-4b7d-8b43-761c713105db"
        }
    response = requests.request("GET", url, data=payload, headers=headers, params=querystring)
    r = response.content
    # print(response.text)
    if os.path.exists(PATH + '{}.mp3'.format(text)):
        fo = open(PATH + '{}.mp3'.format(text), 'wb')
        fo.write(r)
        fo.close()
        trans_mp3_to_wav(PATH + f'{text}.mp3', num)
    else:
        pass

def trans_mp3_to_wav(filepath, num):
    song = AudioSegment.from_mp3(filepath)
    song.export(PATH + 'new_sets/' + f"{num}.wav", format="wav")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='restaurant', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--max_seq_len', default=40, type=int)
    opt = parser.parse_args([])
    opt.dataset_file = dataset_files[opt.dataset]

    texts = []
    fnames = [opt.dataset_file['train'], opt.dataset_file['test']]
    j = 0
    stats = []
    for fname in fnames:
        find = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = find.readlines()
        find.close()

        for i in range(0, len(lines), 3):
            tmp = []
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            text_aspect = lines[i + 1].lower().strip()
            text_raw = text_left + ' ' + text_aspect + ' ' + text_right
            sent = text_aspect + ', ' + text_raw
            polarity = lines[i+2]

            texts.append(sent[:opt.max_seq_len])
            tmp.append(j)
            tmp.append(text_raw)
            tmp.append(text_aspect)
            tmp.append('{}.mp3'.format(j))
            tmp.append(polarity)
            stats.append(tmp)
            j += 1
    print(len(stats))
    stats = np.array(stats)
    df = pd.DataFrame(stats)
    df.columns = ['num', 'sentence', 'aspect', 'path', 'polarity']
    df.to_excel(PATH + 'new_sets/' + 'records.xlsx')


    length = int(stats.shape[0])
    # Using texts as a sets to transfor text to audio
    j = 0
    for text in texts:
        print('Processing: {}/{}'.format(j+1, length))
        download_audio(text, j)
        j += 1

