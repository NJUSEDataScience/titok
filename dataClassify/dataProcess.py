import json
from urllib import request
import csv

musicPath='D:/myMusic'
musicPath1='D:/myMusic/level1' ##用于下载音乐并保存
musicPath2='D:/myMusic/level2'
musicPath3='D:/myMusic/level3'
dataPath='data.json'##获取自动化抓包的数据
bgm1 = []
bgm2 = []
bgm3 = []

def cleanFile():
    bgm1.clear()
    bgm2.clear()
    bgm3.clear()
    with open('info.txt', 'w', encoding='utf-8')as f:
        f.truncate()
    with open('sortByLikes.csv', 'w', newline='') as d:
        d.truncate()

def getSortKey(elem):
    return elem['likesCount']

def writeTxt(item):
    with open('info.txt', 'a', encoding='utf-8')as f:
        f.write("标题：" + item["desc"] + '   ')
        f.write("bgmurl：" + str(item["music"]['play_url']['uri']) + '    ')
        f.write("点赞：" + str(item["statistics"]['digg_count']) + '   ')
        f.write("评论数：" + str(item["statistics"]['comment_count']) + '    ')
        f.write("下载数：" + str(item["statistics"]['download_count']) + '    ')
        f.write("转发数：" + str(item["statistics"]['forward_count']) + '    ')
        f.write("分享数：" + str(item["statistics"]['share_count']) + '\n')

def downloadMusic(musicName):
        request.urlretrieve(str(item["music"]['play_url']['uri']), filename=musicName)
def downloadMusicByClass():
    for i,item in enumerate(bgm1):
        musicName = musicPath1 + '/' + str(i) + '.mp3'
        request.urlretrieve(item, filename=musicName)
    for i,item in enumerate(bgm2):
        musicName = musicPath2 + '/' + str(i) + '.mp3'
        request.urlretrieve(item, filename=musicName)
    for i,item in enumerate(bgm3):
        musicName = musicPath3+ '/' + str(i) + '.mp3'
        request.urlretrieve(item, filename=musicName)

def writeCSVByLikes(datas):
    headers=['path','likesCount']
    with open('sortByLikes.csv', 'w', newline='') as f:
        # 标头在这里传入，作为第一行数据
        writer = csv.DictWriter(f, headers)
        writer.writeheader()
        for row in datas:
            writer.writerow(row)
2
def classifyByLikes(item):
    if (len(str(item["music"]['play_url']['uri'])) > 1):
        if (item["statistics"]['digg_count'] < 50000):
            bgm1.append(str(item["music"]['play_url']['uri']))
        elif (item["statistics"]['digg_count'] > 50000 and item["statistics"]['digg_count'] < 500000):
            bgm2.append(str(item["music"]['play_url']['uri']))
        else:
            bgm3.append(str(item["music"]['play_url']['uri']))
def classifyByComments(item):
    if (len(str(item["music"]['play_url']['uri'])) > 1):
        if (item["statistics"]['comment_count'] < 500):
            bgm1.append(str(item["music"]['play_url']['uri']))
        elif (item["statistics"]['comment_count'] > 500 and item["statistics"]['comment_count'] < 100000):
            bgm2.append(str(item["music"]['play_url']['uri']))
        else:
            bgm3.append(str(item["music"]['play_url']['uri']))

def classifyByShares(item):
    if (len(str(item["music"]['play_url']['uri'])) > 1):
        if (item["statistics"]['share_count'] < 3000):
            bgm1.append(str(item["music"]['play_url']['uri']))
        elif (item["statistics"]['share_count'] > 3000 and item["statistics"]['share_count'] < 50000):
            bgm2.append(str(item["music"]['play_url']['uri']))
        else:
            bgm3.append(str(item["music"]['play_url']['uri']))

if __name__ == "__main__":
    cleanFile()
    f = open(dataPath, 'rb')
    res = f.read()
    data = json.loads(res)
    contain = []
    likes=[]
    for i in range(0,len(data["res"])):
     temp = data["res"][i]
     for item in temp["aweme_list"]:
        musicName = musicPath + '/' + str(item["desc"]) + '.mp3'
        classifyByComments(item)
        #classifyByLikes(item)
        #classifyByShares(item)
        my_dict = {'title': item["desc"], 'BGMurl': str(item["music"]['play_url']['uri']),
                   'likesCount': item["statistics"]['digg_count'], 'commentsCount': item["statistics"]['comment_count'],
                   'downloadCount': item["statistics"]['download_count'],
                   'forwardCount': item["statistics"]['forward_count'], 'shareCount': item["statistics"]['share_count']}
        likes.append(item["statistics"]['digg_count'])
        contain.append(my_dict)
        contain.sort(key=getSortKey)
        writeTxt(item)
    downloadMusicByClass()

