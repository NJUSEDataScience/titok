import json
from urllib import request
import csv

musicPath='D:/myMusic'
musicPath1='D:/myMusic/level1' ##用于下载音乐并保存
musicPath2='D:/myMusic/level2'
musicPath3='D:/myMusic/level3'
dataPath='data.json'##获取自动化抓包的数据

def cleanFile():
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

def writeCSVByLikes(datas):
    headers=['path','likesCount']
    with open('sortByLikes.csv', 'w', newline='') as f:
        # 标头在这里传入，作为第一行数据
        writer = csv.DictWriter(f, headers)
        writer.writeheader()
        for row in datas:
            writer.writerow(row)


if __name__ == "__main__":
    cleanFile()
    f = open(dataPath, 'rb')
    res = f.read()
    data = json.loads(res)
    contain = []
    like_datas = []
    likes=[]
    comment=[]
    share=[]
    bgm1=[]
    bgm2=[]
    bgm3=[]
    for i in range(0,len(data["res"])):
     temp = data["res"][i]
     for item in temp["aweme_list"]:
        musicName = musicPath + '/' + str(item["desc"]) + '.mp3'
        #downloadMusic(musicName)
        if (len(str(item["music"]['play_url']['uri']))>1):
            if(item["statistics"]['share_count']<3000):
                bgm1.append(str(item["music"]['play_url']['uri']))
            elif(item["statistics"]['share_count']>3000 and item["statistics"]['share_count']<50000):
                bgm2.append(str(item["music"]['play_url']['uri']))
            else:
                bgm3.append(str(item["music"]['play_url']['uri']))
        my_dict = {'title': item["desc"], 'BGMurl': str(item["music"]['play_url']['uri']),
                   'likesCount': item["statistics"]['digg_count'], 'commentsCount': item["statistics"]['comment_count'],
                   'downloadCount': item["statistics"]['download_count'],
                   'forwardCount': item["statistics"]['forward_count'], 'shareCount': item["statistics"]['share_count']}
        like_dict = {'likesCount': item["statistics"]['digg_count']}
        likes.append(item["statistics"]['digg_count'])
        comment.append(item["statistics"]['comment_count'])
        share.append(item["statistics"]['share_count'])
        like_datas.append(like_dict)
        like_datas.sort(key=getSortKey)
        contain.append(my_dict)
        contain.sort(key=getSortKey)
        writeTxt(item)
    for i,item in enumerate(bgm1):
        musicName = musicPath1 + '/' + str(i) + '.mp3'
        request.urlretrieve(item, filename=musicName)
    for i,item in enumerate(bgm2):
        musicName = musicPath2 + '/' + str(i) + '.mp3'
        request.urlretrieve(item, filename=musicName)
    for i,item in enumerate(bgm3):
        musicName = musicPath3+ '/' + str(i) + '.mp3'
        request.urlretrieve(item, filename=musicName)
    share.sort()
    print(share)
    print(share[30])
    print(share[80])
    writeCSVByLikes(like_datas)
