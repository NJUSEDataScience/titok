# _*_ coding: utf-8 _*_
import requests
import sys
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36" }

#去重方法
def distinct_data():
    #读取txt中文档的url列表
    datalist_blank=[]
    pathtxt='H:/Request.txt'
    with open(pathtxt) as f:
        f_data_list=f.readlines()#d得到的是一个list类型
        for a in f_data_list:
            datalist_blank.append(a.strip())#去掉\n strip去掉头尾默认空格或换行符
    data_dict={}
    for data in datalist_blank:
        #url中以/为切分,在以m为切分   ##把m后面的值放进字典key的位置，利用字典特性去重
        if int(data.split('/').index('m'))==4 :#此处为v6开头的url
            #print(data,44,data.split('/')[5])
            data_key1=data.split("/")[5]
            data_dict[data_key1]=data
        elif int(data.split('/').index('m'))==6: #此处为v1或者v3或者v9开头的url
            data_key2=data.split("/")[7]
            data_dict[data_key2] =data
    #print(len(data_dict),data_dict)
    data_new=[]
    for x,y in data_dict.items():
        data_new.append(y)
    return data_new

def responsedouyin():
    data_url=distinct_data()
    #使用request获取视频url的内容
    #stream=True作用是推迟下载响应体直到访问Response.content属性
    #将视频写入文件夹
    num = 1
    for url in data_url:
        res = requests.get(url,stream=True,headers=headers)
        #res = requests.get(url=url, stream=True, headers=headers)
        #定义视频存放的路径
        pathinfo = 'H:/douyin-video/%d.mp4' % num  #%d 用于整数输出   %s用于字符串输出
        #实现下载进度条显示，这一步需要得到总视频大小
        total_size = int(res.headers['Content-Length'])
        #设置流的起始值为0
        temp_size = 0
        if res.status_code == 200:
            with open(pathinfo, 'wb') as file:
                #file.write(res.content)
                #print(pathinfo + '下载完成啦啦啦啦啦')
                num += 1
                #当流下载时，下面是优先推荐的获取内容方式，iter_content()函数就是得到文件的内容，指定chunk_size=1024，大小可以自己设置哟，设置的意思就是下载一点流写一点流到磁盘中
                for chunk in res.iter_content(chunk_size=1024):
                    if chunk:
                        temp_size += len(chunk)
                        file.write(chunk)
                        file.flush() #刷新缓存
                #############下载进度条部分start###############
                        done = int(50 * temp_size / total_size)
                        #print('百分比:',done)
                        sys.stdout.write("\r[%s%s] %d % %" % ('█' * done, ' ' * (50 - done), 100 * temp_size / total_size)+" 下载信息："+pathinfo + "下载完成")
                        sys.stdout.flush()#刷新缓存
                #############下载进度条部分end###############
                print('\n')#每一条打印在屏幕上换行输出


if __name__ == '__main__':
    responsedouyin()