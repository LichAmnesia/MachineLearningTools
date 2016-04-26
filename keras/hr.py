# -*- coding: utf-8 -*-
'''
输入Database下面的hhr和fas文件，以及得到的结果和Swiss-Prot下面现有的匹配的那个bond情况进行匹配
得到的结果放到Output文件夹下面
'''
import os
import re
# filepath = "D:\\Work\\bioinformatics\\hhsearch\\PDBCYS\\"
filepath = "E:\\bioinformatics\\hhsearch\\PDBCYS\\"
datafilepath = filepath + 'Database\\'
# 寻找所有.hhr和fas的文件
def findfile():
    # filepath = os.getcwd()

    s = os.listdir(datafilepath)
    fileres = []
    for filename in s:
        if len(filename.split('.')) > 1 and filename.split('.')[1] == 'seq':
            fileres.append(filename.split('.')[0])
    return fileres
    # return ['Q9H2X3']

# 打开hhr和fas文件
def openfile(filename):
    filename = datafilepath + filename

    hhrfile = open(filename + '.hhr', 'r')
    hhrlist, hhrNum = solvehhr(hhrfile)
    hhrfile.close()
    fasfile = open(filename + '.fas', 'r')
    hhrlist = solvefas(hhrlist, hhrNum, fasfile)
    fasfile.close()
    seqfile = open(filename, 'r')
    # hhrlist增加res
    reslist = compare(hhrlist, hhrNum, seqfile)
    seqfile.close()
    outfile(reslist, hhrNum, filename)

    #print hhrlist

def outfile(reslist, resNum, filename):
    resfile = open(filename + '.score', 'wb')
    for i in range(1, resNum + 1):
        if reslist[i]['Identities'] <= 25:
            resfile.write('>' + reslist[i]['Name'] + '\n')
            resfile.write('Evalue ' + str(reslist[i]['Evalue']) + ' ' + str(reslist[i]['Identities']) + '\n')
            resfile.write(str(len(reslist[i]['Q_Res'])) + '\n')
            for j in range(len(reslist[i]['Q_Res'])):
                a,b = reslist[i]['Q_Res'][j]
                resfile.write(str(a) + ' ' + str(b) + ' ' + '\n')

    resfile.close()

# 与bond进行比较
def compare(hhrlist, hhrNum, seqfile):
    seqfilelines = seqfile.readlines()
    # 对应情况放进[[(Qid,Tid),(Qid,Tid)],[()]]对应i相应T序列
    confiBond = [[] for i in range(hhrNum + 10)]
    for i in range(1, hhrNum + 1):
        Q_Consensus = hhrlist[i]['Q_Consensus']
        Query = hhrlist[i]['Query']
        T_Consensus = hhrlist[i]['T_Consensus']
        Template = hhrlist[i]['Template']
        # Q 位移 T 位移 Fas用-表示
        Q_Cnt = 0
        T_Cnt = 0
        for j in range(0, len(hhrlist[i]['Q_Consensus'])):
            if Q_Consensus[j] == '-' and Query[j] == '-':
                Q_Cnt -= 1
            if T_Consensus[j] == '-' and Template[j] == '-':
                T_Cnt -= 1

            if Q_Consensus[j] == 'C' and Query[j] == 'C' and T_Consensus[j] == 'C' and Template[j] == 'C':
                confiBond[i].append((hhrlist[i]['Q_begin']+ j + Q_Cnt, hhrlist[i]['T_begin'] + j + T_Cnt))
        print "confiBond", confiBond[i]

        ## confiBond直接去取Template序列情况
        T_Name = hhrlist[i]['Name']
        print T_Name
        # 可配置修改
        path = filepath + 'Swiss-Prot\\'
        T_File = open(path + T_Name,'r')
        T_Line = T_File.readline()
        T_Num = int(T_Line)
        T_Dic = {}
        for j in range(T_Num):
            T_Line = T_File.readline()
            first = re.search('[0-9]+', T_Line)
            first = int(first.group(0))
            second = re.search('[0-9]+$', T_Line)
            second = int(second.group(0))
            print first,second,j
            T_Dic[str(first)] = j + 1
            T_Dic[str(second)] = j + 1
        print "T_Dic = ",T_Dic
        #算Q_Dic{"pos":id,"pos":id} id相同则是bond
        Q_Dic = {}
        Q_Res = []
        # 把不出现seq里面的bond对要去掉
        seqNum = int(seqfilelines[0])
        seqDic = {}
        for seqId in range(1, seqNum + 1):
            seqDic[seqfilelines[seqId].split('\n')[0].split(' ')[0]] = 1
            seqDic[seqfilelines[seqId].split('\n')[0].split(' ')[1]] = 1
        print seqDic
        # print hhrlist[i]
        for j in range(len(confiBond[i])):
            for k in range(j + 1, len(confiBond[i])):
                ja,jb = confiBond[i][j]
                ka,kb = confiBond[i][k]
                print ja,jb,ka,kb
                # print T_Dic[str(kb)]
                if not T_Dic.has_key(str(jb)) or (not T_Dic.has_key(str(kb))):
                    continue
                if T_Dic[str(jb)] == T_Dic[str(kb)]:
                    Q_Dic[str(ja)] = T_Dic[str(jb)]
                    Q_Dic[str(ka)] = T_Dic[str(jb)]
                    if ja > ka :
                        ja,ka = ka,ja
                    if seqDic.has_key(str(ja)) and seqDic.has_key(str(ka)):
                        Q_Res.append((ja,ka))

        print str(i) + ' '+ T_Name + ' ' + str(hhrlist[i]['Identities']) + " Q_DIC "+ str(Q_Dic) + str(Q_Res)
        hhrlist[i]['Q_Res'] = Q_Res
        print hhrlist[i]['Q_Res']
    return hhrlist

# 处理fas文件
def solvefas(hhrlist, hhrNum, fasfile):
    # 处理No和序列名称的对应关系
    fasfilelines = fasfile.readlines()
    for i in range(1, hhrNum + 1):
        for fileNum in range(0, len(fasfilelines)):
            if '# No ' + str(i) in fasfilelines[fileNum]:
                fileNum += 2
                # cnt 0-Q_Consensus 1-Query 2-T_Consensus 3-Template
                cnt = 0
                tmpstr = ''
                while (fileNum < len(fasfilelines)) and (not ('# No ' + str(i + 1) in fasfilelines[fileNum])):
                    if '>' in fasfilelines[fileNum]:
                        if cnt == 0:
                            hhrlist[i]['Q_Consensus'] = tmpstr
                        elif cnt == 1:
                            hhrlist[i]['Query'] = tmpstr
                        elif cnt == 2:
                            hhrlist[i]['T_Consensus'] = tmpstr
                        cnt += 1
                        tmpstr = ''
                        fileNum += 1
                    else :
                        tmpstr = tmpstr + fasfilelines[fileNum][:-1]
                        fileNum += 1
                hhrlist[i]['Template'] = tmpstr
                break
    return hhrlist

# 处理hhr文件
def solvehhr(hhrfile):
    # 处理No和序列名称的对应关系，并获取Q和T长度信息
    hhrfilelines = hhrfile.readlines()
    noHitNum = 0
    for fileNum in range(0, len(hhrfilelines)):
        line = hhrfilelines[fileNum]
        if ' No Hit' in line:
            noHitNum = fileNum
            break
    no1Num = 0
    for fileNum in range(0, len(hhrfilelines)):
        line = hhrfilelines[fileNum]
        if 'No 1' in line:
            no1Num = fileNum
            break
    ## hhrlist 指匹配的序列的队列 hhrNum匹配的个数
    hhrlist = [{} for i in range(no1Num - noHitNum)]
    hhrNum = no1Num - noHitNum - 2

    for fileNum in range(noHitNum + 1, no1Num - 1):
        line = hhrfilelines[fileNum]
        Name = line[4:10]
        Id = re.search('[0-9]+', line[0:4])
        Id = int(Id.group())
        Evalue = line[41:49]
        Evalue = float(Evalue)
        Query = line[74:85]
        Q_begin = Query.split('-')[0]
        Q_begin = int(Q_begin)
        Q_end = Query.split('-')[1]
        Q_end = int(Q_end)
        Template = line[85:94]
        T_begin = Template.split('-')[0]
        T_begin = int(T_begin)
        T_end = Template.split('-')[1]
        T_end = int(T_end)
        hhrlist[Id]['Name'] = Name
        hhrlist[Id]['Id'] = Id
        hhrlist[Id]['Evalue'] = Evalue
        hhrlist[Id]['Q_begin'] = Q_begin
        hhrlist[Id]['Q_end'] = Q_end
        hhrlist[Id]['T_begin'] = T_begin
        hhrlist[Id]['T_end'] = T_end

    ## 确定Identity
    for i in range(1, hhrNum + 1):
        for fileNum in range(no1Num - 1, len(hhrfilelines)):
            if '>' + hhrlist[i]['Name'] in hhrfilelines[fileNum]:
                fileNum += 1
                Identities = re.search('Identities=[0-9]+%', hhrfilelines[fileNum])
                Identities = re.search('[0-9]+', Identities.group())
                Identities = int(Identities.group())
                hhrlist[i]['Identities'] = Identities
                break
    ## 返回hhrlist匹配的list，list每个元素是一个字典，并且从1开始和相应Num相同{'Q_begin': 5, 'Name': 'B3A0L5', 'Q_end': 13, 'Evalue': 15.0, 'T_begin': 16, 'Identities': 33, 'Id': 14, 'T_end': 24}
    return hhrlist, hhrNum


def main():
    filelist = findfile()
    for filename in filelist:
        openfile(filename)


if __name__ == '__main__':
    main()
