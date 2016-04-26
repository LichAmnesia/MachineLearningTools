# -*- coding: utf-8 -*-
'''
seqConfidentDic表示目前的最多匹配的confident bond序列信息{
    'P13726': [(49, 57), (182, 205)],
    'P13725': [(124, 144)],
    'Q53IQ4': [(271, 316)],
    'P04745': [],
}
'''
import os
import re
filepath = "D:\\Work\\bioinformatics\\hhsearch\\PDBCYS\\"
# filepath = "E:\\bioinformatics\\hhsbearch\\PDBCYS\\"
datafilepath = filepath + 'Database\\'
outfilepath = filepath + 'Output\\'
predictpath = filepath + 'Predictout\\'
setpath = filepath + 'Set\\'
# 寻找所有.hhr和fas的文件
def findfile():
    # filepath = os.getcwd()
    s = os.listdir(datafilepath)
    fileres = []
    for filename in s:
        if len(filename.split('.')) > 1 and filename.split('.')[1] == 'seq':
            fileres.append(filename.split('.')[0])
    return fileres


# seq和score文件
def openfile(seqConfidentDic, filename):
    seqfile = open(datafilepath + filename, 'r')
    # hhrlist增加res
    scorefile = open(outfilepath + filename + '.score', 'r')
    reslist = compare(seqConfidentDic, filename, scorefile, seqfile)
    seqfile.close()
    scorefile.close()


# 读取score文件到list[] ，每个是一个dic{'seqName': 'Q9Y6Y9', 'Name': 'Q60648', 'list': [(21, 132), (79, 89)], 'Num': 2, 'E-value': 1e-19, 'Identities': 17}
def scorelist(filename, scorefilelines):
    res = []
    resNum = 0
    for fileNum in range(0, len(scorefilelines)):
        tmpscore = {}
        if '>' in scorefilelines[fileNum]:
            tmpscore['seqName'] = filename
            tmpscore['Name'] = scorefilelines[fileNum][1:-1]
            fileNum += 1
            tmplist = scorefilelines[fileNum].split(' ')
            tmpscore['E-value'] = float(tmplist[1])
            tmpscore['Identities'] = int(tmplist[2])
            fileNum += 1
            tmpNum = int(scorefilelines[fileNum])
            tmpscore['Num'] = tmpNum
            tmpscore['list'] = []
            fileNum += 1
            for i in range(tmpNum):
                tmplist = scorefilelines[fileNum].split(' ')
                tmplist[0] = int(tmplist[0])
                tmplist[1] = int(tmplist[1])
                tmpscore['list'].append((tmplist[0],tmplist[1]))
                fileNum += 1
            # print tmpscore
            res.append(tmpscore)
    resNum = len(res)
    return res,resNum

# 与bond进行比较
def compare(seqConfidentDic, filename, scorefile, seqfile):

    scorefilelines = scorefile.readlines()
    scoreRes,scoreNum = scorelist(filename, scorefilelines)
    MaxNum = -1
    MaxId = -1
    for scoreId in range(scoreNum):
        if scoreRes[scoreId]['Num'] > MaxNum:
            MaxId = scoreId
            MaxNum = scoreRes[scoreId]['Num']
    # 所有匹配结果进行或运算得到
    scoreSet = set([])
    for scoreId in range(scoreNum):
        tmpset = set(scoreRes[scoreId]['list'])
        scoreSet = tmpset | scoreSet
    # print "scoreset = ",scoreSet
    # print "MaxscoreRes = ", scoreRes[MaxId]['list']
    # 取Num最大的那个作为匹配对象
    scoreResSet = set(scoreRes[0]['list'])

    # 读取seqfile的信息
    seqfilelines = seqfile.readlines()
    seqNum = int(seqfilelines[0])
    seqDic = {}
    seqSet = set([])
    for seqId in range(1, seqNum + 1):
        a = int(seqfilelines[seqId].split('\n')[0].split(' ')[0])
        b = int(seqfilelines[seqId].split('\n')[0].split(' ')[1])
        seqSet = set([(a,b)]) | seqSet
    # print filename, " seq = ",seqSet,len(seqSet)
    # print "resseq = ", scoreRes[MaxId]['list'], len(scoreRes[MaxId]['list'])

    seqConfidentDic[filename] = scoreResSet
    return
    # 正确的数目
    scoreCnt = 0
    seqSum = len(scoreResSet)
    for i in scoreSet:
        if i in seqSet:
            scoreCnt += 1
        else:
            continue
            # print filename,scoreResSet,seqSet,i
    print scoreCnt
    # print ""
    # print scoreRes,scoreNum
    return
    # 对应情况放进[[(Qid,Tid),(Qid,Tid)],[()]]对应i相应T序列

# 读取set信息
def readset(setId):
    setFile = open(setpath + 'set' + str(setId))
    setTable = [0,0,6,15,28,45]
    seqDic = {}
    bondNumDic = {}
    seqNameArray = []
    for setFileLine in setFile.readlines():
        if setFileLine[0] == '>':
            # 每个C的对应
            tmpNumDic = {}
            tmpNum = 0
            proteinName = setFileLine[1:-1]
            seqNameArray.append(proteinName)
            # 读取seqfile的信息
            seqfile = open(datafilepath + proteinName,'r')
            seqfilelines = seqfile.readlines()
            seqNum = int(seqfilelines[0])
            seqSet = set([])
            for seqId in range(1, seqNum + 1):
                a = int(seqfilelines[seqId].split('\n')[0].split(' ')[0])
                tmpNum += 1
                tmpNumDic[str(a)] = tmpNum
                b = int(seqfilelines[seqId].split('\n')[0].split(' ')[1])
                tmpNum += 1
                tmpNumDic[str(b)] = tmpNum
                seqSet = set([(a,b)]) | seqSet
            seqDic[proteinName] = seqSet
            bondNumDic[proteinName] = tmpNumDic
    return seqNameArray, seqDic, bondNumDic

# 初始化Bond对应表比如4个bond对应28个key，'12':0这样
def initPairNum():
    pairDic = [{}]
    pairNum = [0]
    for pair in range(1, 11):
        tmpDic = {}
        cnt = 0
        for i in range(1, pair * 2):
            for j in range(i + 1, pair * 2 + 1):
                tmpDic[str(i) + str(j)] = cnt
                cnt += 1
        pairDic.append(tmpDic)
        pairNum.append(len(tmpDic))
        # print pairNum[pair],pairLen[pair]
    return pairDic,pairNum

# 读取output文件
def readpredictoutput(setId):
    predictFile = open(predictpath + 'output_' + str(setId),'r')
    predictOut = []
    for predictLine in predictFile.readlines():
        predictOut.append(float(predictLine.split('\n')[0].split(' ')[0]))
        # b = float(predictLine.split('\n')[0].split(' ')[1])
        # print a,b
    return predictOut

# num 表示bond对数目，vis是访问数组，now表示
def dfs(seqPredictOut,pairDicSeqBondNum,num,now,vis,ans,li,ali):
    table = [0,0,6,15,28,45]
    if now >= num:
        ret = 0
        for i in range(num):
            ret += seqPredictOut[pairDicSeqBondNum[li[i]]]
        return ret, li
    for i in range(num * 2):
        if vis[i] == 0:
            for j in range(i + 1, num * 2):
                if vis[j] == 0:
                    vis[i] = 1
                    vis[j] = 1
                    li.append(str(i + 1)+str(j + 1))
                    ret,rli = dfs(seqPredictOut,pairDicSeqBondNum,num,now + 1,vis,ans,li,ali)
                    if ans < ret:
                        ali = []
                        for l in range(len(rli)):
                            ali.append(rli[l])
                        ans = ret
                                # ali = rli
                    li.pop()
                    vis[i] = 0
                    vis[j] = 0
    return ans,ali

# 单独的一个seq进行最大权匹配的问题
def solveseqpredict(seqBondNum, pairDicSeqBondNum, seqPredictOut):
    vis = [0 for i in range(20)]
    li = []
    ali = []
    ans = -1000
    # print pairDicSeqBondNum,seqBondNum,0,vis,ans,li,ali
    ret,ali = dfs(seqPredictOut,pairDicSeqBondNum,seqBondNum,0,vis,ans,li,ali)
    print ret,ali
    ali = sorted(ali)
    now = 0
    for i in range(seqBondNum):
        if ali[i] == str(2*i + 1) + str(2*i + 2):
            now += 1
    print seqBondNum,now
    if now < seqBondNum:
        return 0,now
    else:
        return 1,now
    print seqPredictOut,len(seqPredictOut)



def main():
    pairDic,pairNum = initPairNum()
    filelist = findfile()
    seqConfidentDic = {}
    for filename in filelist:
        openfile(seqConfidentDic, filename)
    # 按照set_1-20进行最大权匹配
    answerSeq = 0
    answerBond = 0
    answerSumSeq = 0
    answerSumBond = 0
    for setId in range(1,21):
        seqNameArray, seqDic, bondNumDic = readset(setId)
        pridictOut = readpredictoutput(setId)
        predictNowLine = 0
        # 每个seq要单独算

        for seqName in seqNameArray:
            seqBond = seqDic[seqName]
            seqBondNum = len(seqBond)
            seqPredictOut = []
            for i in range(pairNum[seqBondNum]):
                seqPredictOut.append(pridictOut[predictNowLine])
                predictNowLine += 1
            # 找到confident bond并加上去
            for bondPair in seqConfidentDic[seqName]:
                a,b = bondPair
                a = bondNumDic[seqName][str(a)]
                b = bondNumDic[seqName][str(b)]
                if a > b:
                    a,b = b,a
                pairStr = str(a) + str(b)
                if not (pairStr in pairDic[seqBondNum].keys()):
                    print pairStr,a,b,bondNumDic[seqName],seqDic[seqName]
                    print seqConfidentDic[seqName]
                seqPredictOut[pairDic[seqBondNum][pairStr]] += 1
                for i in range(1, seqBondNum * 2 + 1):
                    tmpstr = str(a) + str(i)
                    if tmpstr in pairDic[seqBondNum].keys() and i != b:
                        seqPredictOut[pairDic[seqBondNum][tmpstr]] -= 1
                    tmpstr = str(i) + str(a)
                    if tmpstr in pairDic[seqBondNum].keys():
                        seqPredictOut[pairDic[seqBondNum][tmpstr]] -= 1
                    tmpstr = str(b) + str(i)
                    if tmpstr in pairDic[seqBondNum].keys():
                        seqPredictOut[pairDic[seqBondNum][tmpstr]] -= 1
                    tmpstr = str(i) + str(b)
                    if tmpstr in pairDic[seqBondNum].keys() and i != a:
                        seqPredictOut[pairDic[seqBondNum][tmpstr]] -= 1
            isSeqTrue, seqTrueNum = solveseqpredict(seqBondNum, pairDic[seqBondNum], seqPredictOut)
            answerSeq += isSeqTrue
            answerBond += seqTrueNum
            answerSumSeq += 1
            answerSumBond += seqBondNum
    print answerSeq,answerBond
    print answerSumSeq,answerSumBond
    print float(answerSeq) / answerSumSeq
    print float(answerBond) / answerSumBond


if __name__ == '__main__':
    main()
