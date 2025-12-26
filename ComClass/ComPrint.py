import numpy as np

import ComClass.BaseClass as bs
import ComClass.ComFun as comf


def PrintValue_Cai(dset: bs._Dset, TestValue, TrueValue, LeftValue):
    c_Out_c = dset.OutNum * dset.Out_QI  # 输出列数
    # GuCount = dset.GuCount * dset.Days
    # TestValue=tf.reshape(TestValue,(-1,c_Out_c))
    TrueValue = np.reshape(TrueValue, (-1, c_Out_c))
    TestValue = np.array(np.round(TestValue))
    tv = []
    tv1 = []
    slen = TrueValue.shape[0]
    tndim = TestValue.ndim
    vndim = TrueValue.ndim
    if tndim == vndim:
        tv = TestValue
        tv1 = TestValue
    else:
        for sx in range(slen):
            tv.append(TestValue[sx][-2])
            tv1.append(TestValue[sx][-1])
    tv = np.array(tv)
    sumvalue = np.abs(TrueValue - tv)
    tv1 = np.array(tv1)
    sumvalue1 = np.abs(TrueValue - tv1)
    i = 0
    maxval = 0
    pp = [0, 0, 0, 0, 0, 0, 0]
    allp = 0
    for x in LeftValue:
        # n0_0 = len(set(TrueValue[i][0:-1]) & set(tv[i][0:-1]))
        # if TrueValue[i][-1] == tv[i][-1]:
        #     n0_0 += 1
        n0_1 = len(set(TrueValue[i][0:-1]) & set(tv1[i][0:-1]))
        if TrueValue[i][-1] == tv1[i][-1]:
            n0_1 += 1
        allp += n0_1
        # if maxval < n0_0:
        #     maxval = n0_0
        if maxval < n0_1:
            maxval = n0_1
        if n0_1 > 0:
            pp[n0_1 - 1] += 1
        print("-----------------------第{}组------------------------".format(i + 1))
        # print("0:期数:{} \r\n预知:{}\r\n 真实:{} \r\n差值:{}-->合计:{} 匹配{}".format(x, tv[i], TrueValue[i], sumvalue[i],
        #                                                                   sum(sumvalue[i]), n0_0))
        print("-1:期数:{} \r\n预知:{}\r\n 真实:{} \r\n差值:{}-->合计:{} 匹配{}".format(x, tv1[i], TrueValue[i],
                                                                                       sumvalue1[i],
                                                                                       sum(sumvalue1[i]), n0_1))
        comf.TfComFun.SaveCaiTest(dset, x, tv1[i], TrueValue[i])
        i += 1
        pass
    allcount = np.size(TrueValue)
    pl = np.round(allp / allcount, 2)
    print(f"匹配计数：{pp},匹配百分比：{pl}")
    del tv
    del tv1
    del TrueValue
    del TestValue
    return [maxval, '匹配计数统计：', [pp, pl]]


def PrintValue(dset: bs._Dset, TestValue, TrueValue, LeftValue, getLastPrint=False):
    """
    一般打印，每行完全匹配量为最大值
    @param getLastPrint:
    @param dset:
    @param TestValue:
    @param TrueValue:
    @param LeftValue:
    @return:
    """
    c_Out_c = dset.OutNum * dset.Out_QI  # 输出列数
    # GuCount = dset.GuCount * dset.Days
    # TestValue=tf.reshape(TestValue,(-1,c_Out_c))
    TrueValue = np.reshape(TrueValue, (-1, c_Out_c))
    TestValue = np.array(np.round(TestValue))
    tv1 = []
    tsp = TestValue.shape
    vsp = TrueValue.shape
    tndim = TestValue.ndim
    vndim = TrueValue.ndim
    slen = TrueValue.shape[0]
    # clen = TrueValue.shape[1]
    if tndim == vndim:
        tv1 = np.array(TestValue)
    else:
        for sx in range(slen):
            tv1.append(TestValue[sx][-1])
        tv1 = np.array(tv1)

    sumvalue1 = np.abs(TrueValue - tv1)
    i = 0
    maxval = 0
    rval_c = 0
    lastprit = ""
    pp = [0, 0, 0, 0, 0, 0, 0]
    for x in LeftValue:
        n0_1 = np.sum(sumvalue1[i] == 0)
        if maxval < n0_1:
            maxval = n0_1
        if n0_1 == vsp[-1]:
            rval_c += 1
        if n0_1 > 0:
            pp[n0_1 - 1] += 1
        print("-----------------------第{}组------------------------".format(i + 1))
        # print("0:期数:{} \r\n预知:{}\r\n 真实:{} \r\n差值:{}-->合计:{} 匹配{}".format(x, tv[i], TrueValue[i], sumvalue[i],
        #                                                                   sum(sumvalue[i]), n0_0))
        print("-1:期数:{} \r\n预知:{}\r\n 真实:{} \r\n差值:{}-->合计:{} 匹配{}".format(x, tv1[i], TrueValue[i],
                                                                                       sumvalue1[i],
                                                                                       sum(sumvalue1[i]), n0_1))
        comf.TfComFun.SaveCaiTest(dset, x, tv1[i], TrueValue[i])  # 保存预测数据
        if getLastPrint:
            lastprit = f"预知:{tv1[i]}{GetSzClassName(tv1[i])}"
        i += 1
        pass
    rval_c = rval_c / i
    print(f'最大匹配：{maxval}，中标百分比{rval_c:.2f}')
    print(f"匹配计数：{pp}")
    del tv1
    del TrueValue
    del TestValue
    if getLastPrint:
        return rval_c, lastprit
    else:
        return [maxval, '匹配计数统计：', [pp, rval_c]]


def PrintValue_Class(dset: bs._Dset, TestValue, TrueValue, LeftValue, getLastPrint=False):
    """
      一般打印，每行完全匹配量为最大值
      @param getLastPrint:
      @param dset:
      @param TestValue:
      @param TrueValue:
      @param LeftValue:
      @return:
      """
    c_Out_c = dset.OutNum * dset.Out_QI  # 输出列数
    # GuCount = dset.GuCount * dset.Days
    # TestValue=tf.reshape(TestValue,(-1,c_Out_c))
    if dset.UseOneHot:
        TrueValue = np.argmax(TrueValue, axis=1, keepdims=True)  # 获取每行最大值的索引，保持原数组形状
    TrueValue = np.reshape(TrueValue, (-1, c_Out_c))
    TestValue = np.array(TestValue)
    tsp = TestValue.shape
    vsp = TrueValue.shape
    tndim = TestValue.ndim
    vndim = TrueValue.ndim
    tv1 = np.argmax(TestValue, axis=1, keepdims=True)  # 获取每行最大值的索引，保持原数组形状
    sumvalue1 = np.abs(TrueValue - tv1)
    i = 0
    maxval = 0
    rval_c = 0
    for x in LeftValue:
        n0_1 = np.sum(sumvalue1[i] == 0)
        if maxval < n0_1:
            maxval = n0_1
        if n0_1 == vsp[-1]:
            rval_c += 1
        print("-----------------------第{}组------------------------".format(i + 1))
        testClassName = ''
        TrueClassName = ''
        if dset.ClassNameDict != '':
            testClassName = dset.ClassNameDict[tv1[i][0]]
            TrueClassName = dset.ClassNameDict[TrueValue[i][0]]

        print(
            f"-1:期数:{x} \r\n预知:{tv1[i]}{testClassName}\r\n 真实:{TrueValue[i]}{TrueClassName} \r\n差值:{sumvalue1[i]}-->合计:{sum(sumvalue1[i])} 匹配{n0_1}")
        i += 1
        pass
    rval_c = rval_c / i
    print(f'最大匹配：{maxval}，中标百分比{rval_c:.2f}')
    del tv1
    del TrueValue
    del TestValue
    return rval_c
    pass


def GetClassName(dset: bs._Dset, invalue):
    pass


def PrintValue_KCai(dset: bs._Dset, TestValue, TrueValue, LeftValue, getLastPrint=False):
    """
      一般打印，每行完全匹配量为最大值
      @param getLastPrint:
      @param dset:
      @param TestValue:
      @param TrueValue:
      @param LeftValue:
      @return:
      """
    c_Out_c = dset.OutNum * dset.Out_QI  # 输出列数
    # GuCount = dset.GuCount * dset.Days
    # TestValue=tf.reshape(TestValue,(-1,c_Out_c))
    TrueValue = np.reshape(TrueValue, (-1, c_Out_c))
    TestValue = np.array(TestValue)
    tsp = TestValue.shape
    vsp = TrueValue.shape
    tndim = TestValue.ndim
    vndim = TrueValue.ndim
    tv1 = []
    if tndim == vndim:
        if tsp[-1] > 1:
            TestValue = np.argmax(TestValue, axis=1, keepdims=True)  # 获取每行最大值的索引，保持原数组形状
            tv1 = TestValue
        else:
            tv1 = np.round(TestValue)
    else:
        TestValue = np.round(TestValue)
        slen = TrueValue.shape[0]
        # clen = TrueValue.shape[1]
        for sx in range(slen):
            tv1.append(TestValue[sx][-1])
        tv1 = np.array(tv1)

    sumvalue1 = np.abs(TrueValue - tv1)
    i = 0
    maxval = 0
    rval_c = 0
    lastprit = ""
    for x in LeftValue:
        n0_1 = np.sum(sumvalue1[i] == 0)
        if maxval < n0_1:
            maxval = n0_1
        if n0_1 == vsp[-1]:
            rval_c += 1
        print("-----------------------第{}组------------------------".format(i + 1))
        # print("0:期数:{} \r\n预知:{}\r\n 真实:{} \r\n差值:{}-->合计:{} 匹配{}".format(x, tv[i], TrueValue[i], sumvalue[i],
        #                                                                   sum(sumvalue[i]), n0_0))
        print(
            f"-1:期数:{x} \r\n预知:{tv1[i]}{GetSzClassName(tv1[i])}\r\n 真实:{TrueValue[i]}{GetSzClassName(TrueValue[i])} \r\n差值:{sumvalue1[i]}-->合计:{sum(sumvalue1[i])} 匹配{n0_1}")
        lastprit = f"预知:{tv1[i]}{GetSzClassName(tv1[i])}"
        i += 1
        pass
    rval_c = rval_c / i
    print(f'最大匹配：{maxval}，中标百分比{rval_c:.2f}')
    del tv1
    del TrueValue
    del TestValue
    if getLastPrint:
        return rval_c, lastprit
    else:
        return rval_c
    pass


def PrintValue_cup(dset: bs._Dset, TestValue, TrueValue, LeftValue):
    """
    一般打印，每行完全匹配量为最大值
    @param dset:
    @param TestValue:
    @param TrueValue:
    @param LeftValue:
    @return:
    """
    c_Out_c = dset.OutNum * dset.Out_QI  # 输出列数
    # GuCount = dset.GuCount * dset.Days
    # TestValue=tf.reshape(TestValue,(-1,c_Out_c))
    TrueValue = np.reshape(TrueValue, (-1, c_Out_c))
    TestValue = np.array(np.round(TestValue))
    tv1 = []
    slen = TrueValue.shape[0]
    for sx in range(slen):
        tv1.append(TestValue[sx][-1])
    tv1 = np.array(tv1)
    sumvalue1 = np.abs(TrueValue - tv1)
    i = 0
    maxval = 0
    rval = 0
    rval_c = 0
    for x in LeftValue:
        n0_1 = np.sum(sumvalue1[i] == 0)
        rjg = '平'
        if TrueValue[i][0] > TrueValue[i][1]:
            rjg = '胜'
        if TrueValue[i][0] < TrueValue[i][1]:
            rjg = '负'
        tjg = '平'
        if tv1[i][0] > tv1[i][1]:
            tjg = '胜'
        if tv1[i][0] < tv1[i][1]:
            tjg = '负'
        zb = '未中'
        if rjg == tjg:
            zb = '中标'
            rval += 1
        if maxval < n0_1:
            maxval = n0_1
        if n0_1 == 2:
            rval_c += 1
        print("-----------------------第{}组------------------------".format(i + 1))
        print(
            f'期数:{x} \r\n预知:{tv1[i]}-{tjg}\r\n真实:{TrueValue[i]}-{rjg} \r\n差值:{sumvalue1[i]}-->合计:{sum(sumvalue1[i])} 匹配{n0_1} 结果:{zb}')
        i += 1
        pass
    rval_c = rval_c / i
    rval = rval / i
    print(f'最大匹配：{maxval}，比分中标百分比{rval_c:.2f}，结果中标百分比：{rval:.2f}')
    del tv1
    del TrueValue
    del TestValue
    return (rval + rval_c) / 2


def PrintValue_cup_z(dset: bs._Dset, TestValue, TrueValue, LeftValue):
    """
    一般打印，每行完全匹配量为最大值
    @param dset:
    @param TestValue:
    @param TrueValue:
    @param LeftValue:
    @return:
    """
    c_Out_c = dset.OutNum * dset.Out_QI  # 输出列数
    # GuCount = dset.GuCount * dset.Days
    # TestValue=tf.reshape(TestValue,(-1,c_Out_c))
    TrueValue = np.reshape(TrueValue, (-1, c_Out_c))
    TestValue = np.array(np.round(TestValue))
    tv1 = []
    slen = TrueValue.shape[0]
    for sx in range(slen):
        tv1.append(TestValue[sx][-1])
    tv1 = np.array(tv1)
    sumvalue1 = np.abs(TrueValue - tv1)
    i = 0
    maxval = 0
    rval_c = 0
    for x in LeftValue:
        n0_1 = np.sum(sumvalue1[i] == 0)
        if maxval < n0_1:
            maxval = n0_1
        if n0_1 == 1:
            rval_c += 1
        print("-----------------------第{}组------------------------".format(i + 1))
        print(
            f'期数:{x} \r\n预知:{tv1[i]}\r\n真实:{TrueValue[i]} \r\n差值:{sumvalue1[i]}-->合计:{sum(sumvalue1[i])} 匹配{n0_1}')
        i += 1
        pass
    rval_c = rval_c / i
    print(f'最大匹配：{maxval}，球数中标百分比{rval_c:.2f}')
    del tv1
    del TrueValue
    del TestValue
    return rval_c


def PrintValue_gu(dset: bs._Dset, TestValue, TrueValue, LeftValue):
    """
    股票打印，以小于估算值小于X，总成功比率
    @param dset:
    @param TestValue:
    @param TrueValue:
    @param LeftValue:
    @return:
    """
    c_Out_c = dset.OutNum * dset.Out_QI  # 输出列数
    # GuCount = dset.GuCount * dset.Days
    # TestValue=tf.reshape(TestValue,(-1,c_Out_c))
    TrueValue = np.reshape(TrueValue, (-1, c_Out_c))
    TestValue = np.array(TestValue)
    tv1 = []
    slen = TrueValue.shape[0]
    for sx in range(slen):
        tv1.append(TestValue[sx][-1])
    tv1 = np.array(tv1)
    sumvalue1 = np.abs(TrueValue - tv1)
    i = 0
    maxval = 0
    rightval = 0
    for x in LeftValue:
        n0_1 = np.sum(sumvalue1[i] < 1.5)
        if n0_1 > 0:
            rightval += n0_1
        print("-----------------------第{}组------------------------".format(i + 1))
        print(
            f'-1:期数:{x} \r\n预知:{tv1[i]}\r\n 真实:{TrueValue[i]} \r\n差值:{sumvalue1[i][0]}-->合计:{sum(sumvalue1[i])} 匹配{n0_1}')
        i += 1
        pass
    allcount = np.size(TrueValue)
    maxval = rightval / allcount
    print(f'评估值：{maxval}')
    del tv1
    del TrueValue
    del TestValue
    return maxval


def PrintLuck_Cai(dset: bs._Dset, TestValue, TrueValue, LeftValue):
    c_Out_c = dset.OutNum * dset.Out_QI  # 输出列数
    GuCount = dset.GuCount * dset.Days
    # TestValue=tf.reshape(TestValue,(-1,c_Out_c))
    TrueValue = np.reshape(TrueValue, (-1, c_Out_c))
    TestValue = np.array(np.round(TestValue))
    tv = []
    tv1 = []
    slen = TrueValue.shape[0]
    tndim = TestValue.ndim
    vndim = TrueValue.ndim
    if tndim == vndim:
        tv = TestValue
        tv1 = TestValue
    else:
        for sx in range(slen):
            tv.append(TestValue[sx][0])
            tv1.append(TestValue[sx][-1])
    tv = np.array(tv)
    tv1 = np.array(tv1)
    i = 0
    # print("-----------------------第{}期------------------------".format(LeftValue[0]))
    for x in LeftValue:
        # print("-----------------------第{}------------------------".format(i+1))
        # print("{}".format(tv[i]))
        print("{}".format(tv1[i]))
        comf.TfComFun.SaveCaiLuck(dset, x, tv1[i])
        i += 1
        pass
    del tv
    # del tv1
    del TrueValue
    del TestValue
    return tv1
    pass


def PrintLuck_Class(dset: bs._Dset, TestValue, TrueValue, LeftValue):
    c_Out_c = dset.OutNum * dset.Out_QI  # 输出列数
    GuCount = dset.GuCount * dset.Days
    # TestValue=tf.reshape(TestValue,(-1,c_Out_c))
    TrueValue = np.reshape(TrueValue, (-1, c_Out_c))
    tsp = TestValue.shape
    vsp = TrueValue.shape
    tndim = TestValue.ndim
    vndim = TrueValue.ndim
    tv1 = np.argmax(TestValue, axis=1, keepdims=True)  # 获取每行最大值的索引，保持原数组形状
    i = 0
    # print("-----------------------第{}期------------------------".format(LeftValue[0]))
    result = []
    for x in LeftValue:
        # print("-----------------------第{}------------------------".format(i+1))
        # print("{}".format(tv[i]))
        tvchar = ''
        if dset.ClassNameDict != '':
            tvchar = dset.ClassNameDict[tv1[i][0]]
        result.append([LeftValue[i], tv1[i][0], tvchar])
        print(f"{LeftValue[i]}\r\n{tv1[i][0]}:{tvchar}")
        i += 1
        pass
    # del tv
    del tv1
    del TrueValue
    del TestValue
    result = sorted(result, key=lambda rr: rr[1], reverse=True)
    return result
    pass


def PrintLuck_KCai(dset: bs._Dset, TestValue, TrueValue, LeftValue):
    c_Out_c = dset.OutNum * dset.Out_QI  # 输出列数
    GuCount = dset.GuCount * dset.Days
    # TestValue=tf.reshape(TestValue,(-1,c_Out_c))
    TrueValue = np.reshape(TrueValue, (-1, c_Out_c))
    tsp = TestValue.shape
    vsp = TrueValue.shape
    tndim = TestValue.ndim
    vndim = TrueValue.ndim
    tv1 = []
    if tndim == vndim:
        if tsp[-1] > 1:
            TestValue = np.argmax(TestValue, axis=1, keepdims=True)  # 获取每行最大值的索引，保持原数组形状
            tv1 = TestValue
        else:
            tv1 = np.round(TestValue)
    else:
        TestValue = np.round(TestValue)
        # tv = []
        slen = TrueValue.shape[0]
        for sx in range(slen):
            # tv.append(TestValue[sx][0])
            tv1.append(TestValue[sx][-1])
        # tv = np.array(tv)
        tv1 = np.array(tv1)
    i = 0
    # print("-----------------------第{}期------------------------".format(LeftValue[0]))
    for x in LeftValue:
        # print("-----------------------第{}------------------------".format(i+1))
        # print("{}".format(tv[i]))
        tvchar = [GetSzClassName(tv1[i][0])]
        print("{}:{}".format(tv1[i], tvchar))
        i += 1
        pass
    # del tv
    del tv1
    del TrueValue
    del TestValue
    pass


def GetSzClassName(invalue, classcount=4, classtype=0):
    outname = ''
    if classcount > 4:
        if invalue == 0:
            outname = ['小', '单']
        elif invalue == 1:
            outname = ['小', '双']
        elif invalue == 2:
            outname = ['大', '单']
        elif invalue == 3:
            outname = ['大', '双']
        else:
            outname = ['豹子', '豹子']
    else:
        if classtype == 1:
            if invalue == 0:
                outname = ['小']
            elif invalue == 1:
                outname = ['大']
        else:
            if invalue == 0:
                outname = ['单']
            elif invalue == 1:
                outname = ['双']
    return outname


def PrintLuck_gu(dset: bs._Dset, TestValue, TrueValue, LeftValue):
    """
    股票预测打印
    @param dset:
    @param TestValue:
    @param TrueValue:
    @param LeftValue:
    @return:
    """
    c_Out_c = dset.OutNum * dset.Out_QI  # 输出列数
    GuCount = dset.GuCount * dset.Days
    # TestValue=tf.reshape(TestValue,(-1,c_Out_c))
    TrueValue = np.reshape(TrueValue, (-1, c_Out_c))
    TestValue = np.array(TestValue)
    tv1 = []
    slen = TrueValue.shape[0]
    for sx in range(slen):
        tv1.append([LeftValue[sx], TestValue[sx][-1]])
    for p in tv1:
        print('{}:{}'.format(p[0], np.around(p[1], 2)))
    del tv1
    del TrueValue
    del TestValue
    pass


def PrintLuck_cup(dset: bs._Dset, TestValue, TrueValue, LeftValue):
    """
    股票预测打印
    @param dset:
    @param TestValue:
    @param TrueValue:
    @param LeftValue:
    @return:
    """
    c_Out_c = dset.OutNum * dset.Out_QI  # 输出列数
    GuCount = dset.GuCount * dset.Days
    # TestValue=tf.reshape(TestValue,(-1,c_Out_c))
    TrueValue = np.reshape(TrueValue, (-1, c_Out_c))
    TestValue = np.array(TestValue)
    tv1 = []
    slen = TrueValue.shape[0]
    for sx in range(slen):
        tv1.append([LeftValue[sx], TestValue[sx][-1]])
    for p in tv1:
        rs = '平'
        if np.round(p[1][0]) > np.round(p[1][1]):
            rs = '胜'
        if np.round(p[1][0]) < np.round(p[1][1]):
            rs = '负'
        print('{}:{}->{}{}'.format(p[0], np.around(p[1], 2), np.round(p[1]), rs))
    del tv1
    del TrueValue
    del TestValue
    pass


def PrintLuck_cup_z(dset: bs._Dset, TestValue, TrueValue, LeftValue):
    """
    股票预测打印
    @param dset:
    @param TestValue:
    @param TrueValue:
    @param LeftValue:
    @return:
    """
    c_Out_c = dset.OutNum * dset.Out_QI  # 输出列数
    TrueValue = np.reshape(TrueValue, (-1, c_Out_c))
    TestValue = np.array(TestValue)
    tv1 = []
    slen = TrueValue.shape[0]
    for sx in range(slen):
        tv1.append([LeftValue[sx], TestValue[sx][-1]])
    for p in tv1:
        print('{}:{}->{}'.format(p[0], np.around(p[1], 2), np.round(p[1])))
    del tv1
    del TrueValue
    del TestValue
    pass


"""
打印测试入口
"""


def PrintTestFun(dset: bs._Dset, TestValue, TrueValue, LeftValue, getlastprint=False):
    resout = 0
    # 加入其他返回值数组，第一个是最大值，第二个是其他说明，第三个其他值集合
    if dset.PrintType == "cai":
        resout = PrintValue_Cai(dset, TestValue, TrueValue, LeftValue)
    if dset.PrintType in ("print", 'cai3d'):
        resout = PrintValue(dset, TestValue, TrueValue, LeftValue, getlastprint)
    if dset.PrintType == "kcai3d":
        resout = PrintValue_KCai(dset, TestValue, TrueValue, LeftValue, getlastprint)
    if dset.PrintType == 'printgu':
        resout = PrintValue_gu(dset, TestValue, TrueValue, LeftValue)
    if dset.PrintType == 'printguclass':
        resout = PrintValue_Class(dset, TestValue, TrueValue, LeftValue)
    if dset.PrintType == 'cup':
        resout = PrintValue_cup(dset, TestValue, TrueValue, LeftValue)
    if dset.PrintType == 'cup_z':
        resout = PrintValue_cup_z(dset, TestValue, TrueValue, LeftValue)
    return resout
    pass


"""
打印预测入口
"""


def PrintLuckFun(dset: bs._Dset, TestValue, TrueValue, LeftValue):
    if dset.PrintType in ("cai", 'cai3d'):
        return PrintLuck_Cai(dset, TestValue, TrueValue, LeftValue)
    if dset.PrintType == 'kcai3d':
        PrintLuck_KCai(dset, TestValue, TrueValue, LeftValue)
    if dset.PrintType == 'printgu':
        PrintLuck_gu(dset, TestValue, TrueValue, LeftValue)
    if dset.PrintType == 'printguclass':
        pout = PrintLuck_Class(dset, TestValue, TrueValue, LeftValue)
        return pout
    if dset.PrintType in 'cup':
        PrintLuck_cup(dset, TestValue, TrueValue, LeftValue)
    if dset.PrintType == 'cup_z':
        PrintLuck_cup_z(dset, TestValue, TrueValue, LeftValue)
    pass
