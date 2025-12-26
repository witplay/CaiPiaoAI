# -*- coding: utf-8 -*-
import copy
import json
import os
import shutil
import sys
from collections import deque as deq

import joblib as job
import numpy as np

from ComClass.Comm_Sql import GetSqlData as mssql


class _Dset(object):
    def __init__(self):
        ...

    sqltitle = ''
    LeftNum = 0  # 前置列数
    OutNum = 0  # 输出列数
    Out_QI = 0  # 输出期数
    InputNum = 0  # 输入列表
    TestCount = 10  # 搞多少测试数据
    WitTestCount = 10  # 继续训练时的测试数量
    GuCount = 400  # 每期量
    Days = 5  # LSTM时间序列数
    fixsql = ""  # 训练数据SQL
    testsql = ""  # 测试数据SQL,如果IsFix=False启用
    lucksql = ""  # 预测数据SQL
    kuaiSelectSql = ""  # 快速录入查询SQL
    kuaiInsertSql = ""  # 快速录入插入SQL
    fitcount = 100  # 训练次数
    IsFix = True  # 是否为训练
    IsFixAllData = False  # 是否为全数据训练
    IsUpset = True  # 是否打乱
    UseSC = True  # 是否使用归一化
    UseOneHot: bool = False  # 是否使用OneHot
    OneHotCount = 33  # 独热类数量
    guid = ''
    AppRunPath = os.path.dirname(os.path.realpath(sys.argv[0]))
    scfile = AppRunPath + '\AiData\StandardFile\save_sc.bin'  # 序列化文件
    ModelFile = AppRunPath + '/AiData/FixModels/'
    guidfile = AppRunPath + '/AiData/mainlist/'
    BestDayFile = AppRunPath + '/AiData/BestDays/'
    SearchModeDo = True  # 是否执行搜索
    SearchModelfile = AppRunPath + '/AiData/SearchModels/'
    SearchModelGuid = ""  # 搜索模型，如果为空进行搜索
    SearchModelMaxTry = 100  # 模型最大搜索次数
    SearchModelPerTry = 2  # 模型每回合测试次数
    SearchModelEpochs = 3000  # 搜索时的训练次数
    SearchModelFitSplit = 0.1  # 搜索时的训练数据与测试拆分比例
    fixCallBackValue = 10  # 训练数据保存点
    textCallBackValue = 30  # 测试数据保存占
    WitFixCallValue = 10  # 连续训练训练保存点
    WitTestCallValue = 10  # 连续训练测试保存点
    sqlview = "select definition from sys.sql_modules where object_id=(select object_id from sys.views where name='{}')"
    sqltable = ""  # 表名
    bestModelFile = ""  # 只测试的模板
    bestModelFile_bak = bestModelFile
    LoadModelFiles = []  # 当前加载的模型文件名的集合
    CurrentModelFile = ''  # 当前使用的模型文件名
    bestFitValue = []  # 训练出的模型最好损失值
    maxFitValue = 0  # 训练最大指标结果
    # mixFitSumValue = 0  # 训练最小合值
    wishFitValue = 0  # 希望训练的最结果
    tfModelType = "lstm"  # 训练使用的模型名称，lstm_l_auto：自动回归，lstm_f_auto：自动分类，lstm_l：回归，lstm_f：分类，rnn，lstm
    loss_name = "mse"  # 失函数名
    metrics_name = "mape"  # 验证损失函数名
    PrintType = "cai"  # 打开类型
    data_intype = np.float64  # 入参类型
    data_outtype = np.float64  # 出参类型
    isClassFix = False  # 是否为二维入参，一般自动模型为True
    valLossSaveMode = 'min'  # 验证损失是大好还是小好
    ClassNameDict = ''  # 分类字典对应的名称
    SaveTest = False  # 是否保存预测数据用于二测预测
    NotInKuiTest = False  # 是否进入快速验证通道

    pass


class _Data(object):
    DsFix = np.array([])  # 训练总数据集
    DsLuck = np.array([])  # 预测总数据集

    DsFix_Left = np.array([])  # 前置数据集
    DsFix_R = np.array([])  # 训练真实
    DsFix_In = np.array([])  # 训练输入
    DsFix_Out = np.array([])  # 训练预测

    DsTest_Left = np.array([])  # 前置数据集
    DsTest_R = np.array([])  # 验证真实
    DsTest_In = np.array([])  # 验证输入
    DsTest_Out = np.array([])  # 验证预测

    DsLuck_Left = np.array([])  # 幸运前置
    DsLuck_R = np.array([])  # 幸运真实
    DsLuck_In = np.array([])  # 幸运输入
    DsLuck_Out = np.array([])  # 幸运预测

    agr_FixShape = DsFix.shape  # 训练空间
    agr_LuckShape = DsLuck.shape  # 预测空间
    agr_InLength = 0  # 输入数据量
    agr_OutLength = 0  # 输出数据量
    agr_TestCount = 0  # 测试数据量
    agr_FixCount = 0  # 训练数据量
    agr_LuckCount = 0  # 幸运数据量


class _BestDaySet(object):
    bestdays = deq(maxlen=5)
    bestval_loss = deq(maxlen=5)
    bestloss = np.array([99999999.0] * 7)


class Fun(object):
    def __init__(self):
        self.bestDaySet = _BestDaySet()
        self.loadBestDaySet = False

        pass

    def SaveTxtFile(obj, path):
        """
        将对象的本文保存成json
        @param path: 保存的路径
        @return:
        """
        w = open(path, 'w+')
        w.write(str(obj))
        w.close()
        pass

    # 保存批次入参
    def SaveGuidFile(dset: _Dset):
        print("保存GUID文件: {0}{1}_{2}.txt".format(dset.guidfile, dset.guid, dset.maxFitValue))
        guidfile = "{0}{1}_{2}.txt".format(dset.guidfile, dset.guid, dset.maxFitValue)
        fromindx = dset.fixsql.index("from") + 4
        whereindx = dset.fixsql.index("where")
        tablename = dset.fixsql[fromindx:whereindx].strip()
        dset.sqltable = tablename
        dset.sqlview = dset.sqlview.format(tablename)       
        # 安全处理SQL视图查询
        try:
            view_result = mssql.getAllData(dset.sqlview)
            if view_result and len(view_result) > 0 and len(view_result[0]) > 0:
                sqlviewtable = str(view_result[0][0]).replace("\r\n", "")
                dset.sqlview = sqlviewtable
        except Exception as e:
            print(f"警告: 查询视图定义失败: {e}，使用默认数据")
        saveds = copy.copy(dset)
        saveds.bestFitValue = str(saveds.bestFitValue)
        saveds.data_intype = str(saveds.data_intype)
        saveds.data_outtype = str(saveds.data_outtype)
        saveds.ClassNameDict = str(saveds.ClassNameDict)
        saveds.maxFitValue = int(saveds.maxFitValue)
        saveds.LoadModelFiles = []
        saveds.CurrentModelFile = ''
        # saveds.maxFitValue = int(saveds.maxFitValue)
        savejs = json.dumps(saveds.__dict__, ensure_ascii=False, indent=4)
        # savejs=savejs.replace(",",",\n")
        w = open(guidfile, 'w+')
        w.write(savejs)
        w.close()
        del saveds
        pass

    def LoadGuidFileAll(dset: _Dset) -> []:
        """
        返回所有GUID的设置
        @return Dset:
        """
        rdss = []

        guidfile = ""
        for root, dirs, ifles in os.walk(dset.guidfile):
            for x in ifles:
                guidfile = "{0}{1}".format(dset.guidfile, str(x))
                if guidfile != "":
                    with open(guidfile, 'r') as fr:
                        rds = copy.copy(dset)
                        rds.__dict__ = json.loads(fr.read())
                        rds.IsFix = False
                        rds.TestCount = dset.TestCount  # 批量测试时的测试数据
                        rds.PrintType = dset.PrintType
                        rds.data_intype = dset.data_intype
                        rds.data_outtype = dset.data_outtype
                        rds.bestModelFile = dset.bestModelFile
                        rds.ClassNameDict = dset.ClassNameDict
                        rds.BestDayFile = dset.BestDayFile
                        rds.SearchModelfile = dset.SearchModelfile
                        rds.guidfile = dset.guidfile
                        rds.ModelFile = dset.ModelFile
                        rds.scfile = dset.scfile
                        if hasattr(rds, "SaveTest"):
                            rds.SaveTest = dset.SaveTest
                        if hasattr(rds, "NotInKuiTest"):
                            rds.NotInKuiTest = dset.NotInKuiTest
                        rdss.append(rds)
                else:
                    print('未找到配置:{}'.format(dset.guid))
        return rdss

    def LoadGuidFile(dset: _Dset):
        """
        根据GUID返回数据设置
        @return Dset:
        """
        if dset.IsFix:
            rds = dset
            return rds
        guidfile = ""
        for root, dirs, ifles in os.walk(dset.guidfile):
            for x in ifles:
                if str(x).count(dset.guid) > 0:
                    guidfile = "{0}{1}".format(dset.guidfile, str(x))
                    break
        if guidfile != "":
            with open(guidfile, 'r') as fr:
                rds = _Dset()
                rds.__dict__ = json.loads(fr.read())
                rds.IsFix = False
                rds.TestCount = dset.TestCount  # 批量测试时的测试数据
                rds.PrintType = dset.PrintType
                rds.kuaiInsertSql = dset.kuaiInsertSql
                rds.kuaiSelectSql = dset.kuaiSelectSql
                # rds.fixsql = dset.fixsql
                # rds.testsql = dset.testsql
                # rds.lucksql = dset.lucksql
                rds.bestModelFile = dset.bestModelFile
                rds.data_intype = dset.data_intype
                rds.data_outtype = dset.data_outtype
                rds.ClassNameDict = dset.ClassNameDict
                rds.BestDayFile = dset.BestDayFile
                rds.SearchModelfile = dset.SearchModelfile
                rds.guidfile = dset.guidfile
                rds.ModelFile = dset.ModelFile
                rds.scfile = dset.scfile
                if hasattr(rds, "SaveTest"):
                    rds.SaveTest = dset.SaveTest
                if hasattr(rds, "NotInKuiTest"):
                    rds.NotInKuiTest = dset.NotInKuiTest
                print('-----------------加载配置成功-------------------')
                # print(rds.__dict__)
        else:
            print('未找到配置:{}'.format(dset.guid))
            Fun.SaveGuidFile(dset)
            print('保存当前配置文件成功！')
            rds = dset
        return rds

    def DelFileUseGuid(guid, dirname):
        """
        通过GUID删除文件
        """
        for root, dirs, ifles in os.walk(dirname):
            for x in dirs:
                if x.index('_') == 0:
                    continue
                fname = str(x).split('_')
                if fname[2] == guid:
                    delf = dirname + x
                    try:
                        shutil.rmtree(delf)
                        # print('删除目录{0}'.format(delf))
                        pass
                    except:
                        print('删除目录{0}失败'.format(delf))
                        pass

            break

    def SaveBestDay(self, dset: _Dset):
        """
        计算存储好的的LSTM天数
        @param dset:
        @return:
        """
        savedayfile = "{}BestDaySet.bin".format(dset.BestDayFile)
        if not self.loadBestDaySet:
            if os.path.exists(savedayfile):
                self.bestDaySet = job.load(savedayfile)
                print("找到最佳天数文件：" + savedayfile)
                self.loadBestDaySet = True
                pass
            else:
                print("找到最佳天数文件：" + savedayfile)
            pass
        bestloss = self.bestDaySet.bestloss
        bestdays = self.bestDaySet.bestdays
        bestval_loss = self.bestDaySet.bestval_loss
        bestFitValue = dset.bestFitValue
        bl = np.array(bestloss).astype(np.float) - np.array(bestFitValue).astype(np.float)
        if max(bl) > 0:
            strday = "Day：{}-最大匹配{},loss:{}".format(dset.Days, dset.maxFitValue, bestFitValue)
            bestdays.append(strday)
            for m in range(7):
                if bestFitValue[m] <= bestloss[m]:
                    if m == 3:
                        bestval_loss.append(strday)
                    bestloss[m] = bestFitValue[m]
            print('-------------当前最好的评估---------------')
            print(np.array(bestdays))
            Fun.SaveTxtFile(bestdays, '{}bestdays.txt'.format(dset.BestDayFile))
            print('-------------当前最好的val_loss---------------')
            print(np.array(bestval_loss))
            Fun.SaveTxtFile(bestval_loss, '{}bestval_loss.txt'.format(dset.BestDayFile))
            print("当前最好Loss:{}".format(bestloss))
            Fun.SaveTxtFile(bestloss, '{}bestloss.txt'.format(dset.BestDayFile))
            self.bestDaySet.bestloss = bestloss
            self.bestDaySet.bestdays = bestdays
            self.bestDaySet.bestval_loss = bestval_loss
            job.dump(self.bestDaySet, savedayfile)
        pass
