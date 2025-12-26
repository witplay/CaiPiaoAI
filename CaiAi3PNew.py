import gc
import os
import numpy as np

from RMB888.WLAi02 import WLAi as u
from collections import deque as deq
from ComClass.BaseClass import Fun as cf
from ComClass.Comm_Sql import GetSqlData as mssql
import copy


# 体彩排列3预测
def caiscq():
    """
    体彩排列3预测
    @return:
    """
    try:
        saveDir = 'AiCaiData3PNew'
        sup = u()
        runPath = sup.dset.AppRunPath + '/SaveModels'
        # sup.dset.bestModelFile = f"{runPath}/{saveDir}/FixModels/" \
        #                          f"A20231213@015958_2.7133@0.0907@18.8281@0.3556_43a8acf9991811ee9c1a3c7c3f7f228f"
        # A20231126@083410_0.1018@0.7844@8.0391@0.1667_83e7b39d8bf311ee81cc3c7c3f7f228f #指定模型
        sup.dset.guid = 'all'  # 如果为all时所有都测试，
        sup.dset.IsFix = False
        fixfor = 10  # 循环训练几次
        sup.dset.isClassFix =  True
        sup.dset.SaveTest = False
        sup.dset.valLossSaveMode = 'max' # max min
        # 是否训练
        sup.dset.IsFixAllData = False
        sup.dset.tfModelType = "l_cai_hierarchical_old"#lstm_l
        sup.dset.SearchModelGuid = '30A'  # 搜索模型，如果为空进行搜索
        sup.dset.SearchModelMaxTry = 100  # 模型最大搜索次数
        sup.dset.SearchModelPerTry = 2  # 模型每回合测试次数
        sup.dset.SearchModelEpochs = 2000  # 搜索时的训练次数
        sup.dset.SearchModelFitSplit = 0.1  # 搜索时的训练数据与测试拆分比例
        sup.dset.loss_name = "mse"  # mse mae kl_divergence、huber
        sup.dset.metrics_name = "cai3dloss3"  # mape、cailoss、cai3dloss、cai3dloss3
        sup.dset.PrintType = "cai3d"  # 输出类型
        Days = 30  # 历史回顾天数（推荐7-15天短期模式，20-30天中期模式，50-100天长期模式）
        UseDaysTest = False  # 启用LSTM历史回顾天数测试，自动寻找最优天数
        DayTestMax = 60  # LSTM历史回顾天数测试上限（建议不超过100天）
        GuCount = 1  # 有多组数据
        Testqi = 14   # 测试数据量（推荐：30-50期，约占总数据量0.1%-0.2%）
        GuYear = 17000  # 数据开始时间
        sup.dset.LeftNum = 2  # 前置列数
        sup.dset.OutNum = 3  # 输出列数
        sup.dset.Out_QI = 1  # 输出期数
        sup.dset.wishFitValue = 3  # 希望获取到最大结果
        sup.dset.SearchModeDo = True
        sup.dset.UseSC = False
        sup.dset.IsUpset = True  # 是否打乱
        sup.dset.UseOneHot = False
        sup.dset.OneHotCount = 3
        sup.dset.fixCallBackValue = 0.4  # 训练保存点
        sup.dset.textCallBackValue = 0.15  # 验证保存存点
        fixcount = 5000
        sup.dset.fitcount = fixcount
        strTable = 'Cai3p_LSTM_TF'
        # 使用动态SQL关键字{max_qi}替代硬编码值，在WLAi02.py的ProcessDynamicSql方法中会自动替换为实际最大值
        sup.dset.fixsql = f"select * from {strTable} where qi>{GuYear} and qi<{{max_qi}}  and a1 is not null   order by qi"
        sup.dset.testsql = f"select @top * from {strTable} where qi>{GuYear} and qi<{{max_qi}} and  a1 is not null  order by qi desc"  # top 在加载数据时处理
        sup.dset.lucksql = f"select @top * from {strTable} where qi>{GuYear} order by qi desc"
        sup.dset.Days = Days  # LSTM时间序列数
        sup.dset.GuCount = GuCount  # 每期量
        sup.dset.TestCount = Testqi  # 搞组做测试

        savepath = f'{runPath}/{saveDir}/'
        sup.dset.scfile = f'{runPath}/{saveDir}/StandardFile/save_sc.bin'
        sup.dset.ModelFile = f'{runPath}/{saveDir}/FixModels/'
        sup.dset.guidfile = f'{runPath}/{saveDir}/mainlist/'
        sup.dset.SearchModelfile = f'{runPath}/{saveDir}/SearchModels/'
        sup.dset.BestDayFile = f'{runPath}/{saveDir}/BestDays/'
        isfirst = True
        bestDay = cf()  # 用于保存最好的天数
        isfix = sup.dset.IsFix
        supcopy = copy.deepcopy(sup)  # 保留初始入参
        for i in range(fixfor):
            sup.Start(isfirst)
            if UseDaysTest and sup.dset.IsFix:  # 测试LSTM回顾天数最好的天数
                bestDay.SaveBestDay(sup.dset)
                sup.dset.Days += 5
                if sup.dset.Days > DayTestMax:
                    break
            # else:
            #     isfirst = False
            if not isfix:
                break
            if sup.dset.maxFitValue >= sup.dset.wishFitValue and not UseDaysTest:
                # sup = copy.deepcopy(supcopy)  # 初始化入参
                break
            # del sup
            gc.collect()
        if True and isfix:
            os.system('rundll32 powrprof.dll,SetSuspendState')
        input('任意键退出！')
    except Exception as err:
        print(err)
        pass
    pass


if __name__ == "__main__":
    worktype = "3D"
    if worktype == "3D":
        caiscq()
    ...
