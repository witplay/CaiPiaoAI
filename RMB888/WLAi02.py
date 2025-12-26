import matplotlib.pyplot as plt  # 引用画图
import numpy as np
import tensorflow as tf  # 引用人工智能

import ComClass.ComFun as comf
import ComClass.ComPrint as comp

np.set_printoptions(suppress=True)
from ComClass.Comm_Sql import GetSqlData as mssql
import ComClass.BaseClass as comb
from sklearn.preprocessing import StandardScaler as ss
import joblib as job
import os
import uuid
from RMB888.AiModels import TFModels
from collections import deque as deq


class WLAi(object):
    def __init__(self):
        self.isprint = True
        print('人工智能加载成功！tensorflow:' + tf.__version__)
        print('--------------------------------------------------------------')
        print('GPU是否可用：{}'.format(tf.test.is_gpu_available()))
        print('--------------------------------------------------------------')
        self.dset = comb._Dset()
        self.DDD = comb._Data()
        self.SC = ss()
        self.encoder = None
        self.scHave = False
        self.tfModel = []
        # 初始化最大值缓存字典，格式：{(table, field): max_value}
        self.max_value_cache = {}

    def Print(self, info):
        if self.isprint:
            print(info)
        ...

    def ProcessDynamicSql(self, sql):
        """
        处理SQL中的动态关键字，如{max_qi}
        @param sql: 原始SQL语句
        @return: 替换后的SQL语句
        """
        import re
        # 查找所有{max_字段名}格式的关键字
        pattern = r'\{max_([a-zA-Z0-9_]+)\}'
        matches = re.finditer(pattern, sql)
        
        for match in matches:
            keyword = match.group(0)
            field = match.group(1)
            
            # 提取表名
            table_pattern = r'from\s+([a-zA-Z0-9_]+)'
            table_match = re.search(table_pattern, sql, re.IGNORECASE)
            if not table_match:
                continue
            
            table = table_match.group(1)
            cache_key = (table, field)
            
            # 检查缓存中是否已有该值
            if cache_key in self.max_value_cache:
                max_value = self.max_value_cache[cache_key]
                self.Print(f"动态SQL替换（从缓存）：{keyword} -> {max_value} (来自表 {table} 的字段 {field})")
            else:
                # 构建查询最大值的SQL
                max_sql = f"select max({field}) from {table}"
                max_result = mssql.getAllData(max_sql, '', '')
                
                # 获取最大值
                max_value = max_result[0][0] if max_result and max_result[0][0] is not None else 0
                self.Print(f"动态SQL替换（新查询）：{keyword} -> {max_value} (来自表 {table} 的字段 {field})")
                
                # 将结果存入缓存
                self.max_value_cache[cache_key] = max_value
            
            # 替换SQL中的关键字
            sql = sql.replace(keyword, str(max_value))
        
        return sql
    
    def DataLoad(self, isUseWitplay=False):
        if os.path.exists(self.dset.scfile):
            self.SC = job.load(self.dset.scfile)
            self.scHave = True
            self.Print("找到归一化文件：" + self.dset.scfile)
            pass
        else:
            self.Print("未找到归一化文件：" + self.dset.scfile)
        pass
        c_Out_c = self.dset.OutNum * self.dset.Out_QI  # 输出列数
        self.DDD.agr_OutLength = c_Out_c
        c_R_s = self.dset.LeftNum  # 真实值开始位置
        c_R_e = self.dset.LeftNum + c_Out_c  # 真实值结束位置
        c_In_s = c_R_e  # 训练数值开始位置
        c_test_count = ((self.dset.TestCount - 1) * self.dset.GuCount) + (self.dset.GuCount * self.dset.Days)
        
        # 处理动态SQL
        self.dset.fixsql = self.ProcessDynamicSql(self.dset.fixsql)
        self.dset.testsql = self.ProcessDynamicSql(self.dset.testsql)
        self.dset.lucksql = self.ProcessDynamicSql(self.dset.lucksql)
        
        self.dset.sqltitle = mssql.getSqlTitle(self.dset.fixsql)  # 获取sql标题
        if isUseWitplay:  # 连接训练设置
            self.dset.TestCount = self.dset.WitTestCount
        if self.dset.IsFix or isUseWitplay:
            DsFix = np.array(mssql.getAllData(self.dset.fixsql, '', ''))
            DsFix = self.ShapeCat(DsFix)
            self.DDD.DsFix = DsFix
            self.DDD.agr_FixShape = self.DDD.DsFix.shape
            if self.dset.data_outtype == "":
                self.dset.data_outtype = np.float32  # 修改为float32以兼容TensorFlow
            if self.dset.data_intype == "":
                self.dset.data_intype = np.float32  # 修改为float32以兼容TensorFlow
            if self.dset.IsFixAllData:
                c_fix_count = DsFix.shape[0]
            else:
                c_fix_count = DsFix.shape[0] - (self.dset.TestCount * self.dset.GuCount)

            self.DDD.agr_FixCount = c_fix_count
            self.DDD.DsFix_Left = DsFix[0:c_fix_count, 0:self.dset.LeftNum]  # 前置数据集
            self.DDD.DsFix_R = DsFix[0:c_fix_count, c_R_s:c_R_e].astype(self.dset.data_outtype)
            self.DDD.DsFix_In = DsFix[0:c_fix_count, c_In_s:].astype(self.dset.data_intype)
            # 验证参数赋值
            self.DDD.agr_TestCount = c_test_count
            self.DDD.DsTest_Left = DsFix[-c_test_count:, 0:self.dset.LeftNum]  # 前置数据集
            self.DDD.DsTest_R = DsFix[-c_test_count:, c_R_s:c_R_e].astype(self.dset.data_outtype)
            self.DDD.DsTest_In = DsFix[-c_test_count:, c_In_s:].astype(self.dset.data_intype)
            pass
        else:
            # 验证参数赋值
            DsTest = np.array(
                mssql.getAllData(self.dset.testsql.replace("@top", "top {}".format(c_test_count)), '', ''))[
                     ::-1]  # 反转
            DsTest = self.ShapeCat(DsTest)
            self.DDD.DsTest_Left = DsTest[0:, 0:self.dset.LeftNum]  # 前置数据集
            self.DDD.DsTest_R = DsTest[0:, c_R_s:c_R_e].astype(self.dset.data_outtype)
            self.DDD.DsTest_In = DsTest[0:, c_In_s:].astype(self.dset.data_intype)
            pass

        # 预测未来参数
        LuckCount = self.dset.GuCount * self.dset.Days
        DsLuck = np.array(mssql.getAllData(self.dset.lucksql.replace("@top", "top {}".format(LuckCount)), '', ''))[::-1]
        DsLuck = self.ShapeCat(DsLuck)
        self.DDD.DsLuck = DsLuck
        self.DDD.agr_LuckShape = self.DDD.DsLuck.shape
        self.DDD.DsLuck_Left = DsLuck[0:, 0:self.dset.LeftNum]
        self.DDD.DsLuck_R = DsLuck[0:, c_R_s:c_R_e].astype(self.dset.data_outtype)
        self.DDD.DsLuck_In = DsLuck[0:, c_In_s:].astype(self.dset.data_intype)
        self.DDD.agr_InLength = DsLuck.shape[1] - c_In_s
        self.DDD.agr_LuckCount = DsLuck.shape[0]
        # del DsFix
        # del DsLuck
        # gc.collect()
        self.DataLoadPrint()
        self.FromDate()  # 归一化数据
        if self.dset.GuCount == 1:  # 转换为LSTM,多分类情况下需要进入另一种算法
            self.DoShape(isUseWitplay)
        else:
            self.DoShape_Long(isUseWitplay)
        self.DoHot()
        self.DoUpset_NP()  # 打乱数据
        if '_auto' in self.dset.tfModelType:
            if 'tensorflow' in str(type(self.DDD.DsFix_In)):
                self.DDD.DsFix_In = self.DDD.DsFix_In.numpy()
            if 'tensorflow' in str(type(self.DDD.DsFix_R)):
                self.DDD.DsFix_R = self.DDD.DsFix_R.numpy()
            if 'tensorflow' in str(type(self.DDD.DsTest_In)):
                self.DDD.DsTest_In = self.DDD.DsTest_In.numpy()
            if 'tensorflow' in str(type(self.DDD.DsTest_R)):
                self.DDD.DsTest_R = self.DDD.DsTest_R.numpy()

        # if not isUseWitplay:
        #     self.DoUpset()  # 打乱数据
        self.DataLoadPrint()
        self.PrintShape()

    # 数据加载显示
    def DataLoadPrint(self):
        if not self.isprint:
            return
        # ii=0
        # for x in self.DDD.DsFix_Left:
        #    print("{}:{}->{}".format(x,self.DDD.DsFix_R[ii],self.DDD.DsFix_In[ii]))
        #    ii+=1
        #    pass
        # return
        print('训练前置:')
        # print(self.DDD.DsFix_Left)
        print('训练真实:')
        print(self.DDD.DsFix_R)
        print('训练输入:')
        print(self.DDD.DsFix_In)
        print('--------------------------------------------------------------')
        print('验证前置:')
        print(self.DDD.DsTest_Left)
        print('验证真实:')
        print(self.DDD.DsTest_R)
        print('验证输入:')
        print(self.DDD.DsTest_In)
        print('--------------------------------------------------------------')
        print('幸运前置：')
        print(self.DDD.DsLuck_Left)
        print('幸运真实：')
        print(self.DDD.DsLuck_R)
        print('幸运输入：')
        print(self.DDD.DsLuck_In)
        print('--------------------------------------------------------------')
        print("训练空间")
        print(f'输入：{self.DDD.DsFix_In.shape},真实：{self.DDD.DsFix_R.shape},信息：{self.DDD.DsFix_Left.shape}')
        print('验证空间')
        print(f'输入：{self.DDD.DsTest_In.shape},真实：{self.DDD.DsTest_R.shape},信息：{self.DDD.DsTest_Left.shape}')
        print('幸运空间')
        print(f'输入：{self.DDD.DsLuck_In.shape},真实：{self.DDD.DsLuck_R.shape},信息：{self.DDD.DsLuck_Left.shape}')
        print('输入列：')
        print(self.DDD.agr_InLength)
        print('输出列：')
        print(self.DDD.agr_OutLength)
        print('训练数据量：')
        print('{}-->{}'.format(self.DDD.agr_FixCount, self.DDD.DsFix_Left.shape[0]))
        print('测试数据量：')
        print('{}-->{}'.format(self.DDD.agr_TestCount, self.DDD.DsTest_Left.shape[0]))
        print('验证数据量：')
        print('{}-->{}'.format(self.DDD.agr_LuckCount, self.DDD.DsLuck_Left.shape[0]))
        print('--------------------------------------------------------------')
        pass

    def PrintShape(self):
        if not self.isprint:
            return
        print('--------------------------------------------------------------')
        print("训练数据集输入空间：{}".format(self.DDD.DsFix_In.shape))
        print("训练数据集真值空间：{}".format(self.DDD.DsFix_R.shape))
        print("训练数据集输出空间：{}".format(self.DDD.DsFix_Out.shape))

        print("验证数据集输入空间：{}".format(self.DDD.DsTest_In.shape))
        print("验证数据集真值空间：{}".format(self.DDD.DsTest_R.shape))
        print("验证数据集输出空间：{}".format(self.DDD.DsTest_Out.shape))

        print("预测数据集输入空间：{}".format(self.DDD.DsLuck_In.shape))
        print("预测数据集输入空间：{}".format(self.DDD.DsLuck_R.shape))
        print("预测数据集输出空间：{}".format(self.DDD.DsLuck_Out.shape))
        print('--------------------------------------------------------------')
        pass

    # 格式化数据
    def FromDate(self):
        """
        格式化数据
        @return:
        """
        if self.dset.UseSC:
            if self.dset.IsFix:
                scini = True
                if self.scHave:
                    scini = False
                self.DDD.DsFix_In = self.ReshapeFun(self.DDD.DsFix_In, False, scini)
            self.DDD.DsTest_In = self.ReshapeFun(self.DDD.DsTest_In)
            self.DDD.DsLuck_In = self.ReshapeFun(self.DDD.DsLuck_In)
            sctest = self.ReshapeFun(self.DDD.DsLuck_In, True)
            self.Print('归一化验证幸运输入：')
            self.Print(sctest)
            self.Print('↓↓↓↓')
            self.Print('归一化幸运输入：')
            self.Print(self.DDD.DsLuck_In)
        pass

    def ReshapeFun(self, indata, isback=False, isini=False):
        """
        归一化处理类
        @param indata:
        @param isback: 否反序列
        @param isini: 是否初始并保存
        @return:
        """
        sp = indata.shape
        indata = np.reshape(indata, (-1, 1))
        if isini:
            indata = self.SC.fit_transform(indata, )  # 拟合训练
            job.dump(self.SC, self.dset.scfile)
        else:
            if not isback:
                indata = self.SC.transform(indata)
                pass
            else:
                indata = self.SC.inverse_transform(indata)
                pass
        indata = np.reshape(indata, sp)
        return indata

    # 截断数据用于整齐
    def ShapeCat(self, indate):
        return indate  # 新的LSTM输入不需要截
        GuCount = self.dset.GuCount * self.dset.Days
        shape_count = int(indate.shape[0] / GuCount)
        shape_count = shape_count * GuCount
        shape_count = indate.shape[0] - shape_count
        indate = indate[shape_count:]
        self.Print("数据对齐，被截断量：{0}".format(shape_count))
        return indate

    def DoHot(self):
        if not self.dset.UseOneHot:
            return
            # 创建OneHotEncoder对象
        # 对颜色变量进行独热编码
        num_classes = self.dset.OneHotCount  # 类别数量
        # one_hot_matrix = np.eye(num_classes)  # 生成独热编码矩阵
        if len(self.DDD.DsFix_R) > 0:
            self.DDD.DsFix_R = tf.keras.utils.to_categorical(self.DDD.DsFix_R, num_classes=num_classes)  # 获取对应的独热编码矩阵
            # self.DDD.DsFix_R = self.encoder.fit_transform(self.DDD.DsFix_R).toarray()
        print('-' * 30 + '独热验证' + '-' * 30)
        print(self.DDD.DsTest_R)
        print('-' * 30 + '独热后' + '-' * 30)
        self.DDD.DsTest_R = tf.keras.utils.to_categorical(self.DDD.DsTest_R, num_classes=num_classes)  # 获取对应的独热编码矩阵
        # self.DDD.DsTest_R = self.encoder.fit_transform(self.DDD.DsTest_R).toarray()
        print(self.DDD.DsTest_R)
        pass

    def DoUpset_NP(self):
        """
        是否打乱数据
        @return:
        """
        if not self.dset.IsUpset:
            return
        # 使用NumPy的shuffle方法打乱NumPy数组
        np.random.seed(6)
        np.random.shuffle(self.DDD.DsFix_In)
        np.random.seed(6)
        np.random.shuffle(self.DDD.DsFix_R)
        np.random.seed(6)
        np.random.shuffle(self.DDD.DsFix_Left)
        pass

    # 打乱
    def DoUpset(self):
        """
        是否打乱数据
        @return:
        """
        if not self.dset.IsUpset:
            return
        tf.random.set_seed(7)
        self.DDD.DsFix_In = tf.random.shuffle(self.DDD.DsFix_In)
        tf.random.set_seed(7)
        self.DDD.DsFix_R = tf.random.shuffle(self.DDD.DsFix_R)
        tf.random.set_seed(7)
        self.DDD.DsFix_Left = tf.random.shuffle(self.DDD.DsFix_Left)
        pass

    # 转变空间
    def DoShape(self, isUseWitplay=False):
        """
        转变空间
        @return:
        """
        InLength = self.DDD.agr_InLength
        OutLength = self.DDD.agr_OutLength
        Gcount = self.dset.GuCount
        GuDayCount = self.dset.GuCount * self.dset.Days
        Rcount = int(self.dset.GuCount)
        if self.dset.IsFix or isUseWitplay:
            self.DDD.DsFix_In = self.CreateLstmInput(self.DDD.DsFix_In, GuDayCount)
            self.DDD.DsFix_R = self.DDD.DsFix_R[GuDayCount - 1:]
            self.DDD.DsFix_Left = self.DDD.DsFix_Left[GuDayCount - 1:]
            if not self.dset.isClassFix:
                self.DDD.DsFix_R = tf.reshape(self.DDD.DsFix_R,
                                              (int(self.DDD.DsFix_R.shape[0] / Rcount), Rcount, OutLength))
        self.DDD.DsTest_In = self.CreateLstmInput(self.DDD.DsTest_In, GuDayCount)
        self.DDD.DsTest_R = self.DDD.DsTest_R[GuDayCount - 1:]
        self.DDD.DsTest_Left = self.DDD.DsTest_Left[GuDayCount - 1:]
        if not self.dset.isClassFix:
            self.DDD.DsTest_R = tf.reshape(self.DDD.DsTest_R,
                                           (int(self.DDD.DsTest_R.shape[0] / Rcount), Rcount, OutLength))

        self.DDD.DsLuck_In = self.CreateLstmInput(self.DDD.DsLuck_In, GuDayCount)
        self.DDD.DsLuck_R = self.DDD.DsLuck_R[GuDayCount - 1:]
        self.DDD.DsLuck_Left = self.DDD.DsLuck_Left[GuDayCount - 1:]
        if not self.dset.isClassFix:
            self.DDD.DsLuck_R = tf.reshape(self.DDD.DsLuck_R,
                                           (int(self.DDD.DsLuck_R.shape[0] / Rcount), Rcount, OutLength))
        pass

    def DoShape_Long(self, isUseWitplay=False):
        """
        多分类数据转换LSTM
        @return:
        """
        if self.dset.IsFix or isUseWitplay:
            self.DDD.DsFix_In, self.DDD.DsFix_R, self.DDD.DsFix_Left = \
                self.CreateLstmInput_Long(self.DDD.DsFix_In, self.DDD.DsFix_R, self.DDD.DsFix_Left)
        self.DDD.DsTest_In, self.DDD.DsTest_R, self.DDD.DsTest_Left = \
            self.CreateLstmInput_Long(self.DDD.DsTest_In, self.DDD.DsTest_R, self.DDD.DsTest_Left)
        self.DDD.DsLuck_In, self.DDD.DsLuck_R, self.DDD.DsLuck_Left = \
            self.CreateLstmInput_Long(self.DDD.DsLuck_In, self.DDD.DsLuck_R, self.DDD.DsLuck_Left)
        pass

    def CreateLstmInput(self, indata, GuDayCount):
        """
        转换为LSTM数据结构
        @param indata:
        @param GuDayCount:
        @return:
        """

        x_deq = deq(maxlen=GuDayCount)
        x_out = []
        for ix in indata:
            x_deq.append(ix)
            if len(x_deq) == GuDayCount:
                x_out.append(np.array(x_deq))
        return np.array(x_out)  # 返回NumPy数组而不是Tensor，以便后续打乱操作
        pass

    def CreateLstmInput_Long(self, X, Y, L):
        """
        转换为LSTM数据结构，多样本数据time_steps
        @param X: 输入
        @param Y: 输出
        @param L: 左信息
        @return:
        """
        # InLength = self.DDD.agr_InLength
        # OutLength = self.DDD.agr_OutLength
        # Gcount = self.dset.GuCount
        if isinstance(X, tf.Tensor) and tf.executing_eagerly():
            X = X.numpy()
        if isinstance(Y, tf.Tensor) and tf.executing_eagerly():
            Y = Y.numpy()
        time_steps = self.dset.Days
        X_grouped = {}
        Y_grouped = {}
        L_grouped = {}
        for i in range(X.shape[0]):
            stock_code = X[i][0]
            if stock_code not in X_grouped:
                X_grouped[stock_code] = []
                Y_grouped[stock_code] = []
                L_grouped[stock_code] = []
            X_grouped[stock_code].append(X[i])
            Y_grouped[stock_code].append(Y[i])
            L_grouped[stock_code].append(L[i])
        X_new = []
        Y_new = []
        L_new = []
        for stock_code in X_grouped:
            X_stock = np.array(X_grouped[stock_code])
            Y_stock = np.array(Y_grouped[stock_code])
            L_stock = np.array(L_grouped[stock_code])
            for i in range(len(X_stock) - time_steps + 1):
                X_new.append(X_stock[i:i + time_steps])
                Y_new.append(Y_stock[i + time_steps - 1])
                L_new.append(L_stock[i + time_steps - 1])
        if self.dset.isClassFix:
            Y_new = np.array(Y_new).astype(np.int_)
        else:
            Y_new = np.array(Y_new)
        return np.array(X_new), Y_new, np.array(L_new)
        pass

    def FixData(self):
        """
        训练数据
        @return:
        """
        mycallback = comf.TfComFun.FitCallBack(self.dset)
        if self.dset.bestModelFile == "":
            # 保存tfm对象引用，以便后续使用其回调函数
            self.tfm = tfm = TFModels(self.dset, self.DDD)
            tfm.inShape = tuple(int(x) for x in self.DDD.DsFix_In.shape[1:])  # 确保形状元素为整数类型
            tfm.OutShape = int(self.DDD.DsFix_R.shape[-1])  # 确保输出形状为整数类型
            # tfm.OutShape = self.DDD.agr_OutLength
            ftmodel = tfm.GetModel(self.dset.tfModelType)
            # ftmodel.summary()
        else:
            models = comf.TfComFun.GetBestModelsByGuid(self.dset)
            if models:
                ftmodel = models[0]
                if self.dset.LoadModelFiles:
                    self.dset.CurrentModelFile = self.dset.LoadModelFiles[0]
                ftmodel.summary()
            else:
                raise Exception("无法获取到模型，请检查模型文件或GUID配置")
        # 合并回调函数 - 如果模型有自带的回调函数
        callbacks = [mycallback]
        if hasattr(tfm, 'callbacks') and tfm.callbacks:
            callbacks.extend(tfm.callbacks)
            
        ftmodel.fit(self.DDD.DsFix_In, self.DDD.DsFix_R,
                    validation_data=(self.DDD.DsTest_In, self.DDD.DsTest_R),
                    epochs=self.dset.fitcount, callbacks=callbacks)
        del ftmodel
        del mycallback
        pass

    def WitFixData(self, sAveModel=False):
        """
        连接训练数据
        @return:
        """
        witfitcallback, wittestcallback = comf.TfComFun.WitFitClaaBack(self.dset)
        for witmodel in self.tfModel:
            if sAveModel:
                self.dset.guid = str(uuid.uuid1()).replace('-', '')
                mycallback = comf.TfComFun.FitCallBack(self.dset)
                witmodel.fit(self.DDD.DsFix_In, self.DDD.DsFix_R,
                             validation_data=(self.DDD.DsTest_In, self.DDD.DsTest_R),
                             epochs=self.dset.fitcount, callbacks=mycallback)
                self.tfModel = comf.TfComFun.GetBestModelsByGuid(self.dset)
            else:
                if self.dset.valLossSaveMode == "max":
                    train_acc = 0
                    train_val = 0
                else:
                    train_acc = 9999
                    train_val = 9999
                while (
                        (
                                train_acc < self.dset.WitFixCallValue or train_val < self.dset.WitTestCallValue) and self.dset.valLossSaveMode == "max") \
                        or (
                        (
                                train_acc > self.dset.WitFixCallValue or train_val > self.dset.WitTestCallValue) and self.dset.valLossSaveMode == "min"):
                    # 合并回调函数 - 使用验证回调和模型自带的回调
                    callbacks = [wittestcallback]
                    if hasattr(self, 'tfm') and hasattr(self.tfm, 'callbacks') and self.tfm.callbacks:
                        callbacks.extend(self.tfm.callbacks)
                    
                    history = witmodel.fit(self.DDD.DsFix_In, self.DDD.DsFix_R,
                                           validation_data=(self.DDD.DsTest_In, self.DDD.DsTest_R),
                                           epochs=self.dset.fitcount, callbacks=callbacks)
                    train_acc = history.history[self.dset.metrics_name][-1]
                    train_val = history.history['val_' + self.dset.metrics_name][-1]
        return True
        pass

    def PredictData(self):
        """
        预测数据
        @return:
        """
        global pout
        self.tfModel = comf.TfComFun.GetBestModelsByGuid(self.dset)
        ai666 = 0
        luck = []
        test = []
        self.dset.maxFitValue = 0
        model_outs = []
        model_ins = []
        for tfModel in self.tfModel:
            ai666 += 1
            # if self.dset.IsFix:
            #     self.DDD.DsFix_Out = tfModel.predict(self.DDD.DsFix_In)
            #     print("------------------------模板{}测试-------------------------".format(ai666))
            #     self.PrintValue(self.DDD.DsFix_Out, self.DDD.DsFix_R, self.DDD.DsFix_Left)
            #     self.PlotLine(self.DDD.DsFix_Out,self.DDD.DsFix_R)
            # tfModel.summary()
            model_outs.append(tfModel.output)
            model_ins.append(tfModel.input)
            self.DDD.DsTest_Out = tfModel.predict(self.DDD.DsTest_In)
            test.append([self.DDD.DsTest_Out, self.DDD.DsTest_R, self.DDD.DsTest_Left])
            self.DDD.DsLuck_Out = tfModel.predict(self.DDD.DsLuck_In)
            luck.append([self.DDD.DsLuck_Out, self.DDD.DsLuck_R, self.DDD.DsLuck_Left])
        # 将模型的输出结合起来
        combined = model_outs
        # 定义模型集成
        # bestmodel = tf.keras.Model(inputs=model_ins, outputs=combined)
        # # bestmodel = VotingClassifier(model_outs, voting='soft')
        # # bestmodel.fit(self.DDD.DsFix_In, self.DDD.DsFix_R)
        # self.DDD.DsTest_Out = bestmodel.predict(self.DDD.DsTest_In)
        # test.append([self.DDD.DsTest_Out, self.DDD.DsTest_R, self.DDD.DsTest_Left])
        # self.DDD.DsLuck_Out = bestmodel.predict(self.DDD.DsLuck_In)
        # luck.append([self.DDD.DsLuck_Out, self.DDD.DsLuck_R, self.DDD.DsLuck_Left])
        print("---------------------------测试6666----------------------------")
        ai666 = 0
        maxlist = []
        maxi = 0
        isint = True
        for tk in test:
            if ai666 < len(self.dset.LoadModelFiles):
                self.dset.CurrentModelFile = self.dset.LoadModelFiles[ai666]
            ai666 += 1
            print(self.dset.CurrentModelFile)
            print("------------------------模板{}测试-------------------------".format(ai666))
            outPrintValue = comp.PrintTestFun(self.dset, tk[0], tk[1], tk[2])

            if type(outPrintValue) == int or type(outPrintValue) == float:  # 加入其他返回值数组，第一个是最大值，第二个是其他说明，第三个其他值集合
                isint = True
                maxval = outPrintValue
                maxlist.append([ai666, maxval])
            else:
                isint = False
                maxval = outPrintValue[0]
                otherVal = f'{outPrintValue[1]}:{outPrintValue[2]}'
                maxlist.append([ai666, maxval, otherVal])
            if self.dset.maxFitValue < maxval:
                self.dset.maxFitValue = maxval
                maxi = ai666
        # 安全获取汇总信息
        summary_info = "未知"
        if self.DDD.DsLuck_Left.shape[0] > 0 and self.DDD.DsLuck_Left[0].shape[0] > 0:
            summary_info = self.DDD.DsLuck_Left[0][0]
        print("-------------------幸运888汇总:{}-----------------------".format(summary_info))
        ai666 = 0
        printsum = []
        outList = []
        for lk in luck:
            if ai666 < len(self.dset.LoadModelFiles):
                self.dset.CurrentModelFile = self.dset.LoadModelFiles[ai666]
            ai666 += 1
            if self.dset.PrintType in ("printgu", "cup", "cup_z", "printguclass"):
                print("------------------------模板{}测试-------------------------".format(ai666))
            isbest = ''
            if ai666 == maxi:
                isbest = '★最佳模板预测'
                print(isbest)
            # if ai666 == len(luck):
            #     print('◆综合实力预测：')
            pout = comp.PrintLuckFun(self.dset, lk[0], lk[1], lk[2])
            outList.append(pout)
            if pout is not None:
                printsum.append([pout, isbest])
        if len(printsum) > 0 and 'gu' in self.dset.PrintType:
            print('-----------------------------------AI推介-----------------------------------')
            for ip, ps in enumerate(printsum):
                print(f"------------------------模板{ip + 1}TOP10-----------------{ps[1]}")
                for pone in ps[0][:10]:
                    print(pone[0])
                    print(pone[1:])
        print("----------------模板评估MAX汇总----------------")
        for index,m in enumerate(maxlist):
            if index < len(self.dset.LoadModelFiles):
                strFileName = self.dset.LoadModelFiles[index]
                strFileName = strFileName.replace(self.dset.ModelFile, '')
            else:
                strFileName = f"模板{m[0]}"
            if isint:
                print(f'模板{m[0]}:最大值{m[1]:.2f}->{strFileName}')
            else:
                print(f'模板{m[0]}:最大值{m[1]:.2f} {m[2]}->{strFileName}')
        del self.tfModel
        del luck
        del test
        return outList
        pass

    def WitPredictData(self):
        """
        预测数据
        @return:
        """
        self.tfModel = comf.TfComFun.GetBestModelsByGuid(self.dset)
        lucktestcount = 1
        while True:
            selectvalue = mssql.getAllData(self.dset.kuaiSelectSql, '', '')
            print(f'历史：{np.array(selectvalue)[0]}')
            maxid = selectvalue[0][0]
            maxid = int(maxid) + 1
            print(f"第{lucktestcount}次幸运数")
            lucktestcount += 1
            invalue = input(f'qfgs输入{maxid}:')
            savemodel = False
            iswitfix = False
            if invalue == "q":
                break
            if invalue == "f":
                invalue = input(f'qfgs输入{maxid}:')
                iswitfix = True
            if invalue == "g":
                maxid = input(f'输入maxid:')
                invalue = input(f'qfgs输入{maxid}:')
            if invalue == "s":
                print("保存模板中")
                invalue = input(f'qfgs输入{maxid}:')
                savemodel = True
            invalue = ','.join(str(invalue))
            insertsql = self.dset.kuaiInsertSql.format(maxid, invalue)
            mssql.getAllData(insertsql, '', '')
            self.DataLoad(True)
            if iswitfix or savemodel:
                lucktestcount = 1
                self.WitFixData(savemodel)  # 智慧训练
            ai666 = 0
            luck = []
            test = []
            self.dset.maxFitValue = 0
            model_outs = []
            model_ins = []
            for tfModel in self.tfModel:
                ai666 += 1
                model_outs.append(tfModel.output)
                model_ins.append(tfModel.input)
                self.DDD.DsTest_Out = tfModel.predict(self.DDD.DsTest_In)
                self.DDD.DsLuck_Out = tfModel.predict(self.DDD.DsLuck_In)
                test.append([self.DDD.DsTest_Out, self.DDD.DsTest_R, self.DDD.DsTest_Left])
                luck.append([self.DDD.DsLuck_Out, self.DDD.DsLuck_R, self.DDD.DsLuck_Left])
            ai666 = 0
            maxi = 0
            maxFitValue = 0
            maxlist = []
            for tk in test:
                ai666 += 1
                print("------------------------模板{}测试-------------------------".format(ai666))
                maxval, lastprint = comp.PrintTestFun(self.dset, tk[0], tk[1], tk[2], True)
                maxlist.append([ai666, maxval, lastprint])
                if maxFitValue < maxval:
                    maxFitValue = maxval
                    maxi = ai666
            for m in maxlist:
                print(f'模板{m[0]}:最大值{m[1]:.2f},{m[-1]}')
            print("-------------------幸运888汇总:{}-----------------------".format(self.DDD.DsLuck_Left[0][0]))
            ai666 = 0
            printsum = []
            for lk in luck:
                ai666 += 1
                isbest = ''
                if ai666 == maxi:
                    isbest = '★最佳模板预测'
                    print('★最佳模板预测：')
                # if ai666 == len(luck):
                #     print('◆综合实力预测：')
                pout = comp.PrintLuckFun(self.dset, lk[0], lk[1], lk[2])
                if pout is not None:
                    printsum.append([pout, isbest])
            if len(printsum) > 0:
                print('-----------------------------------AI推介-----------------------------------')
                for ip, ps in enumerate(printsum):
                    print(f"------------------------模板{ip + 1}TOP10-----------------{ps[1]}")
                    for pone in ps[0][:10]:
                        print(pone[0])
                        print(pone[1:])

        del self.tfModel
        del test
        del luck
        pass

    def PlotLine(self, TestValue, TrueValue):
        c_Out_c = self.dset.OutNum * self.dset.Out_QI  # 输出列数
        GuCount = self.dset.GuCount * self.dset.Days
        TestValue = tf.reshape(TestValue, (-1, c_Out_c))
        TrueValue = tf.reshape(TrueValue, (-1, c_Out_c))
        TestValue = np.array(TestValue)
        tv = []
        slen = TestValue.shape[0]
        for sx in range(slen):
            gcs = int(sx / GuCount)
            tv.append(TrueValue[gcs])
        TrueValue = np.array(tv)
        plt.plot(TestValue, color='red', label='TestValue')
        plt.plot(TrueValue, color='blue', label='TrueValue')
        plt.legend()
        plt.show()
        pass

    def Start(self, isfirst=True):
        outdata = []
        if isfirst and (self.dset.guid != "all" or self.dset.IsFix):
            self.dset = comb.Fun.LoadGuidFile(self.dset)
            self.DataLoad()
        if self.dset.IsFix:  # 需要训练
            self.dset.guid = str(uuid.uuid1()).replace('-', '')
            self.FixData()
            self.dset.bestModelFile = ''

        if self.dset.guid == "all" and not self.dset.IsFix:
            self.isprint = False
            dss = comb.Fun.LoadGuidFileAll(self.dset)
            maxv = 0
            maxguid = ''
            for dset in dss:
                print("-------------GUID:{}-------------".format(dset.guid))
                self.dset = dset
                self.DataLoad()
                self.PredictData()
                if maxv < self.dset.maxFitValue:
                    maxv = self.dset.maxFitValue
                    maxguid = dset.guid
            print(f'最大值GUID：{maxguid}★{maxv:.2f}')
        else:
            outdata = self.PredictData()
            #comb.Fun.SaveGuidFile(self.dset)
            if (self.dset.maxFitValue >= self.dset.wishFitValue or not self.dset.IsFix) and self.dset.NotInKuiTest:
                kuai888 = input('按K进入连续验证通道，按其他保存结果！')
                if kuai888 == 'k':
                    self.WitPredictData()
        if self.dset.IsFix:
            print(f'期望值：{self.dset.wishFitValue}，评估结果：{self.dset.maxFitValue}')
            if not self.dset.maxFitValue >= self.dset.wishFitValue:  # 没获取到最大值 删除模型
                comb.Fun.DelFileUseGuid(self.dset.guid, self.dset.ModelFile)
                print("未获取到最大值删除模型")
                if self.dset.fitcount < 30000:
                    self.dset.fitcount = int(self.dset.fitcount * 1.3)  # 达不到时增加训练次数
                    print(f"增加训练量到{self.dset.fitcount}")
            else:
                comb.Fun.SaveGuidFile(self.dset)
                print("保存模型配置文件成功")
        return outdata
        pass
