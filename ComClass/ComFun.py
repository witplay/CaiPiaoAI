import os
import shutil
import tensorflow as tf
import time

from keras.callbacks import EarlyStopping

import ComClass.BaseClass as bs
#
import numpy as np
from keras import backend as K
from ComClass.Comm_Sql import GetSqlData as mssql
from keras.utils import plot_model

from keras.callbacks import Callback




class TfComFun(object):
    """
    人工智能公共方法类
    """

    def __int__(self):
        ...

    """
    彩票测试结果保存
    leftValue：彩票基数，xValue预测结果，trueValue真实结果
    """

    def SaveCaiTest(dset: bs._Dset, leftValue, xValue, trueValue):
        if not dset.SaveTest:
            return
        if dset.CurrentModelFile == '':
            return
        if dset.PrintType == "cai":
            saveTable = "cai88_x"
        elif dset.PrintType == "cai3d":
            saveTable = "cai3d_x"
        else:
            print(f"未配置{dset.PrintType}的预测保存表")
            return
        strFileKey = dset.CurrentModelFile.replace(dset.ModelFile, '')
        strSelect = f"select count(1) c from {saveTable} where FileKey='{strFileKey}' and ri='{leftValue[0]}' and qi='{leftValue[1]}' and t1 is not null " \
                    f"union all " \
                    f"select count(1) c from {saveTable} where FileKey='{strFileKey}' and ri='{leftValue[0]}' and qi='{leftValue[1]}' and t1 is null "
        iCount = mssql.getAllData(strSelect, '', '')
        if iCount[0][0] > 0:
            print("已存在预测值！")
            return
        if iCount[1][0] > 0:
            delsql = f"delete {saveTable} where FileKey='{strFileKey}' and ri='{leftValue[0]}' and qi='{leftValue[1]}' and t1 is null"
            mssql.getAllData(delsql, '', '')
            print("已删除预测Luck值！")
        strrowid = f"select isnull(max(rowid),0) c from {saveTable} where FileKey='{strFileKey}'"
        rowid = mssql.getAllData(strrowid, '', '')[0][0]
        rowid += 1
        strInsValue = f"'{strFileKey}',{str(leftValue.tolist())},{str(xValue.tolist())},{str(trueValue.tolist())},{rowid}" \
            .replace('[', '').replace(']', '')
        strSql = f"insert {saveTable} values ({strInsValue})"
        mssql.getAllData(strSql, '', '')
        print("预测值已保存！")
        ...

    """
    彩票预测结果保存
    leftValue：彩票基数，xValue预测结果
    """

    def SaveCaiLuck(dset: bs._Dset, leftValue, xValue):
        if not dset.SaveTest:
            return
        if dset.CurrentModelFile == '':
            return
        if dset.PrintType == "cai":
            saveTable = "cai88_x"
            saveColumn = "FileKey,ri,qi,x1,x2,x3,x4,x5,x6,x7,rowid"
        elif dset.PrintType == "cai3d":
            saveTable = "cai3d_x"
            saveColumn = "FileKey,ri,qi,x1,x2,x3,rowid"
        else:
            print(f"未配置{dset.PrintType}的预测保存表")
            return

        strFileKey = dset.CurrentModelFile.replace(dset.ModelFile, '')
        strSelect = f"select count(1) c from {saveTable} where FileKey='{strFileKey}' and ri='{leftValue[0]}' and qi='{leftValue[1]}'  and t1 is null"
        iCount = mssql.getAllData(strSelect, '', '')
        if iCount[0][0] > 0:
            delsql = f"delete {saveTable} where FileKey='{strFileKey}' and ri='{leftValue[0]}' and qi='{leftValue[1]}' and t1 is null"
            mssql.getAllData(delsql, '', '')
            # print("删除已存在Luck预测值！")
        strrowid = f"select isnull(max(rowid),0) c from {saveTable} where FileKey='{strFileKey}'"
        rowid = mssql.getAllData(strrowid, '', '')[0][0]
        rowid += 1
        strInsValue = f"'{strFileKey}',{str(leftValue.tolist())},{str(xValue.tolist())},{rowid}" \
            .replace('[', '').replace(']', '')
        strSql = f"insert {saveTable} ({saveColumn}) values ({strInsValue})"
        mssql.getAllData(strSql, '', '')
        # print("Luck预测值已保存！")
        ...

    """
    生成模型图片，目前不使用
    """

    def SaveModelImg(loadModel, fileName: str):
        return  # 暂时禁用生成模型图片功能，避免pydot和graphviz依赖错误
        # loadModel.summary()
        modelImg = f'{fileName}/model.png'
        if os.path.exists(modelImg):
            print('模型图片路径：{}'.format(modelImg))
        else:
            plot_model(loadModel, to_file=modelImg)
            print('生成模型图片路径：{}'.format(modelImg))
        pass

    def GetBestModelsByGuid(dset: bs._Dset = bs._Dset()):
        """
        通过GUID获取最好的Model，不是最好的删除
        """
        tfmodels = []
        dset.LoadModelFiles = []
        modelFiles = []
        _loss = {'cailoss': TfComFun.cailoss,
                 'cailoss2': TfComFun.cailoss2,
                 'cailoss1': TfComFun.cailoss1,
                 'myloss': TfComFun.cai3dloss, 
                 'cai3dloss': TfComFun.cai3dloss,
                 'cai3dloss2': TfComFun.cai3dloss2,
                 'cai3dloss3': TfComFun.cai3dloss3,
                'rmbloss': TfComFun.rmbloss,
                'cuploss': TfComFun.cuploss,
                'cuploss_z': TfComFun.cuploss_z,
                'myacc': TfComFun.myacc,
                'guaccuracy': TfComFun.guaccuracy,
                }
        if dset.bestModelFile != "":
            print(f'加载已有模板：\r\n{dset.bestModelFile}')
            # 局部导入，避免循环依赖
            from RMB888.AiModels import WarmupCosineSchedule
            _loss['WarmupCosineSchedule'] = WarmupCosineSchedule
            loadModel = tf.keras.models.load_model(dset.bestModelFile, custom_objects=_loss)
            TfComFun.SaveModelImg(loadModel, dset.bestModelFile)
            loadModel.summary()
            dset.LoadModelFiles.append(dset.bestModelFile)
            tfmodels.append(loadModel)
            return tfmodels
        gudi = dset.guid
        dirname = dset.ModelFile
        loss = [99999999.0] * 7
        files = [''] * 7
        delfiles = []
        for root, dirs, ifles in os.walk(dirname):
            for x in dirs:
                if x.index('_') == 0:
                    continue
                fname = str(x).split('_')
                if fname[2] != gudi:
                    continue
                delfiles.append(x)
                s = np.array(fname[1].split('@')).astype(np.float32)
                s = list(s)
                if dset.valLossSaveMode == 'max':
                    s[3] = 1 - s[3]
                    s[1] = 1 - s[1]
                s_l = s[0] + s[2]
                s_r = s[1] + s[3]
                s_s = sum(s)
                s.append(s_l)
                s.append(s_r)
                s.append(s_s)
                for i in range(7):
                    if s[i] < loss[i]:
                        loss[i] = s[i]
                        files[i] = x
                    pass
            break
        havefiles = []
        for x in files:
            if x != '':
                mfile = dirname + x
                if not (mfile in havefiles):
                    delfiles.remove(x)
                    havefiles.append(mfile)
                    print('加载模型路径：{}'.format(mfile))
                    # 局部导入，避免循环依赖
                    from RMB888.AiModels import WarmupCosineSchedule
                    _loss['WarmupCosineSchedule'] = WarmupCosineSchedule
                    dset.LoadModelFiles.append(mfile)
                    loadModel = tf.keras.models.load_model(mfile, custom_objects=_loss)
                    TfComFun.SaveModelImg(loadModel, mfile)
                    tfmodels.append(loadModel)
            pass
        for delf in delfiles:
            delf = dirname + delf
            try:
                shutil.rmtree(delf)
                # print('删除目录{0}'.format(delf))
                pass
            except:
                print('删除目录{0}失败'.format(delf))
                pass
        dset.bestFitValue = loss
        if tfmodels:
            tfmodels[-1].summary()
        return tfmodels

    # 百分比验证
    @tf.function
    def cailoss(y_true_yz, y_pred_yz):
        """
        彩票损失函数，三维输出，Min
        @param y_pred_yz:
        @return:
        """
        y_true = tf.cast(tf.round(y_true_yz), dtype=tf.int32)
        y_pred = tf.cast(tf.round(y_pred_yz), dtype=tf.int32)
        n0_0 = tf.sets.intersection(y_true[:, :, :-1], y_pred[:, -1:, :-1])
        n0_1 = tf.sets.intersection(y_true[:, :, -1], y_pred[:, -1:, -1])
        n0 = tf.size(n0_0.values) + tf.size(n0_1.values)
        allcount = tf.size(y_true_yz)
        per = tf.cast(1 - (n0 / allcount), dtype=tf.float64)
        return per

    # @tf.function
    # def cailoss2(y_true_yz, y_pred_yz):
    #     y_true = tf.cast(tf.round(y_true_yz), dtype=tf.int32)
    #     y_pred = tf.cast(tf.round(y_pred_yz), dtype=tf.int32)
    #     n0_0 = tf.sets.intersection(y_true[:,:-1], y_pred[:,:-1])
    #     n0_1 = tf.math.equal(y_true[:,-1], y_pred[:,-1])
    #     n0 = tf.size(n0_0.values) + tf.reduce_sum(tf.cast(n0_1, dtype=tf.int32))
    #     allcount = tf.size(y_true_yz)
    #     per = tf.cast((n0 / allcount), dtype=tf.float64)
    #     return per

    @tf.function
    def cailoss1(y_true_yz, y_pred_yz):
        """
        彩票损失函数，三维输出，MAX
        @param y_pred_yz:
        @return:
        """
        y_true = tf.cast(tf.round(y_true_yz), dtype=tf.int32)
        y_pred = tf.cast(tf.round(y_pred_yz), dtype=tf.int32)
        n0_0 = tf.sets.intersection(y_true[:, :, :-1], y_pred[:, -1:, :-1])
        n0_1 = tf.sets.intersection(y_true[:, :, -1], y_pred[:, -1:, -1])
        n0 = tf.size(n0_0.values) + tf.size(n0_1.values)
        allcount = tf.size(y_true_yz)
        per = tf.cast(n0 / allcount, dtype=tf.float64)
        return per

    @tf.function
    def cailoss2(y_true_yz, y_pred_yz):
        """
        双色球彩票损失函数，二维输出，MAX
        @param y_pred_yz:
        @return:
        """
        y_true = tf.cast(y_true_yz, dtype=tf.int32)
        y_pred = tf.cast(tf.round(y_pred_yz), dtype=tf.int32)

        # 计算相同值的数量
        intersection_count = tf.size(tf.sets.intersection(y_true[:, :-1], y_pred[:, :-1]).values)
        label_equal_count = tf.reduce_sum(tf.cast(tf.math.equal(y_true[:, -1], y_pred[:, -1]), dtype=tf.int32))

        # 计算相同值的百分比
        # allcount = tf.size(y_true_yz)
        blueCount = tf.size(y_true_yz[:, -1])
        readCount = tf.size(y_true_yz[:, :-1])

        per = tf.cast(((intersection_count / readCount) + (label_equal_count / blueCount) * 2) / 3, dtype=tf.float64)

        return per

    # 百分比验证
    @tf.function
    def cai3dloss(y_true_yz, y_pred_yz):
        """
        结果为三维的计算
        @param y_pred_yz:
        @return:
        """
        # 计算出（总匹配的数量+完全匹配*2)/3
        y_true = tf.cast(tf.round(y_true_yz), dtype=tf.int32)
        y_pred = tf.cast(tf.round(y_pred_yz), dtype=tf.int32)[:, -1:, :]
        ny = tf.abs(y_pred - y_true)
        n0 = tf.size(tf.reduce_sum(tf.where(condition=ny == 0), 1))
        d = tf.reduce_sum(tf.where(ny > 0, 1, 0), 2) / ny.shape[2]  # 完全匹配的数量
        dd = tf.size(tf.reduce_sum(tf.where(condition=d == 0), 1))
        allcount = tf.size(y_true)
        ddcount = tf.size(tf.reduce_sum(d, 1))
        per = ((tf.cast(1 - (dd / ddcount), dtype=tf.float64) * 2) + tf.cast(1 - (n0 / allcount), dtype=tf.float64)) / 3
        return per

    @tf.function
    def cai3dloss2(y_true_yz, y_pred_yz):
        """
        结果为二维的计算
        @param y_pred_yz:
        @return:
        """
        # 计算出（总匹配的数量+完全匹配*2)/3
        y_true = tf.cast(y_true_yz, dtype=tf.int32)
        y_pred = tf.cast(tf.round(y_pred_yz), dtype=tf.int32)
        ny = tf.abs(y_pred - y_true)
        n0 = tf.size(tf.reduce_sum(tf.where(condition=ny == 0), 1))
        dd = tf.reduce_sum(tf.cast(tf.reduce_all(tf.equal(y_true, y_pred), axis=1), dtype=tf.int32))  # 完全匹配的数量
        # 计算匹配的元素数量
        matching_elements = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), dtype=tf.int32), axis=1)
        # 计算匹配两个以上的数量
        count_matching_two_or_more = tf.reduce_sum(tf.cast(matching_elements >= 2, dtype=tf.int32))
        allcount = tf.size(y_true)
        ddcount = tf.cast(tf.shape(y_true)[0], dtype=tf.int32)

        per = (tf.cast(dd / ddcount, dtype=tf.float64) * 0.5) \
              + (tf.cast(count_matching_two_or_more / ddcount, dtype=tf.float64) * 0.3) \
              + (tf.cast(n0 / allcount, dtype=tf.float64) * 0.2)
        return per

    @tf.function
    def cai3dloss3(y_true_yz, y_pred_yz, full_match_weight=0.5, partial_match_weight=0.4, element_match_weight=0.1):
        """
        彩票损失函数，二维输出，MAX
        计算加权损失值：完全匹配(权重50%) + 部分匹配(权重30%) + 元素匹配(权重20%)
        
        Args:
            y_true_yz: 真实标签，形状为[batch_size, features]
            y_pred_yz: 预测值，形状为[batch_size, features]
            full_match_weight: 完全匹配的权重
            partial_match_weight: 部分匹配的权重
            element_match_weight: 元素匹配的权重
        
        Returns:
            加权损失值
        """
        # 四舍五入并转换为整数类型
        y_true = tf.cast(tf.round(y_true_yz), dtype=tf.int32)
        y_pred = tf.cast(tf.round(y_pred_yz), dtype=tf.int32)
        
        # 计算每个样本的元素匹配数量
        element_matches = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), dtype=tf.int32), axis=1)
        
        # 计算完全匹配的样本数（所有元素都匹配）
        full_matches = tf.reduce_sum(tf.cast(tf.reduce_all(tf.equal(y_true, y_pred), axis=1), dtype=tf.int32))
        
        # 计算部分匹配的样本数（匹配两个或更多元素）
        partial_matches = tf.reduce_sum(tf.cast(element_matches >= 2, dtype=tf.int32))
        
        # 计算总元素匹配数
        total_element_matches = tf.reduce_sum(element_matches)
        
        # 获取批次大小和总元素数
        batch_size = tf.cast(tf.shape(y_true)[0], dtype=tf.int32)
        total_elements = tf.cast(tf.size(y_true), dtype=tf.int32)
        
        # 计算加权损失
        per = (tf.cast(full_matches / batch_size, dtype=tf.float64) * full_match_weight) \
            + (tf.cast(partial_matches / batch_size, dtype=tf.float64) * partial_match_weight) \
            + (tf.cast(total_element_matches / total_elements, dtype=tf.float64) * element_match_weight)
        
        return per

    @tf.function
    def rmbloss(y_true_yz, y_pred_yz):
        y_true = tf.cast(y_true_yz, dtype=tf.float64)
        y = y_true_yz.shape[1]
        y_pred = tf.cast(y_pred_yz, dtype=tf.float64)[:, -y:, :]
        ny = tf.abs(y_pred - y_true)
        n0 = tf.size(tf.reduce_sum(tf.where(condition=ny <= 2), 1))
        allcount = tf.size(y_true)
        per = tf.cast(1 - (n0 / allcount), dtype=tf.float64)
        return per

    @tf.function
    def cuploss_z(y_true_yz, y_pred_yz):
        y_true = tf.cast(tf.round(y_true_yz), dtype=tf.int32)
        y_pred = tf.cast(tf.round(y_pred_yz), dtype=tf.int32)
        ny = tf.abs(y_pred - y_true)
        n0 = tf.size(tf.reduce_sum(tf.where(condition=ny == 0), 1))
        allcount = tf.size(y_true)
        per = tf.cast(1 - (n0 / allcount), dtype=tf.float64)
        return per

    @tf.function
    def cuploss(y_true_yz, y_pred_yz):
        # 计算出（结果匹配的数量+完全匹配*2)/3
        y_true = tf.cast(tf.round(y_true_yz), dtype=tf.int32)
        y_pred = tf.cast(tf.round(y_pred_yz), dtype=tf.int32)[:, -1:, :]
        ny = tf.abs(y_pred - y_true)
        ny_j = ny[:, :, 0] + ny[:, :, 1]  # 计算全组匹配-为零时
        ny_ja = tf.reduce_sum(tf.where(ny_j == 0, 1, 0), 1)
        ny_ja_d = tf.size(tf.reduce_sum(tf.where(condition=ny_ja == 1), 1))
        ny_ja_all = tf.size(tf.reduce_sum(ny_j, 1))
        ny_per = tf.cast(1 - (ny_ja_d / ny_ja_all), dtype=tf.float32)  # 完全匹配量
        # 计算结果
        sq_true = y_true[:, :, 0] - y_true[:, :, 1]
        sq_pred = y_pred[:, :, 0] - y_pred[:, :, 1]
        sq_true_ja = tf.reduce_sum(tf.where(sq_true > 0, 2, 0), 1)  # 真：主胜为2
        sq_pred_ja = tf.reduce_sum(tf.where(sq_pred > 0, 1, 0), 1)  # 测：主胜为1
        sq_ja_b1b = sq_true_ja - sq_pred_ja
        sq_ja_b1b_true = tf.size(tf.reduce_sum(tf.where(condition=sq_ja_b1b == 1), 1))  # 预测胜成功的数量
        sq_true_pi = tf.reduce_sum(tf.where(sq_true == 0, 2, 0), 1)  # 真：平为2
        sq_pred_pi = tf.reduce_sum(tf.where(sq_pred == 0, 1, 0), 1)  # 测：平为1
        sq_pi_b1b = sq_true_pi - sq_pred_pi
        sq_pi_b1b_true = tf.size(tf.reduce_sum(tf.where(condition=sq_pi_b1b == 1), 1))  # 预测平成功的数量
        sq_true_fu = tf.reduce_sum(tf.where(sq_true < 0, 2, 0), 1)  # 真：负为2
        sq_pred_fu = tf.reduce_sum(tf.where(sq_pred < 0, 1, 0), 1)  # 测：负为1
        sq_fu_b1b = sq_true_fu - sq_pred_fu
        sq_fu_b1b_true = tf.size(tf.reduce_sum(tf.where(condition=sq_fu_b1b == 1), 1))  # 预测负成功的数量
        sq_per = tf.cast(1 - ((sq_ja_b1b_true + sq_pi_b1b_true + sq_fu_b1b_true) / ny_ja_all),
                         dtype=tf.float32)  # 结果匹配量
        return (ny_per * 2 + sq_per) / 3

    def myacc(y_true, y_pred):
        """
        Computes classification accuracy
        """
        return 1 - K.mean(K.equal(K.argmax(y_true, axis=-1),
                                  K.argmax(y_pred, axis=-1)))

    def WitFitClaaBack(dset: bs._Dset):
        """
        连接训练停止方法
        @return:
        """
        witFitCallBack = EarlyStopping(monitor=dset.metrics_name, baseline=dset.WitFixCallValue, mode='min')
        witTestCallBack = EarlyStopping(monitor='val_' + dset.metrics_name, patience=5, baseline=dset.WitTestCallValue,
                                        mode='max')
        return witFitCallBack, witTestCallBack
        pass

    def FitCallBack(dset: bs._Dset = bs._Dset()) -> object:
        """
        回调保存模型
        @return: fitBackFun,testBackFun
        """
        fname = "{0}".format(time.strftime("%Y%m%d@%H%M%S", time.localtime())) + "_{loss:.4f}@{mape:.4f}@{" \
                                                                                 "val_loss:.4f}@{" \
                                                                                 "val_mape:.4f}_" + dset.guid
        fname = fname.replace("mape", dset.metrics_name)
        fitfile = dset.ModelFile + "F{0}".format(fname)
        testfile = dset.ModelFile + "T{0}".format(fname)
        bestfile = dset.ModelFile + "A{0}".format(fname)
        mode = 'min'
        if dset.valLossSaveMode != "":
            mode = dset.valLossSaveMode
        else:
            if dset.metrics_name == "mape" or dset.metrics_name == "cailoss":
                mode = 'min'
            if dset.metrics_name == "accuracy":
                mode = 'max'

        fitcallback = tf.keras.callbacks.ModelCheckpoint(
            filepath=fitfile,
            monitor=dset.metrics_name,
            save_best_only=True,
            save_weights_only=False,
            mode=mode,
            initial_value_threshold=dset.fixCallBackValue
            # save_freq='epoch'
        )
        testcallback = tf.keras.callbacks.ModelCheckpoint(
            filepath=testfile,
            monitor='val_' + dset.metrics_name,
            save_best_only=True,
            save_weights_only=False,
            mode=mode,
            initial_value_threshold=dset.textCallBackValue
            # save_freq='epoch'
        )

        checkpoint_callback = CustomModelCheckpoint(
            filepath=bestfile,
            monitor1='val_' + dset.metrics_name,  # 你要监测的指标1
            monitor2=dset.metrics_name,  # 你要监测的指标2
            mode=mode,  # 如果是 'min'，保存最小的值，如果是 'max'，保存最大的值
            monitor1_CheckValue=dset.textCallBackValue,  # 超出值时保存
            monitor2_CheckValue=dset.fixCallBackValue,
            verbose=1  # 显示保存信息
        )

        return [fitcallback, testcallback, checkpoint_callback]
        pass

    @tf.function
    def guaccuracy(y_true, y_pred):
        """
        自定义损失函数
        """
        y_true_class = tf.argmax(y_true, axis=1)
        y_pred_class = tf.argmax(y_pred, axis=1)
        correct = tf.reduce_sum(tf.cast(tf.equal(y_true_class, y_pred_class), tf.float32))
        custom_count = tf.reduce_sum(tf.cast(tf.logical_or(tf.logical_and(y_true_class <= 2, y_pred_class <= 2),
                                                           tf.logical_and(y_true_class >= 3, y_pred_class >= 3)),
                                             tf.float32))

        total_count = tf.cast(tf.shape(y_true)[0], tf.float32)
        accuracy = correct / total_count
        custom_ratio = custom_count / total_count
        loss = (accuracy * 0.4 + custom_ratio * 0.6)
        return loss


class CustomModelCheckpoint(Callback):
    """
    自定义保存监测类
    """

    def __init__(self, filepath, monitor1='val_loss', monitor2='loss', mode='min', monitor1_CheckValue=0,
                 monitor2_CheckValue=0, save_best_only=True, verbose=1):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor1 = monitor1
        self.monitor2 = monitor2
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.monitor1_CheckValue = monitor1_CheckValue
        self.monitor2_CheckValue = monitor2_CheckValue
        self.best_val_loss = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, epoch, logs=None):
        lossValue = logs.get('loss')
        v_lossValue = logs.get('val_loss')
        current_val_loss = logs.get(self.monitor1)
        current_loss = logs.get(self.monitor2)

        if current_val_loss is None:
            print("Warning: Can't save the best model. Validation loss is not available.")
        else:
            sum_loss = current_val_loss + current_loss

            if (self.mode == 'min' and sum_loss < self.best_val_loss and (
                    current_val_loss < self.monitor1_CheckValue or current_loss < self.monitor2_CheckValue)) or (
                    self.mode == 'max' and sum_loss > self.best_val_loss and (
                    current_val_loss > self.monitor1_CheckValue or current_loss > self.monitor2_CheckValue)):
                self.best_val_loss = sum_loss
                saveFile = self.filepath
                saveFile = saveFile.replace(self.monitor1, 'val_ZZloss').replace(self.monitor2, 'ZZloss')
                saveFile = saveFile.format(val_ZZloss=current_val_loss, ZZloss=current_loss, loss=lossValue,
                                           val_loss=v_lossValue)
                if self.verbose > 0:
                    print(
                        f"\nEpoch {epoch + 1}: {self.monitor1} + {self.monitor2} improved from {self.best_val_loss} to {sum_loss}, saving model to {saveFile}")
                self.model.save(saveFile, overwrite=True)
            else:
                if self.verbose > 0:
                    print(
                        f"\nEpoch {epoch + 1}: {self.monitor1} + {self.monitor2} did not improve from {self.best_val_loss}")
