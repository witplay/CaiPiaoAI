# -*- coding: utf-8 -*-
import pymssql
import config

class GetSqlData(object):
    def getAllData(TableName, dtwhere='', orderby=''):
        try:
            db_config = config.get_config()['db']
            connect = pymssql.connect(db_config['host'], db_config['user'], db_config['password'], db_config['database'], charset='utf8')  # 服务器名,账户,密码,数据库名
            if not connect:
                print("数据库连接失败!")
            cursor = connect.cursor()  # 创建一个游标对象,python里的sql语句都要通过cursor来执行
            if TableName.lower().count('delete') > 0 or TableName.lower().count(
                    'select') > 0 or TableName.lower().count('update') > 0 or TableName.lower().count(
                    'insert') > 0 or TableName.lower().count('exec') > 0:
                sql = ("{}").format(TableName)
            else:
                sql = ("select * from {} {} order by {}").format(TableName, dtwhere, orderby)
            row = cursor.execute(sql)  # 执行sql语句
            if sql.lower().count('select') > 0 and sql.lower().count('insert') < 1 and sql.lower().count(
                    'update') < 1 and TableName.lower().count('exec') < 1:
                row = cursor.fetchall()  # 读取查询结果,
            else:
                row = cursor.rowcount
                connect.commit()  # 提交
                pass
            cursor.close()
            connect.close()
            return row
        except Exception as err:
            print(err)
            pass

    def getSqlTitle(sql):
        try:
            db_config = config.get_config()['db']
            connect = pymssql.connect(db_config['host'], db_config['user'], db_config['password'], db_config['database'], charset='utf8')  # 服务器名,账户,密码,数据库名
            cursor = connect.cursor()  # 创建一个游标对象,python里的sql语句都要通过cursor来执行
            if sql.lower().count('where') > 0:
                sql = sql[0:sql.index('where')]
            if sql.lower().count('order') > 0:
                sql = sql[0:sql.index('order')]
            sql += " where 1=2"
            row = cursor.execute(sql)  # 执行sql语句
            title = [i[0] for i in cursor.description]
            cursor.close()
            connect.close()
            return title
        except Exception as err:
            print(err)
            return ''
            pass

    def getcon():
        db_config = config.get_config()['db']
        connect = pymssql.connect(db_config['host'], db_config['user'], db_config['password'], db_config['database'], charset='utf8')  # 服务器名,账户,密码,数据库名
        return connect

# data=getAllData(conn)#获取数据库数据
