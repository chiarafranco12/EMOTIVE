import mysql.connector

MYSQL_USER = "root"
MYSQL_PWD = ""
MYSQL_DB = "Emotive"

cnx = mysql.connector.connect(user=MYSQL_USER, password=MYSQL_PWD, host='localhost',database=MYSQL_DB, auth_plugin='mysql_native_password')
c = cnx.cursor(dictionary=True)

def mysql_query(sql, var):
    c.execute(sql, var)
    
    return c.fetchall()

def mysql_insert(sql, var):
    c.execute(sql, var)
    cnx.commit()

if __name__ == '__main__':
    sql = "select * from esempio"
    print(mysql_query(sql))
