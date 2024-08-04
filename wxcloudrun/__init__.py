from flask import Flask
import config

# 因MySQLDB不支持Python3，使用pymysql扩展库代替MySQLDB库
#pymysql.install_as_MySQLdb()

# 初始化web应用
app = Flask(__name__, instance_relative_config=True)
app.config['DEBUG'] = config.DEBUG

# 加载控制器
from wxcloudrun import views

# 加载配置
app.config.from_object('config')
