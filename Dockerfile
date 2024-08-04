FROM mapllle/pytorch-j-ownuse:v4

# 拷贝当前项目到/app目录下（.dockerignore中文件除外）
COPY . /app

# 设定当前的工作目录
WORKDIR /app

# 暴露端口
EXPOSE 80

RUN ["pip", "install", "-r", "requirements.txt"]

# 执行启动命令
CMD ["python3", "run.py", "0.0.0.0", "80"]



