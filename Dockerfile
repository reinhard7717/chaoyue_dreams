FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /workspace

# 安装MySQL开发依赖和pkg-config
RUN apt-get update && \
    apt-get install -y default-libmysqlclient-dev build-essential pkg-config tmux && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    # 修正pandas_ta的squeeze_pro.py兼容性
    && sed -i 's/from numpy import NaN as npNaN/from numpy import nan as npNaN/' /usr/local/lib/python3.11/dist-packages/pandas_ta/momentum/squeeze_pro.py
    # 默认启动命令（开发时一般不用，进入bash即可）
CMD ["bash"]


