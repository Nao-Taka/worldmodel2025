# CUDA 12.8 + Python 3.11 環境
# nvidia/以下のcuda~から、Docker hubから探してくるといいでしょう
FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

# 作業ディレクトリ
# WORKDIR /workspace/src

# タイムゾーンの選択を自動化する
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && echo "Asia/Tokyo" > /etc/timezone

# 必要なツールをインストール
# 比較的新しいバージョンのPythonはaptの公式のライブラリに存在しません
# なので新しいPythonが必要な場合はそれがあるディストリビューションからとってきます
# セキュリティ上の問題がある場合は自力でコンパイルすことになるかと思います
RUN apt-get update && apt-get install -y \
    sudo \
    software-properties-common \
    wget \
    git \
    build-essential \
    curl \
    python3.12 \
    python3.12-venv \
    python3-pip \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    python3-pil \
    python3-tk \
    python3-pil.imagetk \
    tk-dev \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

#sixelの設定
RUN cd /tmp &&\
    git clone https://github.com/saitoha/libsixel.git &&\
    cd libsixel &&\
    git checkout develop &&\
    ./configure --disable-python &&\
    make &&\
    make install
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# ---- user ----
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000

RUN if ! getent group $USER_GID >/dev/null; then \
        groupadd -g $USER_GID $USERNAME; \
    fi \
 && if ! getent passwd $USER_UID >/dev/null; then \
        useradd -m -u $USER_UID -g $USER_GID -s /bin/bash $USERNAME; \
    else \
        usermod -a -G $USER_GID $(getent passwd $USER_UID | cut -d: -f1); \
    fi \
 && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
 && chmod 0440 /etc/sudoers.d/$USERNAME


# USER $USERNAME
WORKDIR /workspace/src

# Pythonのシンボリックリンクを作成
# インストールしたPythonのバージョンと同じものを
# RUN ln -s $(which python3.12) /usr/bin/python

# pipのインストール（distutils なしで問題なく動作する方法）
# RUN curl -sSL https://bootstrap.pypa.io/get-pip.py | python

#requirements.txtをコンテナ内にコピーする
COPY requirements.txt ./

# 必要なPythonライブラリをインストール
#RUN pip install --upgrade --break-system-packages pip
#RUN pip install --no-cache-dir -r --break-system-packages requirements.txt
# RUN pip install numpy
#RUN mkdir /workspace/src/testDir
#RUN cd / && ls
# RUN python3 -m venv /workspace/src/venv \
#     && /workspace/src/venv/bin/pip install --upgrade pip \
#     && /workspace/src/venv/bin/pip install --no-cache-dir -r requirements.txt
RUN python3.12 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip \
    && /opt/venv/bin/python -m pip install --no-cache-dir -r requirements.txt \
    && /opt/venv/bin/python -m pip install sixel　\
    && chmod -R a+rX /opt/venv
COPY sixel.py /opt/venv/lib/python3.12/site-packages/sixel.py

#RUN python3.12 -m venv /workspace/src/venv \
#    && /workspace/src/venv/bin/pip install --upgrade pip \
#    && /workspace/src/venv/bin/pip install --no-cache-dir -r requirements.txt
#仮想環境の有効化
# CMD ["./venv/bin/python"]

ENV PATH=/opt/venv/bin:$PATH

# Gitの設定を行う
RUN git config --global core.editor 'code --wait'


