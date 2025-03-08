# Use the nvidia official Ubuntu 22.04 as a base image
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Add a non-root user
ARG REMOTE_USER
ARG REMOTE_UID
ARG REMOTE_GID
RUN <<EOF
    addgroup --gid ${REMOTE_GID} ${REMOTE_USER}
    adduser --disabled-password --uid ${REMOTE_UID} --gid ${REMOTE_GID} ${REMOTE_USER}
EOF
RUN apt-get update && apt-get install -y sudo \
    && echo "${REMOTE_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${REMOTE_USER} \
    && chmod 0440 /etc/sudoers.d/${REMOTE_USER}
ENV HOME /home/${REMOTE_USER}

# Install usufull packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    udev \
    wget \
    usbutils \
    curl \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    build-essential \
    libusb-1.0-0-dev \
    git \
    cmake \
    unzip \
    yasm \
    software-properties-common

RUN apt-get install -y \
    libtbb2 \
    libtbb-dev \
    libpng-dev \
    libtiff-dev \
    libpq-dev \
    libxine2-dev \
    libglew-dev \
    libtiff5-dev \
    zlib1g-dev \
    libjpeg-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libpostproc-dev \
    libswscale-dev \
    libeigen3-dev \
    libgtk2.0-dev \
    pkg-config

ARG OPENCV_VERSION=4.9.0

RUN cd /opt/ && wget https://github.com/opencv/opencv/archive/4.9.0.zip &&\
    unzip $OPENCV_VERSION.zip &&\
    rm $OPENCV_VERSION.zip &&\
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    rm ${OPENCV_VERSION}.zip &&\
    mkdir /opt/opencv-${OPENCV_VERSION}/build

RUN cd /opt/opencv-${OPENCV_VERSION}/build && cmake \
    -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
    -DWITH_CUDA=ON \
    -DCUDA_ARCH_BIN=7.5,8.0,8.6 \
    -DWITH_CUDNN=ON \
    -DWITH_CUBLAS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCUDA_FAST_MATH=ON \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DWITH_TBB=ON \
    -DWITH_OPENMP=ON \
    -DWITH_IPP=ON \
    -DBUILD_EXAMPLES=OFF \
    -DWITH_NVCUVID=OFF \
    -DWITH_NVCUVENC=OFF \
    -DBUILD_DOCS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_TESTS=OFF \
    -DWITH_CSTRIPES=ON \
    -DWITH_OPENCL=ON \
    -DWITH_GSTREAMER=ON \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -ffast-math" \
    # Install path will be /usr/local/lib (lib is implicit)
    -DCMAKE_INSTALL_PREFIX=/usr/local ..

RUN cd /opt/opencv-${OPENCV_VERSION}/build && make -j12 ; make -j10 ; make -j8 ; make -j1

RUN cd /opt/opencv-${OPENCV_VERSION}/build &&\
    make install &&\ 
    ldconfig
# END OpenCV installation

# Install Pyenv
RUN curl -fsSL https://pyenv.run | bash

RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc \
    echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc \
    echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile \
    echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile \
    echo 'eval "$(pyenv init - bash)"' >> ~/.profile \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile \
    echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile \
    echo 'eval "$(pyenv init - bash)"' >> ~/.bash_profile

RUN apt update && apt install -y \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev

ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN pyenv install 3.11.11
Run pyenv global 3.11.11

RUN chown -R ${REMOTE_USER}:${REMOTE_USER} $HOME/.pyenv

# Install Pre-commit and Git LFS
RUN pip3 install pre-commit autopep8 pytest
RUN apt update && apt install git-lfs -y

# Install other python dependencies
RUN pip3 install numpy==2.2.3 cython
RUN pip3 install torch
RUN pip3 install ultralytics
RUN pip3 install scipy \
    matplotlib \
    pandas \
    scikit-learn \
    numba \
    filterpy
RUN pip3 install pyrealsense2
RUN pip3 install --extra-index-url https://pypi.nvidia.com tensorrt==10.0.1 tensorrt-cu12==10.0.1

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/workspace/GKRealSense"
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/

USER ${REMOTE_USER}
WORKDIR /workspace
