FROM ubuntu:18.04
 
RUN apt-get update \
  && apt-get install -y build-essential cmake vim \
  && apt-get install -y python3-dev python3-pip \
  && pip3 install pytest \
  && apt-get install -y xz-utils unzip \
  && apt-get install -y curl \
  # install pybind11
  && curl -LJO https://github.com/pybind/pybind11/archive/v2.5.0.tar.gz \
  && tar -zxf pybind11-2.5.0.tar.gz \
  && cd pybind11-2.5.0 \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make \
  && make install \
  && cd / \
  # install cpp-taskflow
  && curl -LJO https://github.com/cpp-taskflow/cpp-taskflow/archive/v2.4.0.tar.gz \
  && tar -zxf cpp-taskflow-2.4.0.tar.gz \
  && cd cpp-taskflow-2.4.0 \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make \
  && make install \
  && cd / \
  # install spdlog
  && curl -LJO https://github.com/gabime/spdlog/archive/v1.5.0.tar.gz \
  && tar -zxf spdlog-1.5.0.tar.gz \
  && cd spdlog-1.5.0 \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make \
  && make install \
  && cd / \
  # install armadillo
  && apt-get install -y libopenblas-dev liblapack-dev libarpack2-dev \
  && curl -LJO http://sourceforge.net/projects/arma/files/armadillo-9.860.1.tar.xz \
  && tar -vxf armadillo-9.860.1.tar.xz \
  && cd armadillo-9.860.1 \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make \
  && make install \
  && cd / \
  # clean cache
  && rm -rf /var/lib/apt/lists/* 

CMD [ "/bin/bash" ]
