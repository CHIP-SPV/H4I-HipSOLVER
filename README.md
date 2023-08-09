# Building

Example:

        mkdir build && cd build
        . /opt/intel/oneapi/setvars.sh
        cmake .. -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_INSTALL_PREFIX=$HOME/local/stow/H4I-HipSOLVER
        make all install

