if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # linux
        sudo apt-get install autoconf automake libtool curl make g++ unzip -y
        git clone https://github.com/google/protobuf.git
        cd protobuf
        git submodule update --init --recursive
        ./autogen.sh
        ./configure
        make
        make check
        sudo make install
        sudo ldconfig
elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac OSX
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        brew install protoc
elif [[ "$OSTYPE" == "cygwin" ]]; then
        # POSIX compatibility layer and Linux environment emulation for Windows
        echo "protoc install in $OSTYPE is not implemented"
elif [[ "$OSTYPE" == "msys" ]]; then
        # Lightweight shell and GNU utilities compiled for Windows (part of MinGW)
        echo "protoc install in $OSTYPE is not implemented"
elif [[ "$OSTYPE" == "win32" ]]; then
        echo "protoc install in $OSTYPE is not implemented"
elif [[ "$OSTYPE" == "freebsd"* ]]; then
        echo "protoc install in $OSTYPE is not implemented"
else
        echo "$OSTYPE is not recognized"
fi
