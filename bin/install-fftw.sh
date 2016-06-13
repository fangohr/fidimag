#!/bin/bash

# This script installs SUNDIALS and FFTW locally. It may need to environment
# variables to work, like 'export CC=gcc' in ARCHER.

FFTW=fftw-3.3.4

set -e

# Create target directory if needed.

HERE_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
FIDIMAG_DIR="$(dirname "$HERE_DIR")"
LIBS_DIR=${FIDIMAG_DIR}/local
mkdir -p $LIBS_DIR
cd ${LIBS_DIR}
echo "Installing FFTW to "$LIBS_DIR"."
echo "Using CC="$CC" "

download_and_install() {
    # $1 name of the package
    # $2 URL where ${1}.tar.gz can be obtained
    # $3 configure options
    if [ ! -e ${1}.tar.gz ]; then
        echo "Downloading "${1}"."
        wget -q ${2}/${1}.tar.gz
    fi;

    if [ ! -e ${1} ]; then
        tar -xzf ${1}.tar.gz
        cd ${1}
        echo "Configuring "${1}"."
        ./configure --quiet --enable-shared --prefix=${LIBS_DIR} $3
        echo "Compiling and installing "${1}"."
        {
            make
            make install
        } > /dev/null
        echo "Done."
        cd ${LIBS_DIR}
    fi;
}

download_and_install ${FFTW} http://www.fftw.org --enable-openmp

echo "Installation succesful."
