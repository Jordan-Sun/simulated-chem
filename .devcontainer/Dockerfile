ARG TARGETPLATFORM=linux/amd64
FROM --platform=$TARGETPLATFORM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye
# Update pip
RUN pip3 install --upgrade pip
# Python libraries
RUN pip3 install numpy pandas
RUN pip3 install ortools
RUN pip3 install matplotlib seaborn animation
RUN pip3 install netCDF4

# Copy scipoptsuite package to the container
COPY SCIPOptSuite-9.2.0-Linux-ubuntu24.deb /tmp/

# Install dependencies for scipoptsuite
RUN apt-get update && apt-get install -y \
    libgfortran5 \
    libreadline8 \
    libncurses5 \
    liblapack3 \
    libblas3

# Install scipoptsuite
RUN dpkg -i /tmp/SCIPOptSuite-9.2.0-Linux-ubuntu24.deb || apt-get install -f -y

# Remove the .deb package
RUN rm /tmp/SCIPOptSuite-9.2.0-Linux-ubuntu24.deb

# install scip python api
RUN pip3 install pyscipopt