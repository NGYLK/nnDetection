#!/bin/bash

# 运行第一个命令
echo "Running: nndet_train 003 --sweep "
nndet_train 003 --sweep

# 检查第一个命令是否成功运行
if [ $? -ne 0 ]; then
    echo "First command failed. Exiting."
    exit 1
fi

# 运行第二个命令
echo "Running: nndet_train 002 -o exp.fold=2 --sweep"
nndet_train 002 -o exp.fold=2 --sweep

# 检查第二个命令是否成功运行
if [ $? -ne 0 ]; then
    echo "Second command failed. Exiting."
    exit 1
fi

echo "Both commands completed successfully."
