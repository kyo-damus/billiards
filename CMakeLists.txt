# CMakeLists.txt
cmake_minimum_required(VERSION 3.12)
project(BilliardsEnv)

# pybind11のパッケージを検索
find_package(pybind11 REQUIRED)

# Pythonモジュールの作成
pybind11_add_module(billiard_env_cpp cpp_simulator.cpp)
