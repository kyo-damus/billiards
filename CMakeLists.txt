cmake_minimum_required(VERSION 3.12)
project(BilliardsEnv)

# 昔のC++の書き方を許容するオプション
add_compile_options(-fpermissive)

find_package(pybind11 REQUIRED)

# GSLライブラリの場所を指定
include_directories($ENV{HOME}/.local/include)
link_directories($ENV{HOME}/.local/lib)

# FastFizのヘッダーを指定
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/fastfiz/FastFiz-0.2/FastFiz)

# モジュールの作成（※AIBase.cppを削除）
pybind11_add_module(billiard_env_cpp 
    cpp_simulator.cpp
    libs/fastfiz/FastFiz-0.2/FastFiz/FastFiz.cpp
    libs/fastfiz/FastFiz-0.2/FastFiz/Rules.cpp
    libs/fastfiz/FastFiz-0.2/FastFiz/Noise.cpp
    libs/fastfiz/FastFiz-0.2/FastFiz/Stopwatch.cpp
    libs/fastfiz/FastFiz-0.2/FastFiz/LogFile.cpp
)

# ライブラリのリンク
target_link_libraries(billiard_env_cpp PRIVATE gsl gslcblas m)