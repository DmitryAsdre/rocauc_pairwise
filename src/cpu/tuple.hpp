#ifndef TUPLE_HPP
#define TUPLE_HPP
#include <cstddef>
#include <tuple>

template<typename T_1, typename T_2, typename T_3, typename T_4>
using tuple_4 = std::tuple<T_1, T_2, T_3, T_4>;

template<typename T>
T get4(tuple_4<T, T, T, T> t, size_t i){
    return std::get<i>(t);
}

#endif