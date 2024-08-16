#pragma once

#include <chrono>
#include "common/PrintDebug.h"

class Timer {
private:
  std::chrono::high_resolution_clock::time_point start_, end_;
  double time_consumed_ms_double_;

public:
  Timer()  = default;
  ~Timer() = default;
  void tic() {
    start_ = std::chrono::high_resolution_clock::now();
  }

  void toc() {
    end_ = std::chrono::high_resolution_clock::now();
    time_consumed_ms_double_ = std::chrono::duration<double, std::milli>(end_ - start_).count();
  }

  double getDuration() const {
    return time_consumed_ms_double_;
  }

  double tocGetDuration() {
    end_ = std::chrono::high_resolution_clock::now();
    time_consumed_ms_double_ = std::chrono::duration<double, std::milli>(end_ - start_).count();
    return time_consumed_ms_double_;
  }
};

/**
 * 统计代码运行时间
 * @tparam FuncT
 * @param func  被调用函数
 * @param func_name 函数名
 */
template <typename FuncT>
void evaluate_and_call(FuncT&& func, const std::string& func_name = "") {
  double total_time = 0;
  auto t1 = std::chrono::high_resolution_clock::now();
  func();
  auto t2 = std::chrono::high_resolution_clock::now();
  total_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;
  
  PRINT_DEBUG("> [%s] cost time %.3f ms\n", func_name, total_time);
}