

#include <algorithm>
#include <chrono>
#include <execution>
#include <iostream>
#include <numeric>
#include <vector>
template <typename TFunc>
void RunAndMeasure(const char* title, TFunc func) {
  const auto start = std::chrono::steady_clock::now();
  auto ret = func();
  const auto end = std::chrono::steady_clock::now();
  std::cout << title << ": "
            << std::chrono::duration<double, std::milli>(end - start).count()
            << " ms, res " << ret << "\n";
}

int main() {
  int size=1024000000;
  std::vector<double> v(1024000000, 0.5);
  std::vector<double> result(v.size());

  std::vector<double> v1(size);
  std::iota(v1.begin(), v1.end(), 1.0);


 RunAndMeasure("std::warm up", [&v] {
    return std::reduce(std::execution::seq, v.begin(), v.end(), 0.0);
  });

  RunAndMeasure("std::accumulate",
                [&v] { return std::accumulate(v.begin(), v.end(), 0.0); });

  RunAndMeasure("std::reduce, seq", [&v] {
    return std::reduce(std::execution::seq, v.begin(), v.end(), 0.0);
  });

  RunAndMeasure("std::reduce, par", [&v] {
    return std::reduce(std::execution::par, v.begin(), v.end(), 0.0);
  });

  RunAndMeasure("std::reduce, par_unseq", [&v] {
    return std::reduce(std::execution::par_unseq, v.begin(), v.end(), 0.0);
  });

  
  RunAndMeasure("std::find, seq", [&v] {
    auto res = std::find(std::execution::seq, std::begin(v), std::end(v), 0.6);
    return res == std::end(v) ? 0.0 : 1.0;
  });

  RunAndMeasure("std::find, par", [&v] {
    auto res = std::find(std::execution::par, std::begin(v), std::end(v), 0.6);
    return res == std::end(v) ? 0.0 : 1.0;
  });

   RunAndMeasure("std::find, par_unseq", [&v] {
    auto res = std::find(std::execution::par_unseq, std::begin(v), std::end(v), 0.6);
    return res == std::end(v) ? 0.0 : 1.0;
  }); 
    RunAndMeasure("std::copy_if, seq", [&v, &result] {
        auto new_end = std::copy_if(std::execution::seq, v.begin(), v.end(), result.begin(),
                                    [](double value) { return value > 0.4; });
        return std::distance(result.begin(), new_end);
    });

    RunAndMeasure("std::copy_if, par", [&v, &result] {
        auto new_end = std::copy_if(std::execution::par, v.begin(), v.end(), result.begin(),
                                    [](double value) { return value > 0.4; });
        return std::distance(result.begin(), new_end);
    });

    RunAndMeasure("std::copy_if, par_unseq", [&v, &result] {
        auto new_end = std::copy_if(std::execution::par_unseq, v.begin(), v.end(), result.begin(),
                                    [](double value) { return value > 0.4; });
        return std::distance(result.begin(), new_end);
    });

    RunAndMeasure("std::inclusive_scan, seq", [&v] {
        std::vector<double> scan_result(v.size());
        std::inclusive_scan(std::execution::seq, v.begin(), v.end(), scan_result.begin());
        return scan_result.back();
    });

    RunAndMeasure("std::inclusive_scan, par", [&v] {
        std::vector<double> scan_result(v.size());
        std::inclusive_scan(std::execution::par, v.begin(), v.end(), scan_result.begin());
        return scan_result.back();
    });

    RunAndMeasure("std::inclusive_scan, par_unseq", [&v] {
        std::vector<double> scan_result(v.size());
        std::inclusive_scan(std::execution::par_unseq, v.begin(), v.end(), scan_result.begin());
        return scan_result.back();
    });


    RunAndMeasure("std::min_element, seq", [&v1] {
        return *std::min_element(std::execution::seq, v1.begin(), v1.end());
    });

    RunAndMeasure("std::min_element, par", [&v1] {
        return *std::min_element(std::execution::par, v1.begin(), v1.end());
    });

    RunAndMeasure("std::min_element, par_unseq", [&v1] {
        return *std::min_element(std::execution::par_unseq, v1.begin(), v1.end());
    });

    RunAndMeasure("std::max_element, seq", [&v1] {
        return *std::max_element(std::execution::seq, v1.begin(), v1.end());
    });

    RunAndMeasure("std::max_element, par", [&v1] {
        return *std::max_element(std::execution::par, v1.begin(), v1.end());
    });

    RunAndMeasure("std::max_element, par_unseq", [&v1] {
        return *std::max_element(std::execution::par_unseq, v1.begin(), v1.end());
    });

    RunAndMeasure("std::minmax_element, seq", [&v1] {
        auto result = std::minmax_element(std::execution::seq, v1.begin(), v1.end());
        return *result.first + *result.second;
    });

    RunAndMeasure("std::minmax_element, par", [&v1] {
        auto result = std::minmax_element(std::execution::par, v1.begin(), v1.end());
        return *result.first + *result.second;
    });

    RunAndMeasure("std::minmax_element, par_unseq", [&v1] {
        auto result = std::minmax_element(std::execution::par_unseq, v1.begin(), v1.end());
        return *result.first + *result.second;
    });

    RunAndMeasure("std::is_partitioned, seq", [&v] {
        return std::is_partitioned(std::execution::seq, v.begin(), v.end(), [](double n) { return n < 1.0; });
    });

    RunAndMeasure("std::is_partitioned, par", [&v] {
        return std::is_partitioned(std::execution::par, v.begin(), v.end(), [](double n) { return n < 1.0; });
    });

    RunAndMeasure("std::is_partitioned, par_unseq", [&v] {
        return std::is_partitioned(std::execution::par_unseq, v.begin(), v.end(), [](double n) { return n < 1.0; });
    });

    RunAndMeasure("std::lexicographical_compare,  seq", [&v] {
        std::vector<double> v2(1024000000, 0.5);
        return std::lexicographical_compare(std::execution::seq, v.begin(), v.end(), v2.begin(), v2.end());
    });

    RunAndMeasure("std::lexicographical_compare, par", [&v] {
        std::vector<double> v2(1024000000, 0.5);
        return std::lexicographical_compare(std::execution::par, v.begin(), v.end(), v2.begin(), v2.end());
    });

    RunAndMeasure("std::lexicographical_compare, par_unseq", [&v] {
        std::vector<double> v2(1024000000, 0.5);
        return std::lexicographical_compare(std::execution::par_unseq, v.begin(), v.end(), v2.begin(), v2.end());
    });

    RunAndMeasure("std::binary_search", [&v] {
        return std::binary_search( v.begin(), v.end(), 0.5);
    });

    RunAndMeasure("std::lower_bound", [&v1] {
        return *std::lower_bound(v1.begin(), v1.end(), 0.5);
    });

    RunAndMeasure("std::upper_bound", [&v1] {
        return *std::upper_bound( v1.begin(), v1.end(), 0.5);
    });

  return 0;
}
