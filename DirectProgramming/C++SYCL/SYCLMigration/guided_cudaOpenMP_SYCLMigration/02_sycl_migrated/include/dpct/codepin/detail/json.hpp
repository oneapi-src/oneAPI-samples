//==---- json.hpp ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_JSON_HPP__
#define __DPCT_JSON_HPP__
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <inttypes.h>
#include <iostream>
#include <map>
#include <memory>
#include <string.h>
#include <string>
#include <sys/types.h>
#include <type_traits>
#include <utility>
#include <vector>

#define error_exit(msg)                                                        \
  {                                                                            \
    std::cerr << "Failed at:" << __FILE__ << "\nLine number is : " << __LINE__ \
              << "\n" msg << std::endl;                                        \
    std::exit(-1);                                                             \
  }
namespace dpct {
namespace experimental {
namespace detail {
namespace dpct_json {
class value;
class array {
public:
  array() = default;
  void push_back(value &&v) { arrayVec.push_back(std::move(v)); }
  void push_back(const value &v) { arrayVec.push_back(v); }
  value &back() { return arrayVec.back(); }
  auto end() { return arrayVec.end(); }
  auto begin() { return arrayVec.begin(); }
  size_t size() { return arrayVec.size(); }
  value &operator[](int index) { return arrayVec[index]; }

private:
  std::vector<value> arrayVec;
};

class object {
public:
  object() = default;
  value &operator[](const std::string &key);
  value &operator[](std::string &&key);
  const value &get(const std::string &key);
  const value &get(std::string &&key);
  size_t size() { return obj_map.size(); }
  bool contains(const std::string &str);

private:
  std::map<std::string, value> obj_map;
};

class value {
public:
  enum type {
    int_t,
    float_t,
    boolean_t,
    string_t,
    object_t,
    array_t,
    nullptr_t
  };

  template <class T, class... U> void create_in_union(U &&...V) {
    // Use placement new to generate the union share type.
    new (reinterpret_cast<T *>(&union_val)) T(std::forward<U>(V)...);
  };

  template <class T, typename = std::enable_if_t<std::is_integral_v<T>>,
            typename = std::enable_if_t<!std::is_floating_point_v<T>>,
            typename = std::enable_if_t<!std::is_same_v<T, bool>>>
  value(T v) : real_type(int_t) {
    create_in_union<int>((int)v);
  }

  template <class T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
  value(T v) : real_type(float_t) {
    create_in_union<double>((double)v);
  }

  template <class T, typename = std::enable_if_t<std::is_same_v<T, bool>>,
            bool = false>
  value(T v) : real_type(boolean_t) {
    create_in_union<bool>((bool)v);
  }

  value(const std::nullptr_t &null) : real_type(nullptr_t) {}
  value(std::nullptr_t &&null = nullptr) : real_type(nullptr_t) {}
  value(const value &v) { copy_mem(v); }
  value(value &&v) { copy_mem(v); }
  value(std::string &v) : real_type(string_t) {
    create_in_union<std::string>(v);
  }
  value(dpct_json::object &v) : real_type(object_t) {
    create_in_union<dpct_json::object>(v);
  }
  value(dpct_json::object &&v) : real_type(object_t) {
    create_in_union<dpct_json::object>(std::move(v));
  }
  value(dpct_json::array &v) : real_type(array_t) {
    create_in_union<dpct_json::array>(std::move(v));
  }
  value(dpct_json::array arr) : real_type(array_t) {
    create_in_union<dpct_json::array>(arr);
  }

  void copy_mem(const value &v) {
    real_type = v.real_type;
    switch (v.real_type) {
    case int_t:
    case float_t:
    case boolean_t:
      memcpy(&union_val, &v.union_val, sizeof(union_val));
      break;
    case array_t:
      create_in_union<dpct_json::array>(v.get_value<dpct_json::array>());
      break;
    case object_t:
      create_in_union<dpct_json::object>(v.get_value<dpct_json::object>());
      break;
    case string_t:
      create_in_union<std::string>(v.get_value<std::string>());
      break;
    case nullptr_t:
      // To do
      break;
    default:
      error_exit("[JSON Parser]: Parsingd unkown value type.\n");
    }
  }
  value &operator=(const value &v) {
    clear();
    copy_mem(v);
    return *this;
  }
  template <class T> T &get_value() const {
    return *static_cast<T *>((void *)union_val);
  }
  ~value() { clear(); }
  type real_type;

private:
  void clear() {
    switch (real_type) {
    case int_t:
    case float_t:
    case boolean_t:
    case nullptr_t:
      break;
    case string_t:
      get_value<std::string>().~basic_string();
      break;
    case object_t:
      get_value<dpct_json::object>().~object();
      break;
    case array_t:
      get_value<dpct_json::array>().~array();
      break;
    default:
      error_exit("[JSON Parser]: Parsingd unkown value type.");
    }
  }

  char union_val[sizeof(
      std::aligned_union_t<1, int, double, bool, std::string, dpct_json::array,
                           dpct_json::object>)];
};

static std::string json_key = "";
class json_parser {
private:
  const char *begin = nullptr;
  const char *cur_p = nullptr;
  const char *end = nullptr;

public:
  json_parser(const std::string &json)
      : begin(json.c_str()), cur_p(json.c_str()),
        end(json.c_str() + json.size()) {}
  bool parse_value(value &v);
  bool parse_str(std::string &ret);
  bool parse_num(char first, int64_t &out);
  void ignore_space() {
    while (cur_p != end && (*cur_p == ' ' || *cur_p == '\t' || *cur_p == '\r' ||
                            *cur_p == '\n'))
      cur_p++;
  }
  char next() { return cur_p != end ? *cur_p++ : 0; }
  char peek() { return cur_p != end ? *cur_p : 0; }
};

inline bool parse(const std::string &json, dpct_json::value &v);

inline value &object::operator[](const std::string &key) {
  if (!contains(key)) {
    return obj_map.try_emplace(key, nullptr).first->second;
  } else {
    return obj_map[key];
  }
}

inline value &object::operator[](std::string &&key) {
  if (!contains(key)) {
    return obj_map.try_emplace(key, nullptr).first->second;
  } else {
    return obj_map[key];
  }
}

inline const value &object::get(const std::string &key) {
  if (contains(key))
    return obj_map[key];
  error_exit("[JSON Parser]: Parsing JSON to cpp failed. Missing key: " + key +
             " in the JSON file.");
}

inline const value &object::get(std::string &&key) {
  if (contains(key))
    return obj_map[key];
  error_exit("[JSON Parser]: Parsing JSON to cpp failed. Missing key: " + key +
             " in the JSON file.");
}

inline bool object::contains(const std::string &str) {
  if (obj_map.find(str) != obj_map.end())
    return true;
  return false;
}

inline bool json_parser::parse_str(std::string &str) {
  while (peek() != '"') {
    if (cur_p == end) {
      return false;
    }
    str += next();
  }
  next();
  return true;
}

inline static bool is_number(char c) {
  return (c == '0') || (c == '1') || (c == '2') || (c == '3') || (c == '4') ||
         (c == '5') || (c == '6') || (c == '7') || (c == '8') || (c == '9') ||
         (c == '-') || (c == '+') || (c == '.') || (c == 'e') || (c == 'E');
}

inline bool json_parser::parse_num(char first, int64_t &out) {
  std::string val = "";
  val += first;
  while (is_number(peek())) {
    val += next();
  }
  char *end;
  out = strtoimax(val.c_str(), &end, 10);
  if (end == (val.c_str() + val.size())) {
    return true;
  }
  error_exit("[JSON Parser]: Parsing JSON value error. The key is " + json_key);
}

inline bool json_parser::parse_value(value &v) {
  ignore_space();
  switch (char c = next()) {
  case '[': {
    v = dpct_json::array();
    dpct_json::array &arr = v.get_value<dpct_json::array>();
    ignore_space();
    if (peek() == ']') {
      ++cur_p;
      return true;
    }
    for (;;) {
      ignore_space();
      arr.push_back(nullptr);
      if (!parse_value(arr.back())) {
        return false;
      }
      ignore_space();
      switch (next()) {
      case ',':
        continue;
      case ']':
        return true;
      default:
        error_exit("[JSON Parser]: Parsing JSON value error. The key is " +
                   json_key);
      }
    }
    break;
  }
  case '{': {
    v = dpct_json::object();
    dpct_json::object &obj = v.get_value<dpct_json::object>();
    for (;;) {
      std::string key = "";
      ignore_space();

      if (next() == '"') {
        if (!parse_str(key)) {
          error_exit("[JSON Parser]: key value of a JSON need to be wrapped in "
                     "\". Please check the JSON file format.");
        } else {
          json_key = key;
        }
      }
      ignore_space();
      if (next() == ':') {
        ignore_space();
        obj[key] = nullptr;
        if (!parse_value(obj[key])) {
          error_exit("[JSON Parser]: Can not parse value, the JSON key is " +
                     key + ".\n");
        }
      }
      ignore_space();
      switch (next()) {
      case ',': {
        continue;
      }
      case '}': {
        return true;
      }
      default:
        error_exit("[JSON Parser]: The " + json_key +
                   " value pair should be end with '}' or ','.\n")
      }
    }
    break;
  }
  case '"': {
    std::string str = "";
    parse_str(str);
    v = str;
    return true;
  }
  case 't':
    if (next() == 'r' && next() == 'u' && next() == 'e') {
      v = true;
      return true;
    }
    error_exit("[JSON Parser]: The bool value of " + json_key +
               " should be \"true\", please check "
               "the spelling.");
  case 'f':
    if (next() == 'a' && next() == 'l' && next() == 's' && next() == 'e') {
      v = false;
      return true;
    }
    error_exit("[JSON Parser]: The bool value of " + json_key +
               " should be \"false\", please "
               "check the spelling.");
  default:
    if (is_number(c)) {
      int64_t value;
      parse_num(c, value);
      v = value;
      return true;
    }
    error_exit("[JSON Parser]: Unkown JSON type, the last key is " + json_key +
               ". Please check the format JSON "
               "format.\n");
  }
}

} // namespace dpct_json
} // namespace detail
} // namespace experimental
} // namespace dpct
#endif
