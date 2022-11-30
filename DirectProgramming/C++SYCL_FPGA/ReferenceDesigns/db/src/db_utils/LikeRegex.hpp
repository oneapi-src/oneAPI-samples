#ifndef __LIKEREGEX_HPP__
#define __LIKEREGEX_HPP__
#pragma once

#include <array>
#include <type_traits>

//
// Regex LIKE engine that can match: *WORD*, *WORD and WORD*
// where the string has max_str_length words or length at most max_word_length.
// If words are shorter than max_word_length, they are padded with '\0'.
//
template <unsigned int max_word_length, unsigned int max_str_length>
class LikeRegex {
  // static asserts
  static_assert(max_word_length < max_str_length,
    "The maximum word length must be less than the maximum string length");
  static_assert(max_word_length > 0,
    "The word must have a positive non-zero maximum length");
  static_assert(max_str_length > 0,
    "The string must have a positive non-zero maximum length");

 public:
  void Match() {
    // find true length of string
    str_true_len = GetStrLength();

    // find true length of WORD
    word_true_len = GetWordLength();

    // determine if there is a match
    match_start_idx = max_str_length;
    match_end = max_str_length;

    #pragma unroll
    for (unsigned int i = 0; i < max_str_length; i++) {
      // check if str[i:i+max_word_length] matches word
      bool matches = true;

      #pragma unroll
      for (unsigned int j = 0; j < max_word_length; j++) {
        if ((i + j < max_str_length) && (word[j] != '\0') &&
            (str[i + j] != '\0') && (word[j] != str[i + j])) {
          matches = false;
          break;
        }
      }

      if (matches && (i < str_true_len) &&
          ((i + word_true_len) <= str_true_len)) {
        match_start_idx = i;
        match_end = i + word_true_len;
      }
    }
  }

  // determines the true length of the input string (null terminated)
  unsigned int GetStrLength() const {
    unsigned int len = 0;

    #pragma unroll
    for (unsigned int i = 0; i < max_str_length; i++) {
      if (len == 0 && str[i] == '\0') {
        len = i;
      }
    }

    return len;
  }

  // determines the true length of the regex word (null terminated)
  unsigned int GetWordLength() const {
    unsigned int len = 0;

    #pragma unroll
    for (unsigned int i = 0; i < max_word_length; i++) {
      if (len == 0 && word[i] == '\0') {
        len = i;
      }
    }

    return len;
  }

  // does the string contain the word (i.e. matches %WORD%)
  bool Contains() {
    return (match_start_idx < max_str_length) && (match_end < max_str_length);
  }

  // does the string start with the word (i.e. matches WORD%)
  bool AtStart() {
    return match_start_idx == 0;
  }

  // does the string end with the word (i.e. matches %WORD)
  bool AtEnd() {
    return match_end == str_true_len;
  }

  char word[max_word_length];
  char str[max_str_length];

  unsigned int match_start_idx, match_end;
  unsigned int word_true_len, str_true_len;
};

#endif /* __LIKEREGEX_HPP__ */
