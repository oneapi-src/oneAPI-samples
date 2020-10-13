#ifndef __LIKEREGEX_HPP__
#define __LIKEREGEX_HPP__
#pragma once

#include <array>
#include <type_traits>

//
// Regex LIKE engine that can match: *WORD*, *WORD and WORD*
// where the string has MaxStrLength words or length at most MaxWordLength.
// If words are shorter than MaxWordLength, they are padded with '\0'.
//
template <unsigned int MaxWordLength, unsigned int MaxStrLength>
class LikeRegex {
  // static asserts
  static_assert(MaxWordLength < MaxStrLength,
    "The maximum word length must be less than the maximum string length");
  static_assert(MaxWordLength > 0,
    "The word must have a positive non-zero maximum length");
  static_assert(MaxStrLength > 0,
    "The string must have a positive non-zero maximum length");

 public:
  void Match() {
    // find true length of string
    str_true_len = GetStrLength();

    // find true length of WORD
    word_true_len = GetWordLength();

    // determine if there is a match
    match_start_idx = MaxStrLength;
    match_end = MaxStrLength;

    #pragma unroll
    for (unsigned int i = 0; i < MaxStrLength; i++) {
      // check if str[i:i+MaxWordLength] matches word
      bool matches = true;

      #pragma unroll
      for (unsigned int j = 0; j < MaxWordLength; j++) {
        if ((i + j < MaxStrLength) && (word[j] != '\0') &&
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
    for (unsigned int i = 0; i < MaxStrLength; i++) {
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
    for (unsigned int i = 0; i < MaxWordLength; i++) {
      if (len == 0 && word[i] == '\0') {
        len = i;
      }
    }

    return len;
  }

  // does the string contain the word (i.e. matches %WORD%)
  bool Contains() {
    return (match_start_idx < MaxStrLength) && (match_end < MaxStrLength);
  }

  // does the string start with the word (i.e. matches WORD%)
  bool AtStart() {
    return match_start_idx == 0;
  }

  // does the string end with the word (i.e. matches %WORD)
  bool AtEnd() {
    return match_end == str_true_len;
  }

  char word[MaxWordLength];
  char str[MaxStrLength];

  unsigned int match_start_idx, match_end;
  unsigned int word_true_len, str_true_len;
};

#endif /* __LIKEREGEX_HPP__ */
