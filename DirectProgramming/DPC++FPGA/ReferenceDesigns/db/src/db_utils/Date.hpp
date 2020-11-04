#ifndef __DATE_HPP__
#define __DATE_HPP__
#pragma once

#include <fstream>
#include <sstream>
#include <string>

//
// Class for storing and computing operations on a date
//
class Date {
 public:
  Date(const int y, const int m, const int d) : year(y), month(m), day(d) {}
  Date(std::string date_str) {
    // parse money string
    std::istringstream ss(date_str);
    std::string year_str, month_str, day_str;

    std::getline(ss, year_str, '-');
    std::getline(ss, month_str, '-');
    std::getline(ss, day_str, '-');

    year = atoi(year_str.c_str());
    month = atoi(month_str.c_str());
    day = atoi(day_str.c_str());
  }

  //
  // determine if the date is valid
  //
  bool Valid() const {
    if (year <= 0 || month > 12 || month <= 0 || day > 31 || day <= 0) {
      return false;
    }

    if (month == 4 || month == 6 || month == 9 || month == 11) {
      if (day > 30) {
        return false;
      }
    } else if (month == 1 || month == 3 || month == 5 || month == 7 ||
               month == 8 || month == 10 || month == 12) {
      if (day > 31) {
        return false;
      }
    } else /* month == 4 (February) */ {
      if (day > 29) {
        return false;
      } else if (day == 29 && !((((year % 4 == 0) && (year % 100 != 0)) ||
                                 (year % 400 == 0)))) {
        return false;
      }
    }

    return true;
  }

  //
  // get the previous day
  //
  Date PreviousDay(const Date& d) {
    if (!d.Valid()) {
      return d;
    } else {
      auto newDate = Date(d.year, d.month, d.day - 1);
      if (newDate.Valid()) return newDate;
      newDate = Date(d.year, d.month - 1, 31);
      if (newDate.Valid()) return newDate;
      newDate = Date(d.year, d.month - 1, 30);
      if (newDate.Valid()) return newDate;
      newDate = Date(d.year, d.month - 1, 29);
      if (newDate.Valid()) return newDate;
      newDate = Date(d.year, d.month - 1, 28);
      if (newDate.Valid()) return newDate;
      newDate = Date(d.year - 1, 12, 31);
      if (newDate.Valid()) return newDate;
      newDate = Date(31, 12, d.year - 1);
      return newDate;
    }
  }

  //
  // get the next day
  //
  Date NextDay(const Date& d) {
    if (!d.Valid()) {
      return d;
    } else {
      auto newDate = Date(d.year, d.month, d.day + 1);
      if (newDate.Valid()) return newDate;
      newDate = Date(d.year, d.month + 1, 1);
      if (newDate.Valid()) return newDate;
      newDate = Date(d.year + 1, 1, 1);
      if (newDate.Valid()) return newDate;

      return newDate;
    }
  }

  //
  // get a date a certain number of days in the future
  //
  Date LaterDate(const int days) {
    Date newDate = *this;
    for (unsigned int i = 0; i < days; i++) {
      newDate = NextDay(newDate);
    }
    return newDate;
  }

  //
  // get a date a certain number of days in the past
  //
  Date PreviousDate(const int days) {
    Date newDate = *this;
    for (unsigned int i = 0; i < days; i++) {
      newDate = PreviousDay(newDate);
    }
    return newDate;
  }

  Date operator++() {
    Date d = *this;
    *this = NextDay(d);
    return d;
  }

  Date operator++(int) {
    *this = NextDay(*this);
    return *this;
  }

  Date operator--() {
    Date d = *this;
    *this = PreviousDay(d);
    return d;
  }

  Date operator--(int) {
    *this = PreviousDay(*this);
    return *this;
  }

  //
  // Date in a compressed format
  //      DAY bits     = ceil(lg(31)) = 5
  //      MONTH bits   = ceil(lg(12)) = 4
  //      YEAR bits    = 32-5-4 = 23
  //      DATE format is:
  //      32                      8  4    0
  //      YYYYYYYYYYYYYYYYYYYYYYYMMMMDDDDD
  //
  unsigned int ToCompact() const {
    unsigned int y = (unsigned int)year;
    unsigned int m = (unsigned int)month;
    unsigned int d = (unsigned int)day;
    return (((y)&0x07FFFFF) << 9) | (((m)&0x000F) << 5) | ((d)&0x001F);
  }

  void FromCompact(const unsigned int date) {
    year = ((date >> 9) & 0x07FFFFF);
    month = ((date >> 5) & 0x000F);
    day = (date & 0x001F);
  }

  int year, month, day;
};

#endif /* __DATE_HPP__ */
