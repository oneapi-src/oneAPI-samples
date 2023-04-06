#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>

#define MS_in_1SEC 1000.0

namespace util
{
    class Timer
    {
    public:
        void start()
        {
            StartTime = std::chrono::steady_clock::now();
            Running = true;
        }

        void stop()
        {
            EndTime = std::chrono::steady_clock::now();
            Running = false;
        }

        double elapsedMicroseconds()
        {
            std::chrono::time_point<std::chrono::steady_clock> endTime;
 
            if (Running)
            {
                endTime = std::chrono::steady_clock::now();
            }
            else
            {
                endTime = EndTime;
            }
 
            return double(std::chrono::duration_cast<std::chrono::microseconds>(endTime - StartTime).count());
        }

        double elapsedMilliseconds()
        {
            std::chrono::time_point<std::chrono::steady_clock> endTime;

            if (Running)
            {
                endTime = std::chrono::steady_clock::now();
            }
            else
            {
                endTime = EndTime;
            }

            return double(std::chrono::duration_cast<std::chrono::milliseconds>(endTime - StartTime).count());
        }

        double elapsedSeconds()
        {
            return elapsedMilliseconds() / MS_in_1SEC;
        }

    private:
        std::chrono::time_point<std::chrono::steady_clock> StartTime;
        std::chrono::time_point<std::chrono::steady_clock> EndTime;
        bool Running = false;
    };

} // namespace util

#endif //TIMER_H