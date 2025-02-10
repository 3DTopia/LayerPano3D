#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>

class Timer
{
public:
    /**
     * start timer.
     * @return (void)
     */
    void start()
    {
        start_time = std::chrono::system_clock::now();
        timer_running = true;
    }

    /**
     * Stop timer.
     * @return (void)
     */
    void stop()
    {
        end_time = std::chrono::system_clock::now();
        timer_running = false;
    }

    /**
     * Return the duration since start.
     * @return the duration in millisecond
     */
    double duration_ms()
    {
        std::chrono::time_point<std::chrono::system_clock> end_time_;

        if (timer_running)
        {
            end_time_ = std::chrono::system_clock::now();
        }
        else
        {
            end_time_ = end_time;
        }

        return std::chrono::duration_cast<std::chrono::milliseconds>(end_time_ - start_time).count();
    }

    /**
     * Return the duration since start.
     * @return the duration in second
     */
    double duration_s()
    {
        return duration_ms() / 1000.0;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start_time;

    std::chrono::time_point<std::chrono::system_clock> end_time;
    
    bool timer_running = false; // true, the timer is started.
};
