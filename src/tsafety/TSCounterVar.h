/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __TSCounterVar_h__
#define __TSCounterVar_h__

#include <mutex>
#include <condition_variable>

class TSCounterVar
{
public:
    TSCounterVar()
    : counter_(0)
    {}
    void reset()
    {
        {   std::lock_guard<std::mutex> lck(mtx_);
            counter_ = 0;
        }
        cv_.notify_one();
    }
    void set(int value)
    {
        {   std::lock_guard<std::mutex> lck(mtx_);
            counter_ = value;
        }
        cv_.notify_one();
    }
    int isset() const
    {
        return (get() > 0);
    }
    int get() const
    {
        std::lock_guard<std::mutex> lck(mtx_);
        int value = counter_;
        return value;
    }
    void inc()
    {
        std::lock_guard<std::mutex> lck(mtx_);
        counter_++;
    }
    void dec()
    {
        {   std::lock_guard<std::mutex> lck(mtx_);
            counter_--;
        }
        cv_.notify_one();
    }
    void wait0()
    {
        std::unique_lock<std::mutex> lck(mtx_);
        cv_.wait(lck, [this]{return counter_ < 1;});
    }

    std::mutex& get_mutex() {return mtx_;}

    void reset_under_lock() {counter_ = 0;}

    void inc_under_lock() {counter_++;}

    int get_under_lock() const {return counter_;}

    void set_under_lock(int value) {counter_ = value;}

private:
    mutable std::mutex mtx_;
    std::condition_variable cv_;
    int counter_;
};

#endif//__TSCounterVar_h__
