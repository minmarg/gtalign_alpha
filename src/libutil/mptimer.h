/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __mptimer_cuh__
#define __mptimer_cuh__

#include <ratio>
#include <chrono>

////////////////////////////////////////////////////////////////////////////
// CLASS MyMpTimer for measuring performance
//
// template<class Period = std::micro>
template<class Period = std::ratio<1>>
class MyMpTimer {
public:
    MyMpTimer() {}

    //start measuring performance
    void Start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    //stop the timer; measure the elapsed time
    void Stop() {
        stop_ = std::chrono::high_resolution_clock::now();
        elapsed_ = stop_- start_;
    }

    double GetElapsedTime() const {return elapsed_.count();}

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_, stop_;
    std::chrono::duration<double, Period> elapsed_;//time elapsed in us when using std::micro
};

#endif//__mptimer_cuh__
