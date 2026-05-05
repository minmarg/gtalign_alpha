/***************************************************************************
 *   Copyright (C) 2021-2025 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __mymutex_h__
#define __mymutex_h__


__device__ __forceinline__ 
void LOCK(unsigned int* mutex)
{
    //unsigned int ns = 8;
    while(atomicCAS(mutex, 0, 1) == 1) {
        //__nanosleep(ns);
        //if(ns < 256)
        //    ns *= 2;
    }
    // for(int cc = 0; atomicCAS(mutex, 0, 1) != 0 && cc < 1024*1024; cc++);
}

__device__ __forceinline__ 
void UNLOCK(unsigned int* mutex)
{
    atomicExch(mutex, 0);
}

// lock class;
// NOTE: uses an additional register
class UNIQUE_LOCK
{
public:
    __device__ __forceinline__ 
    UNIQUE_LOCK(/*volatile */unsigned int* mutex): mutex_(mutex) {lock();}

    __device__ __forceinline__ 
    ~UNIQUE_LOCK() {unlock();}

    __device__ __forceinline__ void lock() {LOCK(mutex_);}
    __device__ __forceinline__ void unlock() {UNLOCK(mutex_);}
private:
    /*volatile */unsigned int* mutex_;
};


#endif//__mymutex_h__
