/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __platform_h__
#define __platform_h__


// #if defined(__NVCC__) || defined(__CUDACC__)
// #else
// #define __device__
// #define __host__
// #endif


#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)

#define OS_MS_WINDOWS
#define DIRSEP      '\\'
#define DIRSEPSTR   "\\"
#define NEWLINE     "\r\n"
#define NL          "\r\n"
#define NLlast      "\n" //last character of NL as string
#define NLlastchr   '\n' //last character of NL as string

#else   //__linux__, __GNUC__, __unix__, __APPLE__, other

#if defined(__APPLE__)
#define OS_MAC
#endif

#define DIRSEP      '/'
#define DIRSEPSTR   "/"
#define NEWLINE     "\n"
#define NL          "\n"
#define NLlast      "\n" //last character of NL as string
#define NLlastchr   '\n' //last character of NL as string

#endif//if defined(WIN32)...

#define UPDIR ".."


#endif//__platform_h__
