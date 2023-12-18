/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __mytype_h__
#define __mytype_h__

#include "platform.h"

typedef unsigned long long int uint64_mt;

#ifndef ssize_t
#ifdef OS_MS_WINDOWS
        typedef __int64 ssize_t;
#else
//      typedef long long int ssize_t;
#endif
#endif

#ifndef uint
        typedef unsigned int uint;
#endif

#endif//__mytype_h__
