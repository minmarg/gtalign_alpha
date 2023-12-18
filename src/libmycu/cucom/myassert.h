/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __myassert_h__
#define __myassert_h__

#include <assert.h>
#include "libutil/mybase.h"

#ifdef __DEBUG__
#if defined(__CUDA_ARCH__)
#define CUMSG(ARG,lvl)
#if !defined(OS_MAC)
#define MYASSERT(COND,MSG) assert(COND);
#else
#define MYASSERT(COND,MSG)
#endif//!OS_MAC
#else
#define CUMSG(ARG,lvl) message(ARG,true,lvl)
#define MYASSERT(COND,MSG) if(!(COND)) throw MYRUNTIME_ERROR(MSG);
#endif//__CUDA_ARCH__
#else
#define CUMSG(ARG,lvl)
#define MYASSERT(COND,MSG)
#endif//__DEBUG__

#define CUSWAP(T,a,b) do {T _my_tmp_=a; a=b; b=_my_tmp_;} while (0);
#define CNDSWAP(T,CND,a,b) if(CND) {T _my_tmp_=a; a=b; b=_my_tmp_;}

#endif//__myassert_h__
