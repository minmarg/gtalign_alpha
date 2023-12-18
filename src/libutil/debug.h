/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __debug_h__
#define __debug_h__

#include <string.h>
#include "platform.h"
#include "preproc.h"

// #define __DEBUG__
#define __MYSRCFILENAME__ \
    (strstr(__FILE__,DIRSEPSTR "src" DIRSEPSTR)? \
     strstr(__FILE__,DIRSEPSTR "src" DIRSEPSTR)+1: \
     __FILE__)
#define __EXCPOINT__ __MYSRCFILENAME__,__LINE__,__func__


#endif//__debug_h__
