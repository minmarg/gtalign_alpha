/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __preproc_h__
#define __preproc_h__

#define STR( arg )                  #arg
#define TOSTR( arg )                STR( arg )
#define CONCAT( arg1, arg2 )        arg1 ## arg2
#define CONCATSTR( arg1, arg2 )     CONCAT( arg1, arg2 )
#define CONCATSTRA( arg1, arg2 )    TOSTR( arg1 ) arg2

#define DO_PRAGMA_(x) _Pragma(#x)
#define DO_PRAGMA(x) DO_PRAGMA_(x)

#endif//__preproc_h__
