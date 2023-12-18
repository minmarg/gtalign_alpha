/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __macros_h__
#define __macros_h__

#define ALIGN_DOWN( offset, alignment ) \
    ( (offset) & ~((alignment) - 1) )

#define ALIGN_UP( offset, alignment ) \
    ( ((offset) + (alignment) - 1) & ~((alignment) - 1) )

#define PCMIN( a, b )   (( a ) < ( b ) ? ( a ) : ( b ))
#define PCMAX( a, b )   (( a ) > ( b ) ? ( a ) : ( b ))
#define TIMES2( arg )   (( arg )+( arg ))
#define TIMES3( arg )   ( TIMES2( arg )+( arg ))
#define TIMES4( arg )   ( TIMES3( arg )+( arg ))
#define TIMES5( arg )   ( TIMES4( arg )+( arg ))
#define SQRD( arg )     (( arg )*( arg ))
#define CUBE( arg )     (( arg )*( arg )*( arg ))

#define INT2FLOAT(VAL) INT2FLOAT_helper(VAL)
#define INT2FLOAT_helper(VAL) (VAL ## . ## f)

#endif//__macros_h__
