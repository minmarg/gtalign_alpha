/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __cutemplates_h__
#define __cutemplates_h__

#include "libmycu/cucom/cudef.h"

//simple swap template function
template <typename T> __HDINLINE__
void myhdswap( T& a, T& b ){ T c(a); a=b; b=c; }

//simple swap template function
template <typename T> __HDINLINE__
T myhdsqrd( T& a ){ return a*a; }
template <typename T> __HDINLINE__
T myhdsqrdv( T a ){ return a*a; }

template <typename T> __HDINLINE__
T myhdmax( T a, T b ){ return a<b?b:a; }

template <typename T> __HDINLINE__
T myhdmin( T a, T b ){ return a<b?a:b; }

template <typename T, typename T2> __HDINLINE__
void myhdmaxassgn( T& a, T b, T2& c, T2 d ){ if(a<b){a=b; c=d;} }

#endif//__cutemplates_h__
