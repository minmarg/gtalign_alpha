/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __templates_h__
#define __templates_h__

// absolute value of the difference:
template <typename T>
T GetAbsDiff(T v1, T v2) {return (v1 < v2)? v2 - v1: v1 - v2;}

//simple swap template function
template <typename T> void myswap(T& a, T& b) { T c(a); a=b; b=c; }

//simple template functions for squares
template <typename T> T mysqrd(T& a){ return a*a; }
template <typename T> T mysqrdv(T a){ return a*a; }

//min/max
template <typename T> T mymax(T a, T b){ return a<b?b:a; }
template <typename T> T mymin(T a, T b){ return a<b?a:b; }

template <typename T> constexpr T mycemax(T a, T b){ return a<b?b:a; }
template <typename T> constexpr T mycemin(T a, T b){ return a<b?a:b; }

//conditional assignment
template <typename T, typename T2>
void mymaxassgn(T& a, T b, T2& c, T2 d){ if(a<b){a=b; c=d;} }

#endif//__templates_h__
