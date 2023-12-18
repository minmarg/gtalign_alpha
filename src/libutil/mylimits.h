/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __mylimits_h__
#define __mylimits_h__

#define LOG_PROB_MIN ( -32767 )
#define SCORE_MIN ( -32767 )

#ifdef UINT_MAX
#   define MYSIZE_MAX UINT_MAX
#else
#   define MYSIZE_MAX 4294967295
#endif

//1024,1024*1024,1024*1024*1024
#define ONEK       1024
#define ONEM    1048576
#define ONEG 1073741824UL

#ifndef INT_MAX
#   define INT_MAX 2147483647
#endif

#ifndef UINT_MAX
#   define UINT_MAX 4294967295U
#endif

// default size for a short string
#define BUF_MAX     256
#define KBYTE       1024



// maximum number of columns a sequence or profile can have;
// if exceeded the program terminates
#define MAXCOLUMNS  50000

// alignment output width per line
#define OUTPUTWIDTH 60

// output indent
#define OUTPUTINDENT 13

// length of annotations
#define ANNOTATIONLEN 75

// max description length in annotation
#define ANNOTATION_DESCLEN 40
// ANNOTATION_DESCLEN minus the length of '...'
#define ANNOTATION_DESCLEN_3l 37

// alignment width for output
#define DEFAULT_ALIGNMENT_WIDTH 80

// max length of filenames to show in output
#define MAX_FILENAME_LENGTH_TOSHOW 128

// description length saved (>4!!)
#define DEFAULT_DESCRIPTION_LENGTH 128

// maximum description length in the program's output
#define MAX_DESCRIPTION_LENGTH 4096



#endif//__mylimits_h__

