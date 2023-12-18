/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __msg_h__
#define __msg_h__

#include <stdio.h>
#include <string>
#include <chrono>

#if 1//def __DEBUG__
#define MYMSGnonl(CSTR,lev) message(CSTR,false,lev)
#define MYMSG(CSTR,lev) message(CSTR,true,lev)
#define MYMSGBEGl(lvl) if(lvl<=VERBOSE){
#define MYMSGENDl }
#define MYMSGBEG if(1){
#define MYMSGEND }
#else
#define MYMSGnonl(ARG,lev)
#define MYMSG(ARG,lev)
#define MYMSGBEGl(lvl) if(0){
#define MYMSGENDl }
#define MYMSGBEG if(0){
#define MYMSGEND }
#endif

// typedefs
typedef int (*TPrintFunction)( void*, const char* format, ... );

// global variables used
extern const char*  PROGDIR;
extern const char*  PROGNAME;
extern const char*  PROGVERSION;
extern const char*  PROGREFERENCES[];

extern int*         __PARGC__;
extern char***      __PARGV__;

extern int          VERBOSE;
extern bool         WARNINGSRECORDED;

extern const
std::chrono::high_resolution_clock::time_point gtSTART;

// set global variables
void SetVerboseMode( int );
void SetProgramName( const char* name, const char* version = NULL );
void SetArguments( int* pargc, char*** pargv );

// messaging...
void error( const char*, bool = true );
void warning( const char*, bool = true, int minlevel = 1 );
void checkforwarnings();
void message( const char*, bool = true, int minlevel = 1  );
void progname_and_version( FILE* );
int print_cmdline(char* outstr);
void print_dtime(int minlevel);
size_t getdtime(char* tmstr, size_t sztmstr);

// string
std::string usage( const char* progname, const char* instructions, const char* version, const char* date );

// path processing
const char* my_basename( const char* );
const char* my_dirname( const char* );

// printing-to-stream interface
int file_print( void*, const char* format, ... );
int string_print( void*, const char* format, ... );

#endif//__msg_h__
