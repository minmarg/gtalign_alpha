/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __localconfig_h__
#define __localconfig_h__

#include "libutil/preproc.h"
#include "libutil/platform.h"

#define CONCATSTRSEP(arg1)      CONCATSTRA(arg1, DIRSEPSTR)

#if !defined(GTALIGNSTATEDIR)
#   error   "Unable to compile: Undefined configuration directory (GTALIGNSTATEDIR)."
#endif

#define VARDIR var
#define PARAM_DIRNAME  GTALIGNSTATEDIR

#define PARAM_FILENAME gtalign.par
#define PARAM_FULLNAME CONCATSTRSEP( PARAM_DIRNAME ) TOSTR( PARAM_FILENAME )

static const char*  var_param_DIR = TOSTR( VARDIR );
static const char*  var_param_DIRNAME = TOSTR( PARAM_DIRNAME );

static const char*  var_param_FILENAME = TOSTR( PARAM_FILENAME );
static const char*  var_param_FULLPATHNAME = PARAM_FULLNAME;

inline const char* GetParamDirectory()      {   return var_param_DIR;  }
inline const char* GetFullParamDirname()    {   return var_param_DIRNAME;  }

inline const char* GetParamFilename()       {   return var_param_FILENAME;  }
inline const char* GetFullParamFilename()   {   return var_param_FULLPATHNAME;  }

#endif//__localconfig_h__
