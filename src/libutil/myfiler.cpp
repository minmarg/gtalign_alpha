/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>

#include <string.h>
#include <stdlib.h>
#include <errno.h>

#include <string>

#include "platform.h"
#include "mylimits.h"
#include "myfiler.h"

#ifdef OS_MS_WINDOWS
#	include <direct.h>
#	include <malloc.h>
#endif

// -------------------------------------------------------------------------
// file_exists: check if file exists; return true if it does
//
bool file_exists( const char* name,
#ifdef OS_MS_WINDOWS
	unsigned short
#else
	mode_t
#endif
	mode )
{
#ifdef OS_MS_WINDOWS
    struct _stat64 info;
#else
    struct stat info;
#endif
    bool exists = false;

    if( name ) {
#ifdef OS_MS_WINDOWS
        if( _stat64( name, &info ) < 0 ) {
#else
        if( stat( name, &info ) < 0 ) {
#endif
            errno = 0;//reset error
            return exists;
        }

#ifdef OS_MS_WINDOWS
        if(( _S_IFMT & info.st_mode ) == mode )
#else
        if(( S_IFMT & info.st_mode ) == mode )
#endif
            exists = true;
    }

    return exists;
}

// -------------------------------------------------------------------------
// file_size: get the file size for a given filename
//
int file_size(const char* filename, size_t* size)
{
#ifdef OS_MS_WINDOWS
    struct _stat64 info;
#else
    struct stat info;
#endif
    int retcode = -1;

    if(filename) {
#ifdef OS_MS_WINDOWS
        if((retcode = _stat64(filename, &info)) < 0) {
#else
        if((retcode = stat(filename, &info)) < 0) {
#endif
            errno = 0;//reset error
            return retcode;
        }

        //NOTE: no check of valid address
        *size = (size_t)info.st_size;
    }

    return retcode;
}

// -------------------------------------------------------------------------
// mymkdir: make a directory; returns -1 on error;
//
int mymkdir(const char* pathname)
{
#ifdef OS_MS_WINDOWS
    return _mkdir(pathname);
#else
    return mkdir(pathname, 0775/*mode*/);
#endif
}



// -------------------------------------------------------------------------
// skip_comments: skip comments at the current position of file
//
int skip_comments( FILE* fp, char* buffer, size_t bufsize, size_t* readlen, char cc )
{
    size_t  len;
    bool    fulline = true;
    char*   p = buffer;

    const size_t    lsize = KBYTE;
    char            locbuf[lsize] = {0};
    char*           locp = locbuf;
    size_t          loclen;

    if( !fp || !buffer || !readlen )
        return( ERR_SC_MACC );

    while( !feof( fp )) {
        p = fgets( buffer, (int)bufsize, fp );

        len = *readlen = strlen( buffer );

        if( p == NULL && feof( fp ))
            break;

        if( p == NULL && ferror( fp ))
            return( ERR_SC_READ );

        if( fulline ) {
            for( p = buffer; *p == ' ' || *p == '\t'; p++ );
            if( *p != cc && *p != '\n' && *p != '\r' && len )
            {
                if( buffer[len-1] != '\n' && buffer[len-1] != '\r' ) {
                    //read line further till the end of line
                    while( !feof( fp ) && ( locp = fgets( locbuf, lsize, fp )))
                    {
                        loclen = strlen( locbuf );
                        if( loclen && ( locbuf[loclen-1] == '\n' || locbuf[loclen-1] == '\r' ))
                            break;
                    }
                    if( locp == NULL && ferror( fp ))
                        return( ERR_SC_READ );
                }
                break;
            }
        }
        if( len && ( buffer[len-1] != '\n' && buffer[len-1] != '\r' ))
            fulline = false;
    }
    return 0;
}

// -------------------------------------------------------------------------
// skip_comments: skip comments and read full line from file into buffer
//
int skip_comments( FILE* fp, std::string& buffer, char cc )
{
    size_t  len;
    bool    fulline = true;
    char*   p;

    const size_t    lsize = KBYTE;
    char            locbuf[lsize] = {0};
    char*           locp = locbuf;
    size_t          loclen;

    buffer.erase();

    if( !fp )
        return( ERR_SC_MACC );

    while( !feof( fp )) {
        p = fgets( locbuf, lsize, fp );

        len = strlen( locbuf );

        if( p == NULL && feof( fp ))
            break;

        if( p == NULL && ferror( fp ))
            return( ERR_SC_READ );

        if( fulline ) {
            for( p = locbuf; *p == ' ' || *p == '\t'; p++ );
            if( *p != cc && *p != '\n' && *p != '\r' && len )
            {
                buffer = locbuf;

                if( locbuf[len-1] != '\n' && locbuf[len-1] != '\r' ) {
                    //read line further till the end of line
                    while( !feof( fp ) && ( locp = fgets( locbuf, lsize, fp )))
                    {
                        loclen = strlen( locbuf );
                        if( loclen ) {
                            buffer += locbuf;
                            if( locbuf[loclen-1] == '\n' || locbuf[loclen-1] == '\r' )
                                break;
                        }
                    }
                    if( locp == NULL && ferror( fp ))
                        return( ERR_SC_READ );
                }
                break;
            }
        }
        if( len && ( locbuf[len-1] != '\n' && locbuf[len-1] != '\r' ))
            fulline = false;
    }
    return 0;
}

// -------------------------------------------------------------------------
// skip_comments: a template version of skip_comments that reads a full 
// line from a character stream;
// source, source character stream;
// func, functor;
// buffer, argument of the functor corresponding to an address or buffer;
// length, lenght of data to be written on output;
// cc, comment character;
//
template<typename F, typename A1>
inline
int skip_comments( TCharStream* source, F func, A1* arg, size_t* length, char cc )
{
    char* p;
    int pos;

    if( source == NULL )
        return( ERR_SC_MACC );

    if( source->data_ == NULL || source->datlen_ < 1 )
        return 0;

    if( source->datlen_ <= source->curpos_ )
        return ERR_SC_MACC;

    for( p = source->data_ + source->curpos_; source->curpos_ < source->datlen_; ) 
    {
        for( ; source->curpos_ < source->datlen_ && 
            (*p==' ' || *p=='\t' || *p=='\n' || *p=='\r'); p++, source->curpos_++ );

        if( *p == cc ) {
            for( ; source->curpos_ < source->datlen_ && (*p!='\n' && *p!='\r'); p++, source->curpos_++ );
            for( ; source->curpos_ < source->datlen_ && (*p=='\n' || *p=='\r'); p++, source->curpos_++ );
            continue;
        }

        pos = (int)source->curpos_;

        for( ; source->curpos_ < source->datlen_ && 
            (*p!=cc && *p!='\n' && *p!='\r'); p++, source->curpos_++ );

        func(source->data_ + pos, source->curpos_ - pos, arg, length);

        for( ; source->curpos_ < source->datlen_ && (*p=='\n' || *p=='\r'); p++, source->curpos_++ );

        break;
    }

    return 0;
}

// -------------------------------------------------------------------------
// skip_comments: a version of skip_comments that reads a full 
// line from a character stream into the buffer
//
int skip_comments( TCharStream* source, std::string& buffer, char cc )
{
    buffer.erase();
    return skip_comments(source, CopyToBuffer(), &buffer, NULL, cc);
}

// -------------------------------------------------------------------------
// skip_comments: a version of skip_comments that reads a full 
// line from a character stream and sets the current position of data;
// source, character stream;
// char*& ptr, output pointer to data to be set;
// size_t, unused parameter of the size allocated for ptr;
// length, length of data starting from address ptr;
// cc, comment character;
//
int skip_comments( TCharStream* source, char*& ptr, size_t, size_t* length, char cc )
{
    *length = 0;
    return skip_comments(source, GetPtr(), &ptr, length, cc);
}





// -------------------------------------------------------------------------
// read_double: read double value
//
int read_double( const char* readfrom, size_t readlen, double* membuf, size_t* rbytes )
{
    const char* p = readfrom;
    const char* pbeg = NULL;
    const char* pend = NULL;

    double      tmpval;
    char*       paux;
	int errcode = 0;

    if( !readfrom || !membuf )
        return( ERR_RD_MACC );

    for( ; *p == ' ' || *p == '\t'; p++ );
    for( pbeg = p, pend = p + readlen; p < pend &&
        *p && *p != ' ' && *p != '\t' && 
        *p != ';' && *p != ',' && *p != '(' && *p != ')' &&
        *p != '\n' && *p != '\r'; p++ );

    if( pbeg == p )
        return( ERR_RD_NOVL );

    const size_t    numlen = size_t( p - pbeg );
#ifdef OS_MS_WINDOWS
	char* number = (char*)_malloca(numlen + 1);
#else
	char number[numlen+1];
#endif
    memcpy( number, pbeg, numlen );
    number[numlen] = 0;

    errno = 0;//NOTE:
    tmpval = strtod( number, &paux );

	errcode = (errno || *paux);

#ifdef OS_MS_WINDOWS
	_freea(number);
#endif

	if( errcode )
		return( ERR_RD_INVL );

    *membuf = tmpval;

    if( rbytes )
        *rbytes = size_t( p - readfrom );

    return 0;
}

// -------------------------------------------------------------------------
// read_float: read single-precision value
//
int read_float( const char* readfrom, size_t readlen, float* membuf, size_t* rbytes )
{
    const char* p = readfrom;
    const char* pbeg = NULL;
    const char* pend = NULL;

    float tmpval;
    char* paux;
	int errcode = 0;

    if( !readfrom || !membuf )
        return( ERR_RD_MACC );

    for( ; *p == ' ' || *p == '\t'; p++ );
    for( pbeg = p, pend = p + readlen; p < pend &&
        *p && *p != ' ' && *p != '\t' && 
        *p != ';' && *p != ',' && *p != '(' && *p != ')' &&
        *p != '\n' && *p != '\r'; p++ );

    if( pbeg == p )
        return( ERR_RD_NOVL );

    const size_t    numlen = size_t( p - pbeg );
#ifdef OS_MS_WINDOWS
	char* number = (char*)_malloca(numlen + 1);
#else
	char number[numlen + 1];
#endif

    memcpy( number, pbeg, numlen );
    number[numlen] = 0;

    errno = 0;//NOTE:
    tmpval = strtof( number, &paux );

	errcode = (errno || *paux);

#ifdef OS_MS_WINDOWS
	_freea(number);
#endif

	if( errcode )
        return( ERR_RD_INVL );

    *membuf = tmpval;

    if( rbytes )
        *rbytes = (size_t)( p - readfrom );

    return 0;
}

// -------------------------------------------------------------------------
// read_integer: read integer value
//
int read_integer( const char* readfrom, size_t readlen, int* membuf, size_t* rbytes )
{
    const char* p = readfrom;
    const char* pbeg = NULL;
    const char* pend = NULL;

    int         tmpval;
    char*       paux;
	int errcode = 0;

    if( !readfrom || !membuf )
        return( ERR_RI_MACC );

    for( ; *p == ' ' || *p == '\t'; p++ );
    for( pbeg = p, pend = p + readlen; p < pend &&
        *p && *p != ' ' && *p != '\t' && 
        *p != ';' && *p != '(' && *p != ')' &&
        *p != '\n' && *p != '\r'; p++ );

    if( p <= pbeg )
        return( ERR_RI_NOVL );

    const size_t    numlen = size_t( p - pbeg );
#ifdef OS_MS_WINDOWS
	char* number = (char*)_malloca(numlen + 1);
#else
	char number[numlen + 1];
#endif

    memcpy( number, pbeg, numlen );
    number[numlen] = 0;

    errno = 0;//NOTE:
    tmpval = strtol( number, &paux, 10 );

	errcode = (errno || *paux);

#ifdef OS_MS_WINDOWS
	_freea(number);
#endif

	if( errcode )
        return( ERR_RI_INVL );

    *membuf = tmpval;

    if( rbytes )
        *rbytes = size_t( p - readfrom );

    return 0;
}

// -------------------------------------------------------------------------
// read_llinteger: read long long integer value
//
int read_llinteger( const char* readfrom, size_t readlen, long long int* membuf, size_t* rbytes )
{
    const char* p = readfrom;
    const char* pbeg = NULL;
    const char* pend = NULL;

    long long int   tmpval;
    char*           paux;
	int errcode = 0;

    if( !readfrom || !membuf )
        return( ERR_RL_MACC );

    for( ; *p == ' ' || *p == '\t'; p++ );
    for( pbeg = p, pend = p + readlen; p < pend &&
        *p && *p != ' ' && *p != '\t' && 
        *p != ';' && *p != '(' && *p != ')' &&
        *p != '\n' && *p != '\r'; p++ );

    if( p <= pbeg )
        return( ERR_RL_NOVL );

    const size_t    numlen = size_t( p - pbeg );
#ifdef OS_MS_WINDOWS
	char* number = (char*)_malloca(numlen + 1);
#else
	char number[numlen + 1];
#endif

    memcpy( number, pbeg, numlen );
    number[numlen] = 0;

    errno = 0;//NOTE:
    tmpval = strtoll( number, &paux, 10 );

	errcode = (errno || *paux);

#ifdef OS_MS_WINDOWS
	_freea(number);
#endif

	if( errcode )
        return( ERR_RL_INVL );

    *membuf = tmpval;

    if( rbytes )
        *rbytes = size_t( p - readfrom );

    return 0;
}

// -------------------------------------------------------------------------
// read_symbol: read single character
//
int read_symbol( const char* readfrom, size_t readlen, char* membuf, size_t* rbytes )
{
    const char* p = readfrom;
    const char* pbeg = NULL;
    const char* pend = NULL;

    if( !readfrom || !membuf )
        return( ERR_RS_MACC );

    for( ; *p == ' ' || *p == '\t'; p++ );
    for( pbeg = p, pend = p + readlen; p < pend &&
        *p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r'; p++ );

    if( size_t( p - pbeg ) != 1 )
        return( ERR_RS_INVL );

    if( membuf )
        *membuf = *pbeg;

    if( rbytes )
        *rbytes = size_t( p - readfrom );

    return 0;
}
