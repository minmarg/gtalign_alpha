/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __myfiler_h__
#define __myfiler_h__

#include <stdio.h>
#include <string>

#include <sys/types.h>
#include <sys/stat.h>
#include "platform.h"

#ifdef OS_MS_WINDOWS
#	include <Windows.h>
#endif

struct TCharStream {
    TCharStream()
    :
#ifdef OS_MS_WINDOWS
        hMapFile_(NULL),
#endif
        data_(NULL), datlen_(0), curpos_(0),
        pagenr_(0), pagesize_(0), pageoff_(0)
    {}
    TCharStream( 
#ifdef OS_MS_WINDOWS
        HANDLE hmapfile,
#endif
        char* data, size_t datlen, size_t curpos,
        size_t pagenr, size_t pagesize, size_t pageoff)
    :
#ifdef OS_MS_WINDOWS
        hMapFile_(hmapfile),
#endif
        data_(data), datlen_(datlen), curpos_(curpos),
        pagenr_(pagenr), pagesize_(pagesize), pageoff_(pageoff)
    {}
    TCharStream& operator=(const TCharStream& chstr) {
        datlen_ = chstr.datlen_;
        curpos_ = chstr.curpos_;
        pagenr_ = chstr.pagenr_;
        pageoff_ = chstr.pageoff_;
        return *this;
    }
    void ResetCounters() {
        datlen_ = 0;
        curpos_ = 0;
        pagenr_ = 0;
        pageoff_ = 0;
    }
    void incpos(size_t by) {curpos_ += by;}
    void incposnl(size_t by) {
        curpos_ += by;
        const char* pbeg = data_ + curpos_;
        const char* pend = data_ + datlen_;
        const char* p = pbeg;
        for(; p < pend && *p != '\n' && *p != '\r'; p++);
        for(; p < pend && (*p == '\n' || *p == '\r'); p++);
        curpos_ += (size_t)(p-pbeg);
    }
    bool eof() const {return datlen_ <= curpos_;}
    //
#ifdef OS_MS_WINDOWS
    HANDLE hMapFile_;//handle of the file mapping object
#endif
    char* data_;//data
    size_t datlen_;//data size
    size_t curpos_;//next position in the current data to read
    size_t pagenr_;//number of the page where the data begins
    size_t pagesize_;//page size in bytes
    size_t pageoff_;//offset of page pagenr_
};

// file/directory routines
bool file_exists( const char*,
#ifdef OS_MS_WINDOWS
    unsigned short mode = _S_IFREG
#else
    mode_t mode = S_IFREG
#endif
);
inline
bool directory_exists( const char* pathname ) {
    return file_exists( pathname, 
#ifdef OS_MS_WINDOWS
    _S_IFDIR
#else
    S_IFDIR
#endif
    );
}
int file_size(const char*, size_t*);
int mymkdir(const char* pathname);


//functor for reading from a stream into the string buffer
struct CopyToBuffer {
    void operator()(char* srcptr, size_t srclen, std::string* buf, size_t*) {
        *buf = std::string(srcptr, srclen);
    }
};
//functor for reading from a stream and setting the address where the data begins
struct GetPtr {
    void operator()(char* srcptr, size_t srclen, char** outptr, size_t* outlen) {
        *outptr = srcptr;
        *outlen = srclen;
    }
};


int skip_comments( FILE* fp, char* buffer, size_t bufsize, size_t* readlen, char cc = '#');
int skip_comments( FILE* fp, std::string& buffer, char cc = '#');
int skip_comments( TCharStream* source, std::string& buffer, char cc  = '#');
int skip_comments( TCharStream* source, char*& ptr, size_t, size_t* len, char cc  = '#');
int read_double( const char* readfrom, size_t readlen, double* membuf, size_t* rbytes );
int read_float( const char* readfrom, size_t readlen, float* membuf, size_t* rbytes );
int read_integer( const char* readfrom, size_t readlen, int* membuf, size_t* rbytes );
int read_llinteger( const char* readfrom, size_t readlen, long long int* membuf, size_t* rbytes );
int read_symbol( const char* readfrom, size_t readlen, char* membuf, size_t* rbytes );

inline int feof(TCharStream* chs ) {
    if( chs == NULL || chs->data_ == NULL || 
        chs->datlen_ < 1 || chs->datlen_ <= chs->curpos_ )
        return 1;
    return 0;
}

#define ERR_SC_MACC ( 111 )
#define ERR_SC_READ ( 113 )
#define ERR_RD_MACC ( 115 )
#define ERR_RD_NOVL ( 117 )
#define ERR_RD_INVL ( 119 )
#define ERR_RI_MACC ( 131 )
#define ERR_RI_NOVL ( 133 )
#define ERR_RI_INVL ( 135 )
#define ERR_RL_MACC ( 137 )
#define ERR_RL_NOVL ( 139 )
#define ERR_RL_INVL ( 151 )
#define ERR_RS_MACC ( 153 )
#define ERR_RS_INVL ( 155 )

//error messages
inline const char* TranslateReadError( int code )
{
    switch( code ) {
        case 0: return "OK";
        case ERR_SC_MACC: return "skip_comments: Memory access error";
        case ERR_SC_READ: return "skip_comments: Reading error";
        case ERR_RD_MACC: return "read_double: Memory access error";
        case ERR_RD_NOVL: return "read_double: No double value read";
        case ERR_RD_INVL: return "read_double: Invalid double value";
        case ERR_RI_MACC: return "read_integer: Memory access error";
        case ERR_RI_NOVL: return "read_integer: No integer read";
        case ERR_RI_INVL: return "read_integer: Invalid integer value";
        case ERR_RL_MACC: return "read_llinteger: Memory access error";
        case ERR_RL_NOVL: return "read_llinteger: No integer read";
        case ERR_RL_INVL: return "read_llinteger: Invalid integer value";
        case ERR_RS_MACC: return "read_symbol: Memory access error";
        case ERR_RS_INVL: return "read_symbol: Not a single character";
    }
    return "Unknown";
}

#endif//__myfiler_h__
