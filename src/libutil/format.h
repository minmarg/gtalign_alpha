/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __format_h__
#define __format_h__

#include <string.h>
#include "platform.h"

// -------------------------------------------------------------------------
// PutNL: put a new line delimiter in the output character string;
// returns the number of bytes written;
inline
int PutNL( char*& outptr ) {
    static const char* nl = NL;
    static const int sznl = (int)strlen(NL);
    for(int i=0; i<sznl; i++) *outptr++ = nl[i];
    return sznl;
}

// -------------------------------------------------------------------------
// FormatDescription: format description line;
// NOTE: space is assumed to be pre-allocated;
// outptr, pointer to the output buffer;
// desc, description to copy;
// maxoutlen, maximum length of output description;
// indent, new line indentation;
// width, width to wrap the description;
// outpos, current position in the output buffer;
// linepos, current position in the current output line;
// addsep, add a separator in the output list of alignments;
inline
void FormatDescription( 
    char*& outptr,
    const char* desc,
    const int maxoutlen,
    const int indent,
    const int width,
    int& outpos,
    int& linepos,
    char addsep )
{
    int loc = 0;

    if(!desc) return;

    for(; *desc && (*desc==' '||*desc=='\t'||*desc=='\n'||*desc=='\r'); desc++);

    if(addsep) {
        *outptr++ = addsep;
        outpos++;
        linepos++;
    }

//     char* pbo = outptr;//beginning

    while(*desc && outpos < maxoutlen) {
        if(*desc==' '||*desc=='\t') {

            for(; *desc && (*desc==' '||*desc=='\t'); desc++);

            if(width < maxoutlen) {
                loc = linepos + 1;
                for(const char* p = desc; 
                    *p && *p!=' ' && *p!='\t' && loc < width; p++, loc++);
                if( width <= loc ) {
                    outpos += PutNL(outptr);
                    for(int i=0; i<indent; i++, outpos++ ) *outptr++ = ' ';
                    linepos = indent;
                }
                else {
                    *outptr++ = ' ';
                    outpos++;
                    linepos++;
                }
            }
            else {
                *outptr++ = ' ';
                outpos++;
                linepos++;
            }

            continue;
        }

        *outptr++ = *desc++;
        outpos++;
        linepos++;
    }

    if(maxoutlen <= outpos && 3 < maxoutlen) {
        for(int i=1; i<=3; i++) *(outptr-i) = '.';
    }
}

// -------------------------------------------------------------------------
// FormatDescriptionJSON: format description line in JSON format;
// NOTE: space is assumed to be pre-allocated;
// outptr, pointer to the output buffer;
// desc, description to copy;
// maxoutlen, maximum length of output description;
// outpos, current position in the output buffer;
inline
void FormatDescriptionJSON( 
    char*& outptr,
    const char* desc,
    const int maxoutlen,
    int& outpos)
{
    if(!desc) return;

    for(; *desc && (*desc==' '||*desc=='\t'||*desc=='\n'||*desc=='\r'); desc++);

    while(*desc && outpos < maxoutlen) {
        if(*desc==' '||*desc=='\t') {
            for(; *desc && (*desc==' '||*desc=='\t'); desc++);
            *outptr++ = ' ';
            outpos++;
            continue;
        }
        else if(*desc=='"'||*desc=='\\'||*desc=='/') {
            *outptr++ = '\\';
            outpos++;
            if(maxoutlen <= outpos)///
                break;
        }

        *outptr++ = *desc++;
        outpos++;
    }

    if(maxoutlen <= outpos && 3 < maxoutlen) {
        int i;
        for(i=1; i<=3; i++) *(outptr-i) = '.';
        for(; i <= outpos && *(outptr-i) == '\\'; i++) *(outptr-i) = '.';
    }
}

#endif//__format_h__
