/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __mygetopt_h__
#define __mygetopt_h__

#include <string>
#include <vector>

enum TArgFlag {
    my_required_argument,
    my_optional_argument,
    my_no_argument,
    my_n_targflags
};

// predefined return values
enum TRetValues {
    my_gorv_term = -1,//terminated processing of command line
    my_gorv_value = '*',//value, not option
    my_gorv_noarg = ':',//argument missing for option
    my_gorv_notfound = '?',//option not found
    my_gorv_illformed = '!',//ill-formed option
};

struct myoption {
    const char* sLongname_;//option's name
    TArgFlag    eFlag_;//option's property
    int         nRet_;//return value
};

// _________________________________________________________________________
// CLASS MyGetopt
// for parsing command line
//
class MyGetopt {
public:
    MyGetopt( const myoption*, const char *argv[], int argc );
    ~MyGetopt();

    int GetNextOption( std::string* argument );

protected:
    void Init( const myoption*, const char *argv[], int argc );
    void Reset() { ncnt_ = 0; stopped_ = false; };

private:
    int ncnt_;//word counter in options; max value is argc
    bool stopped_;//flag of stopping processing command line
    const char** argv_;//command line
    int argc_;//number of words in command line
    std::vector<myoption> srtoptions_;//sorted options
};

// --- INLINES -------------------------------------------------------------
// comparison operators for two option names
//
inline
bool operator<(const myoption& left, const myoption& right )
{
    if( left.sLongname_ == NULL || right.sLongname_ == NULL )
        return false;
    return strcmp(left.sLongname_, right.sLongname_) < 0;
}

inline
bool operator==(const myoption& left, const myoption& right )
{
    if( left.sLongname_ == NULL || right.sLongname_ == NULL )
        return false;
    return strcmp(left.sLongname_, right.sLongname_) == 0;
}

// -------------------------------------------------------------------------

#endif//__mygetopt_h__
