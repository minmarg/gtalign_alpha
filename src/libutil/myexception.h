/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __myexception_h__
#define __myexception_h__

#include <string>
#include "debug.h"

#define MYRUNTIME_ERROR( MSG )  myruntime_error( MSG, __EXCPOINT__ )
#define MYRUNTIME_ERROR2( MSG, CLS )  myruntime_error( MSG, __EXCPOINT__, CLS )

#define TRY try {
#define CATCH_ERROR_RETURN(STATEMENTS) \
        STATEMENTS; \
    } catch( myruntime_error const& ex ) { \
        STATEMENTS; \
        error( ex.pretty_format().c_str()); \
        return EXIT_FAILURE; \
    } catch( myexception const& ex ) { \
        STATEMENTS; \
        error( ex.what()); \
        return EXIT_FAILURE; \
    } catch( ... ) { \
        STATEMENTS; \
        error("Unknown exception caught."); \
        return EXIT_FAILURE; \
    }

enum {
    NOCLASS,
    SCALING,
    CRITICAL
};

// _________________________________________________________________________
// CLASS myexception
// for exception handling
//
class myexception
{
public:
    myexception() throw() {}
    myexception( const myexception& ex ) throw() { operator=(ex);};
    virtual ~myexception() throw() {};
    virtual myexception& operator=( const myexception& ) throw() { return *this;};
    virtual const char* what() const throw();
    virtual int         eclass() const throw();
};

// _________________________________________________________________________
// CLASS myruntime_error
// runtime error exception
//
class myruntime_error: public myexception
{
public:
    myruntime_error() throw();
    myruntime_error( const myexception& ex ) throw(){ operator=(ex);};
    myruntime_error( const myruntime_error& mre ) throw(): myexception(mre) { operator=(mre);};
    explicit myruntime_error( const std::string& arg, int ecl = NOCLASS ) throw();
    myruntime_error( const std::string& arg, 
            const char* fl, unsigned int ln, const char* func, int ecl = NOCLASS ) throw();
    myruntime_error( const char* fl, unsigned int ln, const char* func, int ecl = NOCLASS ) throw();
    virtual ~myruntime_error() throw();

    virtual myexception& operator=( const myexception& ) throw();
    virtual myruntime_error& operator=( const myruntime_error& ) throw();

    virtual bool isset() const throw();

    virtual const char*     what() const throw();       //cause of the error
    virtual int             eclass() const throw();     //error class
    virtual const char*     file() const throw();       //filename
    virtual unsigned int    line() const throw();       //line number
    virtual const char*     function() const throw();   //function name

    virtual std::string     pretty_format( std::string = std::string()) const throw();

private:
    std::string     _errdsc;    //error description
    int             _class;     //error class
    const char*     _file;      //filename
    unsigned int    _line;      //line number
    const char*     _function;  //function name
};

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// CLASS myexception implementation
//
// what: error description
//
inline const char* myexception::what() const throw()
{
    return "{myexception}";
}

// eclass: error class
//
inline int myexception::eclass() const throw()
{
    return NOCLASS;
}

// -------------------------------------------------------------------------
// CLASS myruntime_error
//
// Constructors
//
inline myruntime_error::myruntime_error() throw()
:   myexception(),
    _errdsc(),
    _class( NOCLASS ),
    _file( NULL ),
    _line(0),
    _function( NULL )
{
}

inline myruntime_error::myruntime_error( const std::string& arg, int ecldesc ) throw()
:   myexception(),
    _errdsc( arg ),
    _class( ecldesc ),
    _file( NULL ),
    _line(0),
    _function( NULL )
{
}

inline myruntime_error::myruntime_error( const std::string& arg, 
    const char* fl, unsigned int ln, const char* func, int ecldesc ) throw()
:   myexception(),
    _errdsc( arg ),
    _class( ecldesc ),
    _file( fl ),
    _line( ln ),
    _function( func )
{
}

inline myruntime_error::myruntime_error( 
    const char* fl, unsigned int ln, const char* func, int ecldesc ) throw()
:   myexception(),
    _errdsc(),
    _class( ecldesc ),
    _file( fl ),
    _line( ln ),
    _function( func )
{
}

// Destructor
//
inline myruntime_error::~myruntime_error() throw()
{
}

// -------------------------------------------------------------------------
// operator=: assignment operators
//
inline myexception& myruntime_error::operator=( const myexception& me ) throw()
{
    const myruntime_error* pme = dynamic_cast<const myruntime_error*>(&me);
    if( pme ) {
        operator=(*pme);
    }
    else {
        _errdsc = myexception::what();
        _class = myexception::eclass();
        _file = NULL;
        _line = 0;
        _function = NULL;
    }
    return *this;
}

inline myruntime_error& myruntime_error::operator=( const myruntime_error& mre ) throw()
{
    _errdsc = mre._errdsc;
    _class = mre.eclass();
    _file = mre.file();
    _line = mre.line();
    _function = mre.function();
    return *this;
}

// -------------------------------------------------------------------------
// isset: check whether the error is set
//
inline bool myruntime_error::isset() const throw()
{
    return !_errdsc.empty() || _class != NOCLASS || 
          _file || _line || _function;
}

// -------------------------------------------------------------------------
// what: error description
//
inline const char* myruntime_error::what() const throw()
{
    return _errdsc.c_str();
}
// -------------------------------------------------------------------------
// eclass: error class 
//
inline int myruntime_error::eclass() const throw()
{
    return _class;
}
// -------------------------------------------------------------------------
// file: filename
//
inline const char* myruntime_error::file() const throw()
{
    return _file;
}
// -------------------------------------------------------------------------
// line: line number
//
inline unsigned int myruntime_error::line() const throw()
{
    return _line;
}
// -------------------------------------------------------------------------
// function: function name
//
inline const char* myruntime_error::function() const throw()
{
    return _function;
}


#endif//__myexception_h__
