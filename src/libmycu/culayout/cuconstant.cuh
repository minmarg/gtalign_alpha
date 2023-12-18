/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __cuconstant_h__
#define __cuconstant_h__

#include "libutil/macros.h"
#include "libutil/alpha.h"
#include "libgenp/gdats/PM2DVectorFields.h"

// -------------------------------------------------------------------------
// constant memory addresses for device code
//
//enum for indexing structure data sections of addresses in constant memory
//(corresponds to the total number of fields too):
enum {
    ndx_qrs_dc_pm2dvfields_ = 0,
    ndx_dbs_dc_pm2dvfields_ = ndx_qrs_dc_pm2dvfields_ + pmv2DTotFlds,
    ndx_qrs_dc_pm2dvndxfds_ = ndx_dbs_dc_pm2dvfields_ + pmv2DTotFlds,
    ndx_dbs_dc_pm2dvndxfds_ = ndx_qrs_dc_pm2dvndxfds_ + pmv2DTotIndexFlds,
    sztot_pm2dvfields = ndx_dbs_dc_pm2dvndxfds_ + pmv2DTotIndexFlds
};

//device addresses of vectors representing query AND database structure data:
extern __constant__ char* dc_pm2dvfields_[sztot_pm2dvfields];

//Gonnet scores
extern __constant__ float dc_Gonnet_scores_[NEA * NEA];

// -------------------------------------------------------------------------

#endif//__cuconstant_h__
