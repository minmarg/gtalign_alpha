/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include <stdio.h>
#include <math.h>

#include <string>
#include <functional>
#include <unordered_map>

// #include "mybase.h"
#include "cnsts.h"
#include "alpha.h"

const char* gSSAlphabet = "CEH";
const char* gSSlcAlphabet = " eh";

_GONNET_SCORES_  GONNET_SCORES;

// -------------------------------------------------------------------------
//
struct myresnamehash
{
    //simple hash for 3-letter residue names
    //NOTE: name3 is expected to contain at least 3 bytes
    std::size_t operator()(const char* name3) const noexcept
    {
        constexpr std::size_t seed = 131;
        return ((std::size_t)name3[0] * seed + name3[1]) * seed + name3[2];
    }
};

struct myresnameequal_to
{
    //simple equal_to functional for 3-letter residue names
    //NOTE: name31 and name32 are expected to contain at least 3 bytes
    bool operator()(const char* name31, const char* name32) const
    {
        return name31[0]==name32[0] && name31[1]==name32[1] && 
               name31[2]==name32[2];
    }
};

// -------------------------------------------------------------------------
// ResidueMap: Structure for implementing access to mapping between residue 
// names and codes
//
struct ResidueMap {
    ResidueMap();
    std::unordered_map<const char*, char, myresnamehash, myresnameequal_to> 
        name2codemap_;
    std::unordered_map<char, const char*> code2namemap_;
};

// -------------------------------------------------------------------------
// CONSTRUCTION

struct ResidueMap _gRM;

// -------------------------------------------------------------------------
// function definitions:
//
char ResName2Code(const char* name)
{
    auto itcode = _gRM.name2codemap_.find(name);
    if(itcode == _gRM.name2codemap_.end())
        return 
        name[0]==' '
        ?   (name[1]==' '
            ?   (name[2]==' '? 'X': name[2])
            :   (name[2]==' '? name[1]: 'X')
            )
        :   (name[1]==' '? name[0]: 'X')
        ;
//         (name[0]!=' ' && name[1]==' ')
//         ?   name[0]
//         :   ((name[0]==' ' && name[1]!=' ')
//             ?   name[1]
//             :   ((name[1]==' ' && name[2]!=' ')
//                 ?   name[2]
//                 :   'X'
//                 )
//             )
//         ;
        //return 'X';
    return itcode->second;
}

const char* ResCode2Name(char code)
{
    auto itname = _gRM.code2namemap_.find(code);
    if(itname == _gRM.code2namemap_.end())
        return "UNK";
    return itname->second;
}

// -------------------------------------------------------------------------
// test hash functions:
void testmyresnamehash()
{
    fprintf(stdout, "Name-to-code stats:\n");
    for(auto mypair: _gRM.name2codemap_)
        fprintf(stdout, "  %s -> %c : %zu\n", mypair.first, mypair.second, 
            _gRM.name2codemap_.bucket_size(_gRM.name2codemap_.bucket(mypair.first)));

    fprintf(stdout, "\nCode-to-name stats:\n");
    for(auto mypair: _gRM.code2namemap_)
        fprintf(stdout, "  %c -> %s : %zu\n", mypair.first, mypair.second, 
            _gRM.code2namemap_.bucket_size(_gRM.code2namemap_.bucket(mypair.first)));
}

// -------------------------------------------------------------------------
// Constructor:
//
ResidueMap::ResidueMap()
{
    //names to codes:
    name2codemap_["ALA"] = 'A'; name2codemap_["DAL"] = 'A';
    name2codemap_["ASX"] = 'B';
    name2codemap_["CYS"] = 'C'; name2codemap_["DCY"] = 'C';
    name2codemap_["ASP"] = 'D'; name2codemap_["DAS"] = 'D';
    name2codemap_["GLU"] = 'E'; name2codemap_["DGL"] = 'E';
    name2codemap_["PHE"] = 'F'; name2codemap_["DPN"] = 'F';
    name2codemap_["GLY"] = 'G';
    name2codemap_["HIS"] = 'H'; name2codemap_["DHI"] = 'H';
    name2codemap_["ILE"] = 'I'; name2codemap_["DIL"] = 'I';
    name2codemap_["LYS"] = 'K'; name2codemap_["DLY"] = 'K';
    name2codemap_["LEU"] = 'L'; name2codemap_["DLE"] = 'L';
    name2codemap_["MET"] = 'M'; name2codemap_["MED"] = 'M'; name2codemap_["MSE"] = 'M';
    name2codemap_["ASN"] = 'N'; name2codemap_["DSG"] = 'N';
    name2codemap_["PYL"] = 'O';
    name2codemap_["PRO"] = 'P'; name2codemap_["DPR"] = 'P';
    name2codemap_["GLN"] = 'Q'; name2codemap_["DGN"] = 'Q';
    name2codemap_["ARG"] = 'R'; name2codemap_["DAR"] = 'R';
    name2codemap_["SER"] = 'S'; name2codemap_["DSN"] = 'S';
    name2codemap_["THR"] = 'T'; name2codemap_["DTH"] = 'T';
    name2codemap_["SEC"] = 'U';
    name2codemap_["VAL"] = 'V'; name2codemap_["DVA"] = 'V';
    name2codemap_["TRP"] = 'W'; name2codemap_["DTR"] = 'W';
    name2codemap_["TYR"] = 'Y'; name2codemap_["DTY"] = 'Y';
    name2codemap_["GLX"] = 'Z';
    //
    //codes to names:
    code2namemap_['A'] = "ALA"; code2namemap_['B'] = "ASX";
    code2namemap_['C'] = "CYS"; code2namemap_['D'] = "ASP";
    code2namemap_['E'] = "GLU"; code2namemap_['F'] = "PHE";
    code2namemap_['G'] = "GLY"; code2namemap_['H'] = "HIS";
    code2namemap_['I'] = "ILE"; code2namemap_['K'] = "LYS";
    code2namemap_['L'] = "LEU"; code2namemap_['M'] = "MET";
    code2namemap_['N'] = "ASN"; code2namemap_['O'] = "PYL";
    code2namemap_['P'] = "PRO"; code2namemap_['Q'] = "GLN";
    code2namemap_['R'] = "ARG"; code2namemap_['S'] = "SER";
    code2namemap_['T'] = "THR"; code2namemap_['U'] = "SEC";
    code2namemap_['V'] = "VAL"; code2namemap_['W'] = "TRP";
    code2namemap_['Y'] = "TYR"; code2namemap_['Z'] = "GLX";
}

// Gonnet frequency ratios (expanded over the English alphabet);
// characteristics of the bit scaled matrix
//             Scale: 1/4 (ln2/4); 1/3 (ln2/3)
//           Entropy: 0.2465141
//     ExpectedScore: -0.2065916
//      HighestScore: 19; 14 (after rescaling)
//       LowestScore: -7; -5 (after rescaling)
const float GONNET_FREQRATIOS[NEA * NEA] = {
//  A          B          C          D          E          F          G          H          I          J          K          L          M          N          O          P          Q          R          S          T          U          V          W          X          Y          Z           
1.737801f, 1.000000f, 1.122018f, 0.933254f, 1.000000f, 0.588844f, 1.122018f, 0.831764f, 0.831764f, 1.000000f, 0.912011f, 0.758578f, 0.851138f, 0.933254f, 1.000000f, 1.071519f, 0.954993f, 0.870964f, 1.288250f, 1.148154f, 1.000000f, 1.023293f, 0.436516f, 1.000000f, 0.602560f, 1.000000f, // A 
1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, // B 
1.122018f, 1.000000f,14.125375f, 0.478630f, 0.501187f, 0.831764f, 0.630957f, 0.741310f, 0.776247f, 1.000000f, 0.524807f, 0.707946f, 0.812831f, 0.660693f, 1.000000f, 0.489779f, 0.575440f, 0.602560f, 1.023293f, 0.891251f, 1.000000f, 1.000000f, 0.794328f, 1.000000f, 0.891251f, 1.000000f, // C 
0.933254f, 1.000000f, 0.478630f, 2.951209f, 1.862087f, 0.354813f, 1.023293f, 1.096478f, 0.416869f, 1.000000f, 1.122018f, 0.398107f, 0.501187f, 1.659587f, 1.000000f, 0.851138f, 1.230269f, 0.933254f, 1.122018f, 1.000000f, 1.000000f, 0.512861f, 0.301995f, 1.000000f, 0.524807f, 1.000000f, // D 
1.000000f, 1.000000f, 0.501187f, 1.862087f, 2.290868f, 0.407380f, 0.831764f, 1.096478f, 0.537032f, 1.000000f, 1.318257f, 0.524807f, 0.630957f, 1.230269f, 1.000000f, 0.891251f, 1.479108f, 1.096478f, 1.047129f, 0.977237f, 1.000000f, 0.645654f, 0.371535f, 1.000000f, 0.537032f, 1.000000f, // E 
0.588844f, 1.000000f, 0.831764f, 0.354813f, 0.407380f, 5.011872f, 0.301995f, 0.977237f, 1.258925f, 1.000000f, 0.467735f, 1.584893f, 1.445440f, 0.489779f, 1.000000f, 0.416869f, 0.549541f, 0.478630f, 0.524807f, 0.602560f, 1.000000f, 1.023293f, 2.290868f, 1.000000f, 3.235937f, 1.000000f, // F 
1.122018f, 1.000000f, 0.630957f, 1.023293f, 0.831764f, 0.301995f, 4.570882f, 0.724436f, 0.354813f, 1.000000f, 0.776247f, 0.363078f, 0.446684f, 1.096478f, 1.000000f, 0.691831f, 0.794328f, 0.794328f, 1.096478f, 0.776247f, 1.000000f, 0.467735f, 0.398107f, 1.000000f, 0.398107f, 1.000000f, // G 
0.831764f, 1.000000f, 0.741310f, 1.096478f, 1.096478f, 0.977237f, 0.724436f, 3.981072f, 0.602560f, 1.000000f, 1.148154f, 0.645654f, 0.741310f, 1.318257f, 1.000000f, 0.776247f, 1.318257f, 1.148154f, 0.954993f, 0.933254f, 1.000000f, 0.630957f, 0.831764f, 1.000000f, 1.659587f, 1.000000f, // H 
0.831764f, 1.000000f, 0.776247f, 0.416869f, 0.537032f, 1.258925f, 0.354813f, 0.602560f, 2.511886f, 1.000000f, 0.616595f, 1.905461f, 1.778279f, 0.524807f, 1.000000f, 0.549541f, 0.645654f, 0.575440f, 0.660693f, 0.870964f, 1.000000f, 2.041738f, 0.660693f, 1.000000f, 0.851138f, 1.000000f, // I 
1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, // J 
0.912011f, 1.000000f, 0.524807f, 1.122018f, 1.318257f, 0.467735f, 0.776247f, 1.148154f, 0.616595f, 1.000000f, 2.089296f, 0.616595f, 0.724436f, 1.202264f, 1.000000f, 0.870964f, 1.412538f, 1.862087f, 1.023293f, 1.023293f, 1.000000f, 0.676083f, 0.446684f, 1.000000f, 0.616595f, 1.000000f, // K 
0.758578f, 1.000000f, 0.707946f, 0.398107f, 0.524807f, 1.584893f, 0.363078f, 0.645654f, 1.905461f, 1.000000f, 0.616595f, 2.511886f, 1.905461f, 0.501187f, 1.000000f, 0.588844f, 0.691831f, 0.602560f, 0.616595f, 0.741310f, 1.000000f, 1.513561f, 0.851138f, 1.000000f, 1.000000f, 1.000000f, // L 
0.851138f, 1.000000f, 0.812831f, 0.501187f, 0.630957f, 1.445440f, 0.446684f, 0.741310f, 1.778279f, 1.000000f, 0.724436f, 1.905461f, 2.691535f, 0.602560f, 1.000000f, 0.575440f, 0.794328f, 0.676083f, 0.724436f, 0.870964f, 1.000000f, 1.445440f, 0.794328f, 1.000000f, 0.954993f, 1.000000f, // M 
0.933254f, 1.000000f, 0.660693f, 1.659587f, 1.230269f, 0.489779f, 1.096478f, 1.318257f, 0.524807f, 1.000000f, 1.202264f, 0.501187f, 0.602560f, 2.398833f, 1.000000f, 0.812831f, 1.174898f, 1.071519f, 1.230269f, 1.122018f, 1.000000f, 0.602560f, 0.436516f, 1.000000f, 0.724436f, 1.000000f, // N 
1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, // O 
1.071519f, 1.000000f, 0.489779f, 0.851138f, 0.891251f, 0.416869f, 0.691831f, 0.776247f, 0.549541f, 1.000000f, 0.870964f, 0.588844f, 0.575440f, 0.812831f, 1.000000f, 5.754399f, 0.954993f, 0.812831f, 1.096478f, 1.023293f, 1.000000f, 0.660693f, 0.316228f, 1.000000f, 0.489779f, 1.000000f, // P 
0.954993f, 1.000000f, 0.575440f, 1.230269f, 1.479108f, 0.549541f, 0.794328f, 1.318257f, 0.645654f, 1.000000f, 1.412538f, 0.691831f, 0.794328f, 1.174898f, 1.000000f, 0.954993f, 1.862087f, 1.412538f, 1.047129f, 1.000000f, 1.000000f, 0.707946f, 0.537032f, 1.000000f, 0.676083f, 1.000000f, // Q 
0.870964f, 1.000000f, 0.602560f, 0.933254f, 1.096478f, 0.478630f, 0.794328f, 1.148154f, 0.575440f, 1.000000f, 1.862087f, 0.602560f, 0.676083f, 1.071519f, 1.000000f, 0.812831f, 1.412538f, 2.951209f, 0.954993f, 0.954993f, 1.000000f, 0.630957f, 0.691831f, 1.000000f, 0.660693f, 1.000000f, // R 
1.288250f, 1.000000f, 1.023293f, 1.122018f, 1.047129f, 0.524807f, 1.096478f, 0.954993f, 0.660693f, 1.000000f, 1.023293f, 0.616595f, 0.724436f, 1.230269f, 1.000000f, 1.096478f, 1.047129f, 0.954993f, 1.659587f, 1.412538f, 1.000000f, 0.794328f, 0.467735f, 1.000000f, 0.645654f, 1.000000f, // S 
1.148154f, 1.000000f, 0.891251f, 1.000000f, 0.977237f, 0.602560f, 0.776247f, 0.933254f, 0.870964f, 1.000000f, 1.023293f, 0.741310f, 0.870964f, 1.122018f, 1.000000f, 1.023293f, 1.000000f, 0.954993f, 1.412538f, 1.778279f, 1.000000f, 1.000000f, 0.446684f, 1.000000f, 0.645654f, 1.000000f, // T 
1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, // U 
1.023293f, 1.000000f, 1.000000f, 0.512861f, 0.645654f, 1.023293f, 0.467735f, 0.630957f, 2.041738f, 1.000000f, 0.676083f, 1.513561f, 1.445440f, 0.602560f, 1.000000f, 0.660693f, 0.707946f, 0.630957f, 0.794328f, 1.000000f, 1.000000f, 2.187762f, 0.549541f, 1.000000f, 0.776247f, 1.000000f, // V 
0.436516f, 1.000000f, 0.794328f, 0.301995f, 0.371535f, 2.290868f, 0.398107f, 0.831764f, 0.660693f, 1.000000f, 0.446684f, 0.851138f, 0.794328f, 0.436516f, 1.000000f, 0.316228f, 0.537032f, 0.691831f, 0.467735f, 0.446684f, 1.000000f, 0.549541f,26.302680f, 1.000000f, 2.570396f, 1.000000f, // W 
1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, // X 
0.602560f, 1.000000f, 0.891251f, 0.524807f, 0.537032f, 3.235937f, 0.398107f, 1.659587f, 0.851138f, 1.000000f, 0.616595f, 1.000000f, 0.954993f, 0.724436f, 1.000000f, 0.489779f, 0.676083f, 0.660693f, 0.645654f, 0.645654f, 1.000000f, 0.776247f, 2.570396f, 1.000000f, 6.025596f, 1.000000f, // Y 
1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f  // Z 
};

// -------------------------------------------------------------------------
// _GONNET_SCORES_: calculate scaled Gonnet scores 
//
_GONNET_SCORES_::_GONNET_SCORES_()
{
    //scaling constant for Gonnet frequencies:
    // constexpr float scGonnet = 3.0f;
    for(int aa = 0; aa < NEA * NEA; aa++ )
        data_[aa] = logf(GONNET_FREQRATIOS[aa]) * LN2f_recip;// * scGonnet;
}
