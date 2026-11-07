#!/bin/bash
#************************************************************************
#                                                                      **
#                 Polarizable Gaussian Multipole (pGM)                 **
#                                                                      **
#                A performance-enhanced version of pGM                 **
#                                                                      **
#  Copyright (C) 2026 The Regents of the University of California      **
#                                                                      **
#  Author: Yong Duan <duan@ucdavis.edu>                                **
#  Department of Biomedical Engineering and UC Davis Genome Center     **
#  University of California, Davis, California                         **
#                                                                      **
#  Please acknowledge your use of pGM by citing:                       **
#                                                                      **
#  Huang, Z., Y. Wu, Y. Duan, R. Luo                                   **
#  J. Chem. Theory Comput. 2025, 21(2): 847-858.                       **
#  DOI: 10.1021/acs.jctc.4c01368                                       **
#                                                                      **
#  Huang, Z., S.J. Zhao, P. Cieplak, Y. Duan, R. Luo, H.X. Wei         **
#  J. Chem. Theory Comput. 2023, 19(15): 5047-5057                     **
#  DOI: 10.1021/acs.jctc.3c00226                                       **
#                                                                      **
#  Wei, H.X., P. Cieplak, Y. Duan, R.Y. Luo                            **
#  J. Chem. Phys. 2022, 156(11): 114114                                **
#  DOI: 10.1063/5.0082548                                              **
#                                                                      **
#  Wei, H.X., R.X. Qi, J.M. Wang, P. Cieplak, Y. Duan, R. Luo          **
#  J. Chem. Phys. 2020, 153(11): 114116                                **
#  DOI: 10.1063/5.0082548                                              **
#                                                                      **
#  Major Developers:                                                   **
#                                                                      **
#     Yong Duan, developer for high-performance pGM                    **
#     Zhen Huang, developer for sander and high-performance pGM        **
#     Haixin Wei, original developer for sander pGM                    **
#     Ray Luo, original developer for sander pGM                       **
#                                                                      **
#************************************************************************
#
# This is a wrapper to use the native cpp, instead of the built-in cpp in fortran compilers.
# The native cpp can process concatenation of multiple tokens whereas the build-in cpp in fortran
# compilers are unable to, at least gfortran is unable to.
# Due to the increased complexity, pGM needs a lot more subroutines for pairs_calc(), to
# the point nearly 1000 lines are needed just to define the subroutine headers.
# With the native cpp, we can reduce that to 10's of lines and improve consistency between
# subroutine and end subroutine.
#

FC_COMPILER=$1

shift

inf=`echo $@ | awk '{for (i=1; i<=NF; i++) {if (index($i,".F90.o") ==0 && index($i,".F90")>0) print $i}}'`

inff=$(basename $inf)
if [[ $inff == "pme_direct.F90" ]]; then
   if [ -z "$CPP" ]; then
      if command -v cpp > /dev/null 2>&1; then
	 CPP=$(command -v cpp)
      elif [ -x /usr/bin/cpp ]; then
	 CPP=/usr/bin/cpp
      elif [ -x /lib/cpp ]; then
	 CPP=/lib/cpp
      else
	 echo "Error: C preprocessor (cpp) not found. Please set CPP manually." >&2
	 exit 1
      fi
   fi
   infd=$(dirname $inf)
   cppdirec=(`echo $@ | awk '{for (i=1; i<=NF; i++) {if (substr($i,1,2)=="-D" || substr($i,1,2)=="-I") {if(length($i)==2) {print $i,$(i+1);i++} else print $i}}}'`)
   Outf=`echo $@ | awk '{for (i=1; i<=NF; i++) {if (index($i,".F90.o") >0) print $i}}'`
   fout=${Outf/.F90.o/.f90}
   fline=(`echo $@ | awk -v I=$inf -v O=$fout '{gsub(I, O); print}'`)
   $CPP ${cppdirec[*]} $inf -o $fout
   $FC_COMPILER ${fline[*]}
else
   $FC_COMPILER $@
fi
