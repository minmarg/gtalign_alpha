# Create a library with the listed source files.
add_library( mycu STATIC
	culayout/CuDeviceMemory.cu
#
	cubatch/TdCommutator.cpp
#
	cubatch/CuBatch.cu
#
##	cusa/cuspecsim.cu
####	cusa/cuspecsim2.cu
####	cusa/cuspecsim3.cu
##	cusa/cuspecsim32.cu
####	cusa/cuspecsim7.cu
#
	custage1/custage1.cu
	custage1/custage2.cu
	custage1/custage_ssrr.cu
##	custage1/custage_dp.cu
##	custage1/custage_frg.cu
##	custage1/custage_frg2.cu
	custage1/custage_frg3.cu
	custage1/custage_fin.cu
#
	custages/covariance.cu
	custages/covariance_plus.cu
	custages/covariance_refn.cu
	custages/covariance_dp_refn.cu
##	custages/covariance_dp_scan.cu
	custages/covariance_swift_scan.cu
	custages/transform.cu
	custages/scoring.cu
#
	custages2/covariance_complete.cu
	custages2/covariance_refn_complete.cu
	custages2/covariance_dp_refn_complete.cu
	custages2/covariance_fin_dp_refn_complete.cu
	custages2/covariance_production_dp_refn_complete.cu
	custages2/production_2tmscore.cu
#
#	custgfrg/local_similarity0.cu
	custgfrg/local_similarity02.cu
##	custgfrg/linear_scoring.cu
	custgfrg/linear_scoring2.cu
#
	custgfrg2/linear_scoring2_complete.cu
#
	cudp/dpw_score.cu
	cudp/dpw_btck.cu
#	cudp/dpss.cu
	cudp/dpsslocal.cu
	cudp/dpssw_btck.cu
	cudp/dpssw_tfm_btck.cu
##	cudp/dpw_btck_specscore.cu
##	cudp/dpssw_btck_specscore.cu
##	cudp/dpagc_ssw_btck_specscore.cu
##	cudp/dpagc_tfm_ssw_btck_specscore.cu
	cudp/btck2match.cu
	cudp/constrained_btck2match.cu
	cudp/production_match2aln.cu
#
	cudp2/dpw_btck_complete.cu
#
	cuss/cusecstr.cu
	cuss/ssk.cu
#
	cufilter/coverage.cu
	cufilter/similarity.cu
	cufilter/reformatter.cu
#
##	cuspectrum/spatial_domain.cu
##	cuspectrum/freq_domain.cu
##	cuspectrum/psd_score.cu
#
####	cuspectrum2/spatial_domain2.cu
####	cuspectrum2/spatial_score2.cu
#
####	cuspectrum3/spatial_domain3.cu
####	cuspectrum3/spatial_sort3.cu
####	cuspectrum3/spatial_score3.cu
#
##	cuspectrum7/spatial_domain7.cu
##	cuspectrum7/freq_domain7.cu
##	cuspectrum7/psd_score7.cu
#
##	cuspectrum32/spatial_domain32.cu
##	cuspectrum32/spatial_sort32.cu
##	cuspectrum32/spatial_score32.cu
#
	cuproc/Devices.cu
	cuproc/cuprocconf.cpp
	cuproc/JobDispatcher.cpp
)

# Make sure the compiler can find include files for the library
# when other libraries or executables link to it
target_include_directories( mycu PUBLIC 
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set_target_properties( mycu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link to the required libraries 
#target_link_libraries( mycu PUBLIC pro )

