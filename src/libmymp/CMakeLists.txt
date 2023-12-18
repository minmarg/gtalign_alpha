# Create a library with the listed source files.
# Any number of sources could be listed here.
add_library( mymp STATIC 
	mplayout/MpGlobalMemory.cpp
#
	mpbatch/MpBatch.cpp
#
	mpss/MpSecStr.cpp
#
	mpfilter/MpReform.cpp
	mpstage1/MpStage1.cpp
	mpstage1/MpStageFrg3.cpp
	mpstage1/MpStageFin.cpp
#
	mpdp/MpDPHub.cpp
#
	mpproc/TaskScheduler.cpp
)

# Make sure the compiler can find include files for the library
# when other libraries or executables link to it
target_include_directories( mymp PUBLIC
      ${CMAKE_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}
)

