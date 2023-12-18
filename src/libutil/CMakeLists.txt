# Create a library with the listed source files.
# Any number of sources could be listed here.
add_library( util STATIC 
	mydirent.cpp
	msg.cpp
	myexception.cpp
##	root.cpp
	alpha.cpp
##	mysort.cpp
	myfiler.cpp
	mygetopt.cpp
	CLOptions.cpp
)

# Make sure the compiler can find include files for the library
# when other libraries or executables link to it
target_include_directories( util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
