if (TARGET polyscope)
  return()
endif()

include(CPM)

#set(CMAKE_CXX_FLAGS_DEBUG_OLD "${CMAKE_CXX_FLAGS_DEBUG}")
#set(CMAKE_CXX_FLAGS_DEBUG "-w")

CPMAddPackage(
  NAME polyscope
  #VERSION 2.5.0
  GITHUB_REPOSITORY "nmwsharp/polyscope"
  GIT_TAG 0c3dd68b9851417e6b2b976d347adc3250026122
)

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG_OLD}")
