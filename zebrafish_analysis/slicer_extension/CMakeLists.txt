cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(ZebrafishAnalysis)

set(EXTENSION_HOMEPAGE "https://github.com/JonaRichter/Zebrafish_Analysis")
set(EXTENSION_CATEGORY "Quantification")
set(EXTENSION_CONTRIBUTORS "Jona Richter, Mark Daniel Arndt")
set(EXTENSION_DESCRIPTION "Zebrafish body length, curvature, and eye metrics from 2D microscopy images.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/JonaRichter/Zebrafish_Analysis/main/zebrafish_analysis/slicer_extension/ZebrafishAnalysis/Resources/Icons/ZebrafishAnalysis.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/JonaRichter/Zebrafish_Analysis/main/Documentation_images/slicer_screenshot.png")
set(EXTENSION_DEPENDS "NA")

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

add_subdirectory(ZebrafishAnalysis)

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
