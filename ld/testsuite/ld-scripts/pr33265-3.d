#source: start.s
#ld: -r --whole-archive -lpr33265-3a
#error: .*group nested too deeply.*
#
# XFAIL the test case on MinGW/Windows hosts due to the MAX_PATH limitation.
#xfail: [ishost\ *-mingw*]
