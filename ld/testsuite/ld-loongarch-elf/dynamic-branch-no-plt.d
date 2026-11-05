#source: dynamic-branch-no-plt.s
#ld: -e 0 -Ttext=0x100 -shared
#error: .*\(\.nonalloc\+0x0\): error: R_LARCH_B26 against `foo':\n
#error: cannot resolve a dynamic branch without a PLT entry
