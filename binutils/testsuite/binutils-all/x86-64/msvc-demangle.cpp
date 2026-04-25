// Test file for MSVC demangling with DWARF debug info
// This file is used to test that objdump can correctly demangle MSVC-mangled symbols
// when they appear in binaries compiled with clang using the MSVC target and DWARF debug info.
//
// Compile command (used by objdump.exp test):
// clang++ --target=x86_64-pc-windows-msvc -O0 -gdwarf-5 -c msvc-demangle.cpp -o msvc-demangle.obj
//
// The MSVC ABI produces mangled names like:
//   ?myMethod@MyClass@MyNamespace@@QEAAXHN@Z
//   ??$templateMethod@H@MyClass@MyNamespace@@QEAAHH@Z
//
// Which should be demangled to:
//   MyNamespace::MyClass::myMethod
//   MyNamespace::MyClass::templateMethod<int>

namespace MyNamespace {
    class MyClass {
    public:
        void myMethod(int x, double y) {
            // Simple method
        }
        
        template<typename T>
        T templateMethod(T value) {
            return value;
        }
    };
}

int main() {
    MyNamespace::MyClass obj;
    obj.myMethod(42, 3.14);
    obj.templateMethod<int>(100);
    return 0;
}
