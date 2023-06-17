all: compiler

compiler: compiler.cpp dependency.cpp dependencyast.hpp dependency.hpp dependencyvisitor.hpp
	clang++ -std=c++20 compiler.cpp dependency.cpp -o $@ -g