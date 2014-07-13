@echo off


set BUILD_OPTIONS=-DCMAKE_INSTALL_PREFIX=D:\Development\fyp_workspace\ase_project\temp


set BUILD_OPTIONS=%BUILD_OPTIONS% -DREDUCTION_GENCODE_SM20=ON

set BUILD_OPTIONS=%BUILD_OPTIONS% -DBUILD_SHARED_LIBS=OFF

echo BUILD_OPTIONS [%BUILD_OPTIONS%]

cmake %BUILD_OPTIONS% D:\Development\fyp_workspace\ase_project
@echo on
pause