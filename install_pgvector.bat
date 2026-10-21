@echo off
echo ==========================================
echo Installing pgvector into PostgreSQL 18...
echo ==========================================

set "PGROOT=C:\Program Files\PostgreSQL\18"
set "SRC=%TEMP%\pgvector"

if not exist "%SRC%\vector.dll" (
    echo Error: vector.dll not found in %SRC%!
    pause
    exit /b 1
)

copy /Y "%SRC%\vector.dll" "%PGROOT%\lib\"
if errorlevel 1 goto error

copy /Y "%SRC%\vector.control" "%PGROOT%\share\extension\"
if errorlevel 1 goto error

copy /Y "%SRC%\sql\vector--*.sql" "%PGROOT%\share\extension\"
if errorlevel 1 goto error

if not exist "%PGROOT%\include\server\extension\vector" (
    mkdir "%PGROOT%\include\server\extension\vector"
)
copy /Y "%SRC%\src\halfvec.h" "%PGROOT%\include\server\extension\vector\"
copy /Y "%SRC%\src\sparsevec.h" "%PGROOT%\include\server\extension\vector\"
copy /Y "%SRC%\src\vector.h" "%PGROOT%\include\server\extension\vector\"

echo ==========================================
echo SUCCESS! pgvector files installed successfully!
echo ==========================================
pause
exit /b 0

:error
echo ==========================================
echo FAILED: Please make sure you right-clicked and selected "Run as administrator".
echo ==========================================
pause
exit /b 1
