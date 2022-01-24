@echo off

REM Batch file for executing the RunWeka launcher class.
REM   RunWeka.bat <command>
REM Run with option "-h" to see available commands
REM 
REM Notes: 
REM - If you're getting an OutOfMemory Exception, increase the value of
REM   "maxheap" in the RunWeka.ini file.
REM - If you need more jars available in Weka, either include them in your
REM   %CLASSPATH% environment variable or add them to the "cp" placeholder
REM   in the RunWeka.ini file.
REM
REM Author:  FracPete (fracpete at waikato dot ac dot nz)
REM Version: $Revision: 1.6 $

REM Change to directory containing this script
pushd "%~dp0"

REM Set command variable
set _cmd=%1
if "%_cmd%"=="" set _cmd=default

REM Grab remaining options to pass to RunWeka in case user wants to run some WEKA class with options using "class"
shift
set args=%1
shift
:start
if [%1] == [] goto done
set args=%args% %1
shift
goto start
:done

REM Execute WEKA based on desired command
set _java=
for /f "delims=" %%a in ('where.exe /R . javaw.exe') do @set _java=%%a
set _string_to_prepend=start "WEKA Startup Console"
if "%_cmd%"=="-h" set _string_to_prepend= && for /f "delims=" %%a in ('where.exe /R . java.exe') do @set _java=%%a
if "%_cmd%"=="console" set _string_to_prepend= && for /f "delims=" %%a in ('where.exe /R . java.exe') do @set _java=%%a
if "%_cmd%"=="class" set _string_to_prepend= && for /f "delims=" %%a in ('where.exe /R . java.exe') do @set _java=%%a

%_string_to_prepend% "%_java%" -classpath . RunWeka -c %_cmd% -jre-path .\jre\* -- %args%

REM Go back to directory we came from
popd

