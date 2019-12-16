#!/bin/bash
# ----------------------------------------------------------------------------
#  Copyright 2001-2006 The Apache Software Foundation.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ----------------------------------------------------------------------------

#   Copyright (c) 2001-2002 The Apache Software Foundation.  All rights
#   reserved.

#   Copyright (C) 2011-2019 University of Waikato, Hamilton, NZ

#   Shell script for starting Weka under a *nix system. Assumes that there
#   is a JRE in jre/* under the same directory that contains this script.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
CLASSPATH="$DIR/weka.jar"

#JCMD=java
JCMD="$DIR/jre/*/bin/java"
#if [ -f "$JAVA_HOME/bin/java" ]
#then
#  JCMD="$JAVA_HOME/bin/java"
#fi

# check options
MEMORY=
HEAP=
MAIN=weka.gui.GUIChooser
ARGS=
OPTION=
HEADLESS=
WHITESPACE="[[:space:]]"
for ARG in "$@"
do
  if [ "$ARG" = "-memory" ] || [ "$ARG" = "-main" ] || [ "$ARG" = "-no-gui" ]
  then
  	OPTION=$ARG
  	continue
  fi

  if [ "$OPTION" = "-memory" ]
  then
    MEMORY=$ARG
    OPTION=""
    continue
  elif [ "$OPTION" = "-main" ]
  then
    MAIN=$ARG
    OPTION=""
    continue
  elif [ "$OPTION" = "-no-gui" ]
  then
    HEADLESS="-Djava.awt.headless=true"
    OPTION=""
    continue
  fi

  if [[ $ARG =~ $WHITESPACE ]]
  then
    ARGS="$ARGS \"$ARG\""
  else
    ARGS="$ARGS $ARG"
  fi
done

if [ -z "$MEMORY" ]
then
    HEAP=
else
    HEAP="-Xmx$MEMORY"
fi

# launch class
$JCMD \
  --add-opens java.base/java.lang=ALL-UNNAMED \
  -classpath "$CLASSPATH" \
  $HEAP \
  $HEADLESS \
  $MAIN \
  $ARGS
