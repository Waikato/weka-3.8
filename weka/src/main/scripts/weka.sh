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
#   is a JRE in jre/* (Linux) or ../runtime (Mac) relative to the directory that contains this script unless
#   the -jvm option is used to specify the location of the java executable to use.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
CLASSPATH="$DIR/weka.jar"

# On Linux, the JVM is in a folder called jre, otherwise assume we are on a Mac.
if [ -d "$DIR/jre" ]
then
  JCMD="$DIR/jre/*/bin/java"
else
  JCMD="$DIR/../runtime/Contents/Home/bin/java"
fi

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
  if [ "$ARG" = "-memory" ] || [ "$ARG" = "-main" ] || [ "$ARG" = "-no-gui" ] || [ "$ARG" = "-jvm" ]
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
  elif [ "$OPTION" = "-jvm" ]
  then
    JCMD=$ARG
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
  --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.annotation=ALL-UNNAMED --add-opens=java.base/java.lang.constant=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.module=ALL-UNNAMED --add-opens=java.base/java.lang.ref=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.lang.runtime=ALL-UNNAMED --add-opens=java.base/java.math=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.net.spi=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.nio.channels=ALL-UNNAMED --add-opens=java.base/java.nio.channels.spi=ALL-UNNAMED --add-opens=java.base/java.nio.charset=ALL-UNNAMED --add-opens=java.base/java.nio.charset.spi=ALL-UNNAMED --add-opens=java.base/java.nio.file=ALL-UNNAMED --add-opens=java.base/java.nio.file.attribute=ALL-UNNAMED --add-opens=java.base/java.nio.file.spi=ALL-UNNAMED --add-opens=java.base/java.security=ALL-UNNAMED --add-opens=java.base/java.security.cert=ALL-UNNAMED --add-opens=java.base/java.security.interfaces=ALL-UNNAMED --add-opens=java.base/java.security.spec=ALL-UNNAMED --add-opens=java.base/java.text=ALL-UNNAMED --add-opens=java.base/java.text.spi=ALL-UNNAMED --add-opens=java.base/java.time=ALL-UNNAMED --add-opens=java.base/java.time.chrono=ALL-UNNAMED --add-opens=java.base/java.time.format=ALL-UNNAMED --add-opens=java.base/java.time.temporal=ALL-UNNAMED --add-opens=java.base/java.time.zone=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.locks=ALL-UNNAMED --add-opens=java.base/java.util.function=ALL-UNNAMED --add-opens=java.base/java.util.jar=ALL-UNNAMED --add-opens=java.base/java.util.random=ALL-UNNAMED --add-opens=java.base/java.util.regex=ALL-UNNAMED --add-opens=java.base/java.util.spi=ALL-UNNAMED --add-opens=java.base/java.util.stream=ALL-UNNAMED --add-opens=java.base/java.util.zip=ALL-UNNAMED --add-opens=java.base/javax.crypto=ALL-UNNAMED --add-opens=java.base/javax.crypto.interfaces=ALL-UNNAMED --add-opens=java.base/javax.crypto.spec=ALL-UNNAMED --add-opens=java.base/javax.net=ALL-UNNAMED --add-opens=java.base/javax.net.ssl=ALL-UNNAMED --add-opens=java.base/javax.security.auth=ALL-UNNAMED --add-opens=java.base/javax.security.auth.callback=ALL-UNNAMED --add-opens=java.base/javax.security.auth.login=ALL-UNNAMED --add-opens=java.base/javax.security.auth.spi=ALL-UNNAMED --add-opens=java.base/javax.security.auth.x500=ALL-UNNAMED --add-opens=java.base/javax.security.cert=ALL-UNNAMED --add-opens=java.compiler/javax.annotation.processing=ALL-UNNAMED --add-opens=java.compiler/javax.lang.model=ALL-UNNAMED --add-opens=java.compiler/javax.lang.model.element=ALL-UNNAMED --add-opens=java.compiler/javax.lang.model.type=ALL-UNNAMED --add-opens=java.compiler/javax.lang.model.util=ALL-UNNAMED --add-opens=java.compiler/javax.tools=ALL-UNNAMED --add-opens=java.datatransfer/java.awt.datatransfer=ALL-UNNAMED --add-opens=java.desktop/java.applet=ALL-UNNAMED --add-opens=java.desktop/java.awt=ALL-UNNAMED --add-opens=java.desktop/java.awt.color=ALL-UNNAMED --add-opens=java.desktop/java.awt.desktop=ALL-UNNAMED --add-opens=java.desktop/java.awt.dnd=ALL-UNNAMED --add-opens=java.desktop/java.awt.event=ALL-UNNAMED --add-opens=java.desktop/java.awt.font=ALL-UNNAMED --add-opens=java.desktop/java.awt.geom=ALL-UNNAMED --add-opens=java.desktop/java.awt.im=ALL-UNNAMED --add-opens=java.desktop/java.awt.im.spi=ALL-UNNAMED --add-opens=java.desktop/java.awt.image=ALL-UNNAMED --add-opens=java.desktop/java.awt.image.renderable=ALL-UNNAMED --add-opens=java.desktop/java.awt.print=ALL-UNNAMED --add-opens=java.desktop/java.beans=ALL-UNNAMED --add-opens=java.desktop/java.beans.beancontext=ALL-UNNAMED --add-opens=java.desktop/javax.accessibility=ALL-UNNAMED --add-opens=java.desktop/javax.imageio=ALL-UNNAMED --add-opens=java.desktop/javax.imageio.event=ALL-UNNAMED --add-opens=java.desktop/javax.imageio.metadata=ALL-UNNAMED --add-opens=java.desktop/javax.imageio.plugins.bmp=ALL-UNNAMED --add-opens=java.desktop/javax.imageio.plugins.jpeg=ALL-UNNAMED --add-opens=java.desktop/javax.imageio.plugins.tiff=ALL-UNNAMED --add-opens=java.desktop/javax.imageio.spi=ALL-UNNAMED --add-opens=java.desktop/javax.imageio.stream=ALL-UNNAMED --add-opens=java.desktop/javax.print=ALL-UNNAMED --add-opens=java.desktop/javax.print.attribute=ALL-UNNAMED --add-opens=java.desktop/javax.print.attribute.standard=ALL-UNNAMED --add-opens=java.desktop/javax.print.event=ALL-UNNAMED --add-opens=java.desktop/javax.sound.midi=ALL-UNNAMED --add-opens=java.desktop/javax.sound.midi.spi=ALL-UNNAMED --add-opens=java.desktop/javax.sound.sampled=ALL-UNNAMED --add-opens=java.desktop/javax.sound.sampled.spi=ALL-UNNAMED --add-opens=java.desktop/javax.swing=ALL-UNNAMED --add-opens=java.desktop/javax.swing.border=ALL-UNNAMED --add-opens=java.desktop/javax.swing.colorchooser=ALL-UNNAMED --add-opens=java.desktop/javax.swing.event=ALL-UNNAMED --add-opens=java.desktop/javax.swing.filechooser=ALL-UNNAMED --add-opens=java.desktop/javax.swing.plaf=ALL-UNNAMED --add-opens=java.desktop/javax.swing.plaf.basic=ALL-UNNAMED --add-opens=java.desktop/javax.swing.plaf.metal=ALL-UNNAMED --add-opens=java.desktop/javax.swing.plaf.multi=ALL-UNNAMED --add-opens=java.desktop/javax.swing.plaf.nimbus=ALL-UNNAMED --add-opens=java.desktop/javax.swing.plaf.synth=ALL-UNNAMED --add-opens=java.desktop/javax.swing.table=ALL-UNNAMED --add-opens=java.desktop/javax.swing.text=ALL-UNNAMED --add-opens=java.desktop/javax.swing.text.html=ALL-UNNAMED --add-opens=java.desktop/javax.swing.text.html.parser=ALL-UNNAMED --add-opens=java.desktop/javax.swing.text.rtf=ALL-UNNAMED --add-opens=java.desktop/javax.swing.tree=ALL-UNNAMED --add-opens=java.desktop/javax.swing.undo=ALL-UNNAMED --add-opens=java.instrument/java.lang.instrument=ALL-UNNAMED --add-opens=java.logging/java.util.logging=ALL-UNNAMED --add-opens=java.management/java.lang.management=ALL-UNNAMED --add-opens=java.management/javax.management=ALL-UNNAMED --add-opens=java.management/javax.management.loading=ALL-UNNAMED --add-opens=java.management/javax.management.modelmbean=ALL-UNNAMED --add-opens=java.management/javax.management.monitor=ALL-UNNAMED --add-opens=java.management/javax.management.openmbean=ALL-UNNAMED --add-opens=java.management/javax.management.relation=ALL-UNNAMED --add-opens=java.management/javax.management.remote=ALL-UNNAMED --add-opens=java.management/javax.management.timer=ALL-UNNAMED --add-opens=java.management.rmi/javax.management.remote.rmi=ALL-UNNAMED --add-opens=java.naming/javax.naming=ALL-UNNAMED --add-opens=java.naming/javax.naming.directory=ALL-UNNAMED --add-opens=java.naming/javax.naming.event=ALL-UNNAMED --add-opens=java.naming/javax.naming.ldap=ALL-UNNAMED --add-opens=java.naming/javax.naming.ldap.spi=ALL-UNNAMED --add-opens=java.naming/javax.naming.spi=ALL-UNNAMED --add-opens=java.net.http/java.net.http=ALL-UNNAMED --add-opens=java.prefs/java.util.prefs=ALL-UNNAMED --add-opens=java.rmi/java.rmi=ALL-UNNAMED --add-opens=java.rmi/java.rmi.dgc=ALL-UNNAMED --add-opens=java.rmi/java.rmi.registry=ALL-UNNAMED --add-opens=java.rmi/java.rmi.server=ALL-UNNAMED --add-opens=java.rmi/javax.rmi.ssl=ALL-UNNAMED --add-opens=java.scripting/javax.script=ALL-UNNAMED --add-opens=java.security.jgss/javax.security.auth.kerberos=ALL-UNNAMED --add-opens=java.security.jgss/org.ietf.jgss=ALL-UNNAMED --add-opens=java.security.sasl/javax.security.sasl=ALL-UNNAMED --add-opens=java.smartcardio/javax.smartcardio=ALL-UNNAMED --add-opens=java.sql/java.sql=ALL-UNNAMED --add-opens=java.sql/javax.sql=ALL-UNNAMED --add-opens=java.sql.rowset/javax.sql.rowset=ALL-UNNAMED --add-opens=java.sql.rowset/javax.sql.rowset.serial=ALL-UNNAMED --add-opens=java.sql.rowset/javax.sql.rowset.spi=ALL-UNNAMED --add-opens=java.transaction.xa/javax.transaction.xa=ALL-UNNAMED --add-opens=java.xml/javax.xml=ALL-UNNAMED --add-opens=java.xml/javax.xml.catalog=ALL-UNNAMED --add-opens=java.xml/javax.xml.datatype=ALL-UNNAMED --add-opens=java.xml/javax.xml.namespace=ALL-UNNAMED --add-opens=java.xml/javax.xml.parsers=ALL-UNNAMED --add-opens=java.xml/javax.xml.stream=ALL-UNNAMED --add-opens=java.xml/javax.xml.stream.events=ALL-UNNAMED --add-opens=java.xml/javax.xml.stream.util=ALL-UNNAMED --add-opens=java.xml/javax.xml.transform=ALL-UNNAMED --add-opens=java.xml/javax.xml.transform.dom=ALL-UNNAMED --add-opens=java.xml/javax.xml.transform.sax=ALL-UNNAMED --add-opens=java.xml/javax.xml.transform.stax=ALL-UNNAMED --add-opens=java.xml/javax.xml.transform.stream=ALL-UNNAMED --add-opens=java.xml/javax.xml.validation=ALL-UNNAMED --add-opens=java.xml/javax.xml.xpath=ALL-UNNAMED --add-opens=java.xml/org.w3c.dom=ALL-UNNAMED --add-opens=java.xml/org.w3c.dom.bootstrap=ALL-UNNAMED --add-opens=java.xml/org.w3c.dom.events=ALL-UNNAMED --add-opens=java.xml/org.w3c.dom.ls=ALL-UNNAMED --add-opens=java.xml/org.w3c.dom.ranges=ALL-UNNAMED --add-opens=java.xml/org.w3c.dom.traversal=ALL-UNNAMED --add-opens=java.xml/org.w3c.dom.views=ALL-UNNAMED --add-opens=java.xml/org.xml.sax=ALL-UNNAMED --add-opens=java.xml/org.xml.sax.ext=ALL-UNNAMED --add-opens=java.xml/org.xml.sax.helpers=ALL-UNNAMED --add-opens=java.xml.crypto/javax.xml.crypto=ALL-UNNAMED --add-opens=java.xml.crypto/javax.xml.crypto.dom=ALL-UNNAMED --add-opens=java.xml.crypto/javax.xml.crypto.dsig=ALL-UNNAMED --add-opens=java.xml.crypto/javax.xml.crypto.dsig.dom=ALL-UNNAMED --add-opens=java.xml.crypto/javax.xml.crypto.dsig.keyinfo=ALL-UNNAMED --add-opens=java.xml.crypto/javax.xml.crypto.dsig.spec=ALL-UNNAMED --add-opens=javafx.base/javafx.beans=ALL-UNNAMED --add-opens=javafx.base/javafx.beans.binding=ALL-UNNAMED --add-opens=javafx.base/javafx.beans.property=ALL-UNNAMED --add-opens=javafx.base/javafx.beans.property.adapter=ALL-UNNAMED --add-opens=javafx.base/javafx.beans.value=ALL-UNNAMED --add-opens=javafx.base/javafx.collections=ALL-UNNAMED --add-opens=javafx.base/javafx.collections.transformation=ALL-UNNAMED --add-opens=javafx.base/javafx.event=ALL-UNNAMED --add-opens=javafx.base/javafx.util=ALL-UNNAMED --add-opens=javafx.base/javafx.util.converter=ALL-UNNAMED --add-opens=javafx.controls/javafx.scene.chart=ALL-UNNAMED --add-opens=javafx.controls/javafx.scene.control=ALL-UNNAMED --add-opens=javafx.controls/javafx.scene.control.cell=ALL-UNNAMED --add-opens=javafx.controls/javafx.scene.control.skin=ALL-UNNAMED --add-opens=javafx.fxml/javafx.fxml=ALL-UNNAMED --add-opens=javafx.graphics/javafx.animation=ALL-UNNAMED --add-opens=javafx.graphics/javafx.application=ALL-UNNAMED --add-opens=javafx.graphics/javafx.concurrent=ALL-UNNAMED --add-opens=javafx.graphics/javafx.css=ALL-UNNAMED --add-opens=javafx.graphics/javafx.css.converter=ALL-UNNAMED --add-opens=javafx.graphics/javafx.geometry=ALL-UNNAMED --add-opens=javafx.graphics/javafx.print=ALL-UNNAMED --add-opens=javafx.graphics/javafx.scene=ALL-UNNAMED --add-opens=javafx.graphics/javafx.scene.canvas=ALL-UNNAMED --add-opens=javafx.graphics/javafx.scene.effect=ALL-UNNAMED --add-opens=javafx.graphics/javafx.scene.image=ALL-UNNAMED --add-opens=javafx.graphics/javafx.scene.input=ALL-UNNAMED --add-opens=javafx.graphics/javafx.scene.layout=ALL-UNNAMED --add-opens=javafx.graphics/javafx.scene.paint=ALL-UNNAMED --add-opens=javafx.graphics/javafx.scene.robot=ALL-UNNAMED --add-opens=javafx.graphics/javafx.scene.shape=ALL-UNNAMED --add-opens=javafx.graphics/javafx.scene.text=ALL-UNNAMED --add-opens=javafx.graphics/javafx.scene.transform=ALL-UNNAMED --add-opens=javafx.graphics/javafx.stage=ALL-UNNAMED --add-opens=javafx.media/javafx.scene.media=ALL-UNNAMED --add-opens=javafx.swing/javafx.embed.swing=ALL-UNNAMED --add-opens=javafx.web/javafx.scene.web=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.jar=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.desktop/sun.awt=ALL-UNNAMED --add-opens=java.desktop/sun.java2d=ALL-UNNAMED --add-opens=javafx.graphics/com.sun.javafx.tk=ALL-UNNAMED --add-opens=javafx.graphics/com.sun.javafx.tk.quantum=ALL-UNNAMED --add-opens=javafx.graphics/com.sun.glass.ui=ALL-UNNAMED\
  -classpath "$CLASSPATH" \
  $HEAP \
  $HEADLESS \
  $MAIN \
  $ARGS
