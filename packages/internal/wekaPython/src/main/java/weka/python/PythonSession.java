/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    PythonSession.java
 *    Copyright (C) 2015 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.python;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.IOUtils;

import weka.core.Environment;
import weka.core.Instances;
import weka.core.WekaException;
import weka.core.WekaPackageManager;
import weka.gui.Logger;

/**
 * Class that manages interaction with the python micro server. Launches the
 * server and shuts it down on VM exit.
 *
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 * @version $Revision: $
 */
public class PythonSession {

  public static enum PythonVariableType {
    DataFrame, Image, String, Unknown;
  }

  /** The session singleton (for backward compatibility) */
  private static PythonSession s_sessionSingleton;

  /** The results of the python check script */
  private static String s_pythonEnvCheckResults = "";

  /**
   * Static map of initialized servers. A "default" entry will be used for
   * backward compatibility
   */
  private static Map<String, PythonSession> m_pythonServers =
    Collections.synchronizedMap(new HashMap<String, PythonSession>());

  private static Map<String, String> m_pythonEnvCheckResults =
    Collections.synchronizedMap(new HashMap<String, String>());

  /** The command used to start python for this session */
  private String m_pythonCommand;

  /** The unique key for this session/server */
  private String m_sessionKey;

  /** the current session holder */
  private Object m_sessionHolder;

  /** For locking */
  protected SessionMutex m_mutex = new SessionMutex();

  /** Server socket */
  protected ServerSocket m_serverSocket;

  /** Local socket for comms with the python server */
  protected Socket m_localSocket;

  /** The process executing the server */
  protected Process m_serverProcess;

  /** True when the server has been shutdown */
  protected boolean m_shutdown;

  /** A shutdown hook for stopping the server */
  protected Thread m_shutdownHook;

  /** PID of the running python server */
  protected int m_pythonPID = -1;

  /** True to output debugging info */
  protected boolean m_debug;

  /** Logger to use (if any) */
  protected Logger m_log;

  /**
   * Acquire the default session for the requester
   *
   * @param requester the object requesting the session
   * @return the default session singleton
   * @throws WekaException if python is not available
   */
  public static PythonSession acquireSession(Object requester)
    throws WekaException {

    if (s_sessionSingleton == null) {
      throw new WekaException("Python not available!");
    }

    return s_sessionSingleton.getSession(requester);
  }

  /**
   * @param pythonCommand command (either fully qualified path or that which is
   *          in the PATH). This, plus the optional ownerID is used to lookup
   *          and return a session/server.
   * @param ownerID an optional ownerID string for acquiring the session. This
   *          can be used to restrict the session/server to one (or more)
   *          clients (i.e. those with the ID "ownerID").
   * @param requester the object requesting the session/server
   * @return a session
   * @throws WekaException if the requested session/server is not available (or
   *           does not exist).
   */
  public static PythonSession acquireSession(String pythonCommand,
    String ownerID, Object requester) throws WekaException {
    String key =
      pythonCommand + (ownerID != null && ownerID.length() > 0 ? ownerID : "");
    if (!m_pythonServers.containsKey(key)) {
      throw new WekaException(
        "Python session " + key + " does not seem to exist!");
    }
    return m_pythonServers.get(key).getSession(requester);
  }

  /**
   * Release the default session so that other clients can obtain it. This
   * method does nothing if the requester is not the current session holder
   *
   * @param requester the session holder
   */
  public static void releaseSession(Object requester) {
    s_sessionSingleton.dropSession(requester);
  }

  /**
   * Release the user-specified python session.
   *
   * @param pythonCommand command (either fully qualified path or that which is
   *          in the PATH). This, plus the optional ownerID is used to lookup a
   *          session/server.
   * @param ownerID an optional ownerID string for identifying the session. This
   *          can be used to restrict the session/server to one (or more)
   *          clients (i.e. those with the ID "ownerID").
   * @param requester the object requesting the session/server
   * @throws WekaException if the requested session/server is not available (or
   *           does not exist).
   */
  public static void releaseSession(String pythonCommand, String ownerID,
    Object requester) throws WekaException {
    String key =
      pythonCommand + (ownerID != null && ownerID.length() > 0 ? ownerID : "");
    if (!m_pythonServers.containsKey(key)) {
      throw new WekaException(
        "Python session " + key + " does not seem to exist!");
    }
    m_pythonServers.get(key).dropSession(requester);
  }

  /**
   * Returns true if (at least) the default python environment/server is
   * available
   *
   * @return true if the default python environment/server is available
   */
  public static synchronized boolean pythonAvailable() {
    return s_sessionSingleton != null;
  }

  /**
   * Returns true if the user-specified python environment/server (as specified
   * by pythonCommand (and optional ownerID) is available.
   * 
   * @param pythonCommand command (either fully qualified path or that which is
   *          in the PATH). This, plus the optional ownerID is used to lookup a
   *          session/server.
   * @param ownerID an optional ownerID string for identifying the session. This
   *          can be used to restrict the session/server to one (or more)
   *          clients (i.e. those with the ID "ownerID").
   */
  public static synchronized boolean pythonAvailable(String pythonCommand,
    String ownerID) {
    String key =
      pythonCommand + (ownerID != null && ownerID.length() > 0 ? ownerID : "");
    return m_pythonServers.containsKey(key);
  }

  /**
   * Private constructor
   *
   * @param pythonCommand the command used to start python
   * @param ownerID the (optional) owner ID of this server/session
   * @param pathEntries optional additional entries that need to be in the PATH
   *          in order for the python environment to work correctly
   * @param debug true for debugging output
   * @param defaultServer true if the default (i.e. python in the system PATH)
   *          server is to be used
   * @throws IOException if a problem occurs
   */
  private PythonSession(String pythonCommand, String ownerID,
    String pathEntries, boolean debug, boolean defaultServer)
    throws IOException {
    m_debug = debug;

    m_pythonCommand = pythonCommand;
    String key =
      pythonCommand + (ownerID != null && ownerID.length() > 0 ? ownerID : "");
    m_sessionKey = key;
    if (PythonSession.m_pythonServers.containsKey(m_sessionKey)) {
      throw new IOException(
        "A server session for " + m_sessionKey + " Already exists!");
    }

    if (!defaultServer && pathEntries != null && pathEntries.length() > 0) {
      // use a shell/batch script to launch the server (so that we can
      // set the path so that the python server works correctly)
      String envCheckResults = writeAndLaunchPyCheck(pathEntries);
      m_pythonEnvCheckResults.put(m_sessionKey, envCheckResults);
      if (envCheckResults.length() < 5) {
        // launch server
        launchServerScript(pathEntries);
        m_pythonServers.put(m_sessionKey, this);
      }
    } else {
      String tester = WekaPackageManager.PACKAGES_DIR.getAbsolutePath()
        + File.separator + "wekaPython" + File.separator + "resources"
        + File.separator + "py" + File.separator + "pyCheck.py";

      ProcessBuilder builder = new ProcessBuilder(pythonCommand, tester);
      Process pyProcess = builder.start();
      StringWriter writer = new StringWriter();
      IOUtils.copy(pyProcess.getInputStream(), writer);
      String envCheckResults = writer.toString();
      m_shutdown = false;

      m_pythonEnvCheckResults.put(m_sessionKey, envCheckResults);
      if (envCheckResults.length() < 5) {
        // launch server
        launchServer(true);
        m_pythonServers.put(m_sessionKey, this);

        if (s_sessionSingleton == null && defaultServer) {
          s_sessionSingleton = this;
          s_pythonEnvCheckResults = envCheckResults;
        }
      }
    }
  }

  /**
   * Private constructor
   *
   * @param pythonCommand the command used to start python
   * @param debug true for debugging output
   * @throws IOException if a problem occurs
   */
  private PythonSession(String pythonCommand, boolean debug)
    throws IOException {
    this(pythonCommand, null, null, debug, true);
  }

  /**
   * Gets the access to python for a requester. Handles locking.
   *
   * @param requester the requesting object
   * @return the session
   * @throws WekaException if python is not available
   */
  private synchronized PythonSession getSession(Object requester)
    throws WekaException {

    if (m_sessionHolder == requester) {
      return this;
    }

    m_mutex.safeLock();
    m_sessionHolder = requester;
    return this;
  }

  /**
   * Release the session for a requester
   *
   * @param requester the requesting object
   */
  private void dropSession(Object requester) {
    if (requester == m_sessionHolder) {
      m_sessionHolder = null;
      m_mutex.unlock();
    }
  }

  /**
   * Executes the python environment check script via a wrapping shell/batch script.
   *
   * @param pathEntries additional entries for the PATH that are required in order for
   *                    python to execute correctly
   * @return the result of executing the python environment check
   * @throws IOException if a problem occurs
   */
  private String writeAndLaunchPyCheck(String pathEntries) throws IOException {

    String osType = System.getProperty("os.name");
    boolean windows =
      osType != null && osType.toLowerCase().contains("windows");

    String script = getLaunchScript(pathEntries, "pyCheck.py", windows);

    // System.err.println("**** Executing shell script: \n\n" + script);

    String scriptPath =
      File.createTempFile("nixtester_", windows ? ".bat" : ".sh").toString();
    // System.err.println("Script path: " + scriptPath);

    FileWriter fwriter = new FileWriter(scriptPath);
    fwriter.write(script);
    fwriter.flush();
    fwriter.close();

    if (!windows) {
      Runtime.getRuntime().exec("chmod u+x " + scriptPath);
    }
    ProcessBuilder builder = new ProcessBuilder(scriptPath);
    Process pyProcess = builder.start();
    StringWriter writer = new StringWriter();
    IOUtils.copy(pyProcess.getInputStream(), writer);

    return writer.toString();
  }

  /**
   * Generates a script (shell or batch) by which to execute a given python script.
   *
   * @param pathEntries additional PATH entries needed for python to execute correctly
   * @param pyScriptName the name of the script (in wekaPython/resources/py) to execute
   * @param windows true if we are running under Windows
   * @param scriptArgs optional arguments for the python script
   * @return the generated sh/bat script
   */
  private String getLaunchScript(String pathEntries, String pyScriptName,
    boolean windows, Object... scriptArgs) {
    String script = WekaPackageManager.PACKAGES_DIR.getAbsolutePath()
      + File.separator + "wekaPython" + File.separator + "resources"
      + File.separator + "py" + File.separator + pyScriptName;

    String pathOriginal =
      Environment.getSystemWide().getVariableValue(windows ? "Path" : "PATH");
    if (pathOriginal == null) {
      pathOriginal = "";
    }

    File pythFile = new File(m_pythonCommand);
    String exeDir =
      pythFile.getParent() != null ? pythFile.getParent().toString() : "";

    String finalPath = pathEntries != null && pathEntries.length() > 0
      ? pathEntries + File.pathSeparator + pathOriginal
      : pathOriginal;

    finalPath = exeDir.length() > 0 ? exeDir + File.pathSeparator + finalPath
      : "" + finalPath;

    StringBuilder sbuilder = new StringBuilder();
    if (windows) {
      sbuilder.append("@echo off").append("\n\n");
      sbuilder.append("PATH=" + finalPath).append("\n\n");
      sbuilder.append("python " + script);
    } else {
      sbuilder.append("#!/bin/sh").append("\n\n");
      sbuilder.append("export PATH=" + finalPath).append("\n\n");
      sbuilder.append("python " + script);
    }

    for (Object arg : scriptArgs) {
      sbuilder.append(" ").append(arg.toString());
    }
    sbuilder.append("\n");

    return sbuilder.toString();
  }

  /**
   * Starts the server socket.
   *
   * @return the Thread that the server socket is waiting for a connection on.
   * @throws IOException if a problem occurs
   */
  private Thread startServerSocket() throws IOException {
    if (m_debug) {
      System.err.println("Launching server socket...");
    }
    m_serverSocket = new ServerSocket(0);
    m_serverSocket.setSoTimeout(12000);

    Thread acceptThread = new Thread() {
      @Override
      public void run() {
        try {
          m_localSocket = m_serverSocket.accept();
        } catch (IOException e) {
          m_localSocket = null;
        }
      }
    };
    acceptThread.start();

    return acceptThread;
  }

  /**
   * If the local socket could not be created, this method shuts down the
   * server. Otherwise, a shutdown hook is added to bring the server down when
   * the JVM exits.
   * 
   * @throws IOException if a problem occurs
   */
  private void checkLocalSocketAndCreateShutdownHook() throws IOException {
    if (m_localSocket == null) {
      shutdown();
      throw new IOException("Was unable to start python server");
    } else {
      m_pythonPID =
        ServerUtils.receiveServerPIDAck(m_localSocket.getInputStream());

      m_shutdownHook = new Thread() {
        @Override
        public void run() {
          shutdown();
        }
      };
      Runtime.getRuntime().addShutdownHook(m_shutdownHook);
    }
  }

  /**
   * Launches the python server. Performs some basic requirements checks for the
   * python environment - e.g. python needs to have numpy, pandas and sklearn
   * installed.
   *
   * @param startPython true if the server is to actually be started. False is
   *          really just for debugging/development where the server can be
   *          manually started in a separate terminal
   * @throws IOException if a problem occurs
   */
  private void launchServer(boolean startPython) throws IOException {
    Thread acceptThread = startServerSocket();
    int localPort = m_serverSocket.getLocalPort();

    if (startPython) {
      String serverScript = WekaPackageManager.PACKAGES_DIR.getAbsolutePath()
        + File.separator + "wekaPython" + File.separator + "resources"
        + File.separator + "py" + File.separator + "pyServer.py";
      ProcessBuilder processBuilder = new ProcessBuilder(m_pythonCommand,
        serverScript, "" + localPort, m_debug ? "debug" : "");
      m_serverProcess = processBuilder.start();
    }
    try {
      acceptThread.join();
    } catch (InterruptedException e) {
    }

    checkLocalSocketAndCreateShutdownHook();
  }

  /**
   * Launches the server using a shell/batch script generated on the fly. Used
   * when the user supplies a path to a python executable and additional entries
   * are needed in the PATH.
   *
   * @param pathEntries entries to prepend to the PATH
   * @throws IOException if a problem occurs when launching the server
   */
  private void launchServerScript(String pathEntries) throws IOException {
    String osType = System.getProperty("os.name");
    boolean windows =
      osType != null && osType.toLowerCase().contains("windows");

    Thread acceptThread = startServerSocket();
    String script = getLaunchScript(pathEntries, "pyServer.py", windows,
      m_serverSocket.getLocalPort(), m_debug ? "debug" : "");
    if (m_debug) {
      System.err.println("Executing server launch script:\n\n" + script);
    }

    String scriptPath =
      File.createTempFile("pyserver_", windows ? ".bat" : ".sh").toString();

    FileWriter fwriter = new FileWriter(scriptPath);
    fwriter.write(script);
    fwriter.flush();
    fwriter.close();

    if (!windows) {
      Runtime.getRuntime().exec("chmod u+x " + scriptPath);
    }

    ProcessBuilder processBuilder = new ProcessBuilder(scriptPath);
    m_serverProcess = processBuilder.start();
    try {
      acceptThread.join();
    } catch (InterruptedException e) {
    }

    checkLocalSocketAndCreateShutdownHook();
  }

  /**
   * Set a log
   *
   * @param log the log to use
   */
  public void setLog(Logger log) {
    m_log = log;
  }

  /**
   * Get the type of a variable in python
   * 
   * @param varName the name of the variable to get the type for
   * @param debug true for debugging output
   * @return the type of the variable. Known types, for which we can do useful
   *         things with in the Weka environment, are pandas data frames (can be
   *         converted to instances), pyplot figure/images (retrieve as png) and
   *         textual data. Any variable of type unknown should be able to be
   *         retrieved in string form.
   * @throws WekaException if a problem occurs
   */
  public PythonVariableType getPythonVariableType(String varName, boolean debug)
    throws WekaException {

    try {
      return ServerUtils.getPythonVariableType(varName,
        m_localSocket.getOutputStream(), m_localSocket.getInputStream(), m_log,
        debug);
    } catch (Exception ex) {
      throw new WekaException(ex);
    }
  }

  /**
   * Transfer Weka instances into python as a named pandas data frame
   *
   * @param instances the instances to transfer
   * @param pythonFrameName the name of the data frame to use in python
   * @param debug true for debugging output
   * @throws WekaException if a problem occurs
   */
  public void instancesToPython(Instances instances, String pythonFrameName,
    boolean debug) throws WekaException {
    try {
      ServerUtils.sendInstances(instances, pythonFrameName,
        m_localSocket.getOutputStream(), m_localSocket.getInputStream(), m_log,
        debug);
    } catch (Exception ex) {
      throw new WekaException(ex);
    }
  }

  /**
   * Transfer Weka instances into python as a pandas data frame and then extract
   * out numpy arrays of input and target features/columns. These arrays are
   * named X and Y respectively in python. If there is no class set in the
   * instances then only an X array is extracted.
   *
   * @param instances the instances to transfer
   * @param pythonFrameName the name of the pandas data frame to use in python
   * @param debug true for debugging output
   * @throws WekaException if a problem occurs
   */
  public void instancesToPythonAsScikitLearn(Instances instances,
    String pythonFrameName, boolean debug) throws WekaException {
    try {
      ServerUtils.sendInstancesScikitLearn(instances, pythonFrameName,
        m_localSocket.getOutputStream(), m_localSocket.getInputStream(), m_log,
        debug);
    } catch (Exception ex) {
      throw new WekaException(ex);
    }
  }

  /**
   * Retrieve a pandas data frame from Python and convert it to a set of
   * instances. The resulting set of instances will not have a class index set.
   *
   * @param frameName the name of the pandas data frame to extract and convert
   *          to instances
   * @param debug true for debugging output
   * @return an Instances object
   * @throws WekaException if the named data frame does not exist in python or
   *           is not a pandas data frame
   */
  public Instances getDataFrameAsInstances(String frameName, boolean debug)
    throws WekaException {
    try {
      return ServerUtils.receiveInstances(frameName,
        m_localSocket.getOutputStream(), m_localSocket.getInputStream(), m_log,
        debug);
    } catch (IOException ex) {
      throw new WekaException(ex);
    }
  }

  /**
   * Execute an arbitrary script in python
   *
   * @param pyScript the script to execute
   * @param debug true for debugging output
   * @return a List of strings - index 0 contains std out from the script and
   *         index 1 contains std err
   * @throws WekaException if a problem occurs
   */
  public List<String> executeScript(String pyScript, boolean debug)
    throws WekaException {
    try {
      return ServerUtils.executeUserScript(pyScript,
        m_localSocket.getOutputStream(), m_localSocket.getInputStream(), m_log,
        debug);
    } catch (IOException ex) {
      throw new WekaException(ex);
    }
  }

  /**
   * Check if a named variable is set/exists in the python environment
   *
   * @param varName the name of the variable to check
   * @param debug true for debugging output
   * @return true if the variable is set in python
   * @throws WekaException if a problem occurs
   */
  public boolean checkIfPythonVariableIsSet(String varName, boolean debug)
    throws WekaException {
    try {
      return ServerUtils.checkIfPythonVariableIsSet(varName,
        m_localSocket.getOutputStream(), m_localSocket.getInputStream(), m_log,
        debug);
    } catch (IOException ex) {
      throw new WekaException(ex);
    }
  }

  /**
   * Attempt to retrieve the value of a variable in python using serialization
   * to Json. If successful, then the resulting Object is either a Map or List
   * containing more Maps and Lists that represent the Json structure of the
   * serialized variable
   *
   * @param varName the name of the variable to retrieve
   * @param debug true for debugging output
   * @return a Map/List based structure
   * @throws WekaException if a problem occurs
   */
  public Object getVariableValueFromPythonAsJson(String varName, boolean debug)
    throws WekaException {
    try {
      return ServerUtils.receiveJsonVariableValue(varName,
        m_localSocket.getOutputStream(), m_localSocket.getInputStream(), m_log,
        debug);
    } catch (IOException ex) {
      throw new WekaException(ex);
    }
  }

  /**
   * Attempt to retrieve the value of a variable in python using pickle
   * serialization. If successful, then the result is a string containing the
   * pickled object.
   *
   * @param varName the name of the variable to retrieve
   * @param debug true for debugging output
   * @return a string containing the pickled variable value
   * @throws WekaException if a problem occurs
   */
  public String getVariableValueFromPythonAsPickledObject(String varName,
    boolean debug) throws WekaException {
    try {
      return ServerUtils.receivePickledVariableValue(varName,
        m_localSocket.getOutputStream(), m_localSocket.getInputStream(), false,
        m_log, debug);
    } catch (IOException ex) {
      throw new WekaException(ex);
    }
  }

  /**
   * Attempt to retrieve the value of a variable in python as a plain string
   * (i.e. executes a 'str(varName)' in python).
   *
   * @param varName the name of the variable to retrieve
   * @param debug true for debugging output
   * @return the value of the variable as a plain string
   * @throws WekaException if a problem occurs
   */
  public String getVariableValueFromPythonAsPlainString(String varName,
    boolean debug) throws WekaException {
    try {
      return ServerUtils.receivePickledVariableValue(varName,
        m_localSocket.getOutputStream(), m_localSocket.getInputStream(), true,
        m_log, debug);
    } catch (IOException ex) {
      throw new WekaException(ex);
    }
  }

  /**
   * Get a list of variables that are set in python. Returns a list of two
   * element string arrays. Each entry in the list is a variable. The first
   * element of the array is the name of the variable and the second is its type
   * in python.
   *
   * @param debug true if debugging info is to be output
   * @return a list of variables set in python
   * @throws WekaException if a problem occurs
   */
  public List<String[]> getVariableListFromPython(boolean debug)
    throws WekaException {
    try {
      return ServerUtils.receiveVariableList(m_localSocket.getOutputStream(),
        m_localSocket.getInputStream(), m_log, debug);
    } catch (IOException ex) {
      throw new WekaException(ex);
    }
  }

  /**
   * Push a pickled python variable value back into python. Deserializes the
   * value in python.
   *
   * @param varName the name of the variable in python that will hold the
   *          deserialized value
   * @param varValue the pickled string value of the variable
   * @param debug true for debugging output
   * @throws WekaException if a problem occurs
   */
  public void setPythonPickledVariableValue(String varName, String varValue,
    boolean debug) throws WekaException {
    try {
      ServerUtils.sendPickledVariableValue(varName, varValue,
        m_localSocket.getOutputStream(), m_localSocket.getInputStream(), m_log,
        debug);
    } catch (IOException ex) {
      throw new WekaException(ex);
    }
  }

  /**
   * Grab the contents of the debug buffer from the python server. The server
   * redirects both sys out and sys err to StringIO objects. If debug has been
   * specified, then server debugging output will have been collected in these
   * buffers. Note that the buffers will potentially also contain output from
   * the execution of arbitrary scripts too. Calling this method also resets the
   * buffers.
   *
   * @param debug true for debugging output (from the execution of this specific
   *          command)
   * @return the contents of the sys out and sys err streams. Element 0 in the
   *         list contains sys out and element 1 contains sys err
   * @throws WekaException if a problem occurs
   */
  public List<String> getPythonDebugBuffer(boolean debug) throws WekaException {
    try {
      return ServerUtils.receiveDebugBuffer(m_localSocket.getOutputStream(),
        m_localSocket.getInputStream(), m_log, debug);
    } catch (IOException ex) {
      throw new WekaException(ex);
    }
  }

  /**
   * Retrieve an image from python. Assumes that the image in python is stored
   * as a matplotlib.figure.Figure object. Returns a BufferedImage containing
   * the image.
   *
   * @param varName the name of the variable in python that contains the image
   * @param debug true to output debugging info
   * @return a BufferedImage
   * @throws WekaException if the variable doesn't exist, doesn't contain a
   *           Figure object or there is a comms error.
   */
  public BufferedImage getImageFromPython(String varName, boolean debug)
    throws WekaException {
    try {
      return ServerUtils.getPNGImageFromPython(varName,
        m_localSocket.getOutputStream(), m_localSocket.getInputStream(), m_log,
        debug);
    } catch (IOException ex) {
      throw new WekaException(ex);
    }
  }

  /**
   * Shutdown the python server
   */
  private void shutdown() {
    if (!m_shutdown) {
      try {
        m_shutdown = true;
        if (m_localSocket != null) {
          if (m_debug) {
            System.err.println("Sending shutdown command...");
          }
          if (m_debug) {
            List<String> outAndErr =
              ServerUtils.receiveDebugBuffer(m_localSocket.getOutputStream(),
                m_localSocket.getInputStream(), m_log, m_debug);
            if (outAndErr.get(0).length() > 0) {
              System.err
                .println("Python debug std out:\n" + outAndErr.get(0) + "\n");
            }
            if (outAndErr.get(1).length() > 0) {
              System.err
                .println("Python debug std err:\n" + outAndErr.get(1) + "\n");
            }
          }
          ServerUtils.sendServerShutdown(m_localSocket.getOutputStream());
          m_localSocket.close();
          if (m_serverProcess != null) {
            m_serverProcess.destroy();
          }
        }

        if (m_serverSocket != null) {
          m_serverSocket.close();
        }
        s_sessionSingleton = null;
        m_pythonServers.remove(m_sessionKey);
        m_pythonEnvCheckResults.remove(m_sessionKey);
      } catch (Exception ex) {
        ex.printStackTrace();
      }
    }
  }

  /**
   * Initialize the default session. This needs to be called exactly once in
   * order to run checks and launch the server. Creates a session singleton.
   *
   * @param pythonCommand the python command
   * @param debug true for debugging output
   * @return true if the server launched successfully
   * @throws WekaException if there was a problem - missing packages in python,
   *           or python could not be started for some reason
   */
  public static synchronized boolean initSession(String pythonCommand,
    boolean debug) throws WekaException {

    if (s_sessionSingleton == null) {
      try {
        new PythonSession(pythonCommand, debug);
      } catch (IOException ex) {
        throw new WekaException(ex);
      }
    }

    return s_pythonEnvCheckResults.length() < 5;
  }

  /**
   * Initialize a server/session for a user-supplied python path and (optional)
   * ownerID.
   *
   * @param pythonCommand command (either fully qualified path or that which is
   *          in the PATH). This, plus the optional ownerID is used to lookup
   *          and return a session/server.
   * @param ownerID an optional ownerID string for acquiring the session. This
   *          can be used to restrict the session/server to one (or more)
   *          clients (i.e. those with the ID "ownerID").
   * @param pathEntries optional entries that need to be in the PATH in order
   *          for the python environment to work correctly
   * @param debug true for debugging info
   * @return true if the server launched successfully
   * @throws WekaException if the requested session/server is not available (or
   *           does not exist).
   */
  public static synchronized boolean initSession(String pythonCommand,
    String ownerID, String pathEntries, boolean debug) throws WekaException {
    String key =
      pythonCommand + (ownerID != null && ownerID.length() > 0 ? ownerID : "");

    if (!m_pythonServers.containsKey(key)) {
      try {
        new PythonSession(pythonCommand, ownerID, pathEntries, debug, false);
      } catch (IOException ex) {
        throw new WekaException(ex);
      }
    }

    return m_pythonEnvCheckResults.get(key).length() < 5;
  }

  /**
   * Gets the result of running the checks in python
   *
   * @return a string containing the possible errors
   */
  public static String getPythonEnvCheckResults() {
    return s_pythonEnvCheckResults;
  }

  /**
   * Gets the result of running the checks in python for the given python path +
   * optional ownerID.
   *
   * @param pythonCommand command (either fully qualified path or that which is
   *          in the PATH). This, plus the optional ownerID is used to lookup
   *          and return a session/server.
   * @param ownerID an optional ownerID string for acquiring the session. This
   *          can be used to restrict the session/server to one (or more)
   *          clients (i.e. those with the ID "ownerID").
   * @return a string containing the possible errors
   * @throws WekaException if the requested server does not exist
   */
  public static String getPythonEnvCheckResults(String pythonCommand,
    String ownerID) throws WekaException {
    String key =
      pythonCommand + (ownerID != null && ownerID.length() > 0 ? ownerID : "");

    if (!m_pythonEnvCheckResults.containsKey(key)) {
      throw new WekaException("The specified server/environment (" + key
        + ") does not seem to exist!");
    }

    return m_pythonEnvCheckResults.get(key);
  }

  /**
   * Some quick tests...
   *
   * @param args
   */
  public static void main(String[] args) {
    String pythonCommand = "python"; // default - use the python in the PATH
    String pathEntries = null;
    if (args.length > 0) {
      pythonCommand = args[0];
    }
    if (args.length > 1) {
      pathEntries = args[1];
    }
    try {
      String temp = "myTempOwnerID";
      if (!PythonSession.initSession(pythonCommand,
        args.length > 0 ? temp : null, pathEntries, true)) {
        System.err.println("Initialization failed!");
        System.exit(1);
      }

      PythonSession session = PythonSession.acquireSession(pythonCommand,
        args.length > 0 ? temp : null, temp); // temp is requester too
      // String script =
      // "import matplotlib.pyplot as plt\nfig, ax = plt.subplots( nrows=1,
      // ncols=1 )\n"
      // + "ax.plot([0,1,2], [10,20,3])\n";
      // String script = "my_var = 'hello'\n";

      String script =
        "from sklearn import datasets\nfrom pandas import DataFrame\ndiabetes = "
          + "datasets.load_diabetes()\ndd = DataFrame(diabetes.data)\n";
      session.executeScript(script, true);

      script = "def foo():\n\treturn 100\n\nx = foo()\n";
      session.executeScript(script, true);

      script = "import sklearn\nv=sklearn.__version__\n";
      session.executeScript(script, true);

      // BufferedImage img = session.getImageFromPython("fig", true);
      List<String[]> vars = session.getVariableListFromPython(true);
      for (String[] v : vars) {
        System.err.println(v[0] + ":" + v[1]);
      }

      Object result = session.getVariableValueFromPythonAsJson("v", true);
      System.err.println("Value of v: " + result.toString());
    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }
}
