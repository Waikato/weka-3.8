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
 *    SubspaceCluster.java
 *    Copyright (C) 2001-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.datagenerators.clusterers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Range;
import weka.core.RevisionUtils;
import weka.core.Tag;
import weka.core.Utils;
import weka.datagenerators.ClusterDefinition;
import weka.datagenerators.ClusterGenerator;

/**
 * <!-- globalinfo-start --> A data generator that produces data points in
 * hyperrectangular subspace clusters.
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -h
 *  Prints this help.
 * </pre>
 * 
 * <pre>
 * -o &lt;file&gt;
 *  The name of the output file, otherwise the generated data is
 *  printed to stdout.
 * </pre>
 * 
 * <pre>
 * -r &lt;name&gt;
 *  The name of the relation.
 * </pre>
 * 
 * <pre>
 * -d
 *  Whether to print debug informations.
 * </pre>
 * 
 * <pre>
 * -S
 *  The seed for random function (default 1)
 * </pre>
 * 
 * <pre>
 * -a &lt;num&gt;
 *  The number of attributes (default 1).
 * </pre>
 * 
 * <pre>
 * -c
 *  Class Flag, if set, the cluster is listed in extra attribute.
 * </pre>
 * 
 * <pre>
 * -b &lt;range&gt;
 *  The indices for boolean attributes.
 * </pre>
 * 
 * <pre>
 * -m &lt;range&gt;
 *  The indices for nominal attributes.
 * </pre>
 *
 * <pre>
 * -C &lt;cluster-definition&gt;
 *  A cluster definition of class 'SubspaceClusterDefinition'
 *  (definition needs to be quoted to be recognized as 
 *  a single argument).
 * </pre>
 * 
 * <pre>
 * Options specific to weka.datagenerators.clusterers.SubspaceClusterDefinition:
 * </pre>
 * 
 * <pre>
 * -A &lt;range&gt;
 *  Uses a random uniform distribution for the instances in the cluster.
 * </pre>
 *
 * <pre>
 * -U &lt;range&gt;
 *  Generates totally uniformly distributed instances in the cluster.
 * </pre>
 *
 * <pre>
 * -G &lt;range&gt;
 *  Uses a Gaussian distribution for the instances in the cluster.
 * </pre>
 * 
 * <pre>
 * -D &lt;num&gt;,&lt;num&gt;
 *  The attribute min/max (-A and -U) or mean/stddev (-G) for
 *  the cluster.
 * </pre>
 * 
 * <pre>
 * -N &lt;num&gt;..&lt;num&gt;
 *  The range of number of instances per cluster (default 1..50).
 * </pre>
 * 
 * <pre>
 * -I
 *  Uses integer instead of continuous values (default continuous).
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * @author Gabi Schmidberger (gabi@cs.waikato.ac.nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 */
public class SubspaceCluster extends ClusterGenerator {

  /** for serialization */
  static final long serialVersionUID = -3454999858505621128L;

  /** Stores which columns are boolean (default numeric) */
  protected Range m_booleanCols = new Range();

  /** Stores which columns are nominal (default numeric) */
  protected Range m_nominalCols = new Range();

  /** cluster list */
  protected ClusterDefinition[] m_Clusters = new ClusterDefinition[] { new SubspaceClusterDefinition(this) };;

  /** if nominal, store number of values */
  protected int[] m_numValues;

  /** cluster type: uniform/random */
  public static final int UNIFORM_RANDOM = 0;
  /** cluster type: total uniform */
  public static final int TOTAL_UNIFORM = 1;
  /** cluster type: gaussian */
  public static final int GAUSSIAN = 2;
  /** the tags for the cluster types */
  public static final Tag[] TAGS_CLUSTERTYPE = {
    new Tag(UNIFORM_RANDOM, "uniform/random"),
    new Tag(TOTAL_UNIFORM, "total uniform"), new Tag(GAUSSIAN, "gaussian") };

  /** cluster subtype: continuous */
  public static final int CONTINUOUS = 0;
  /** cluster subtype: integer */
  public static final int INTEGER = 1;
  /** the tags for the cluster types */
  public static final Tag[] TAGS_CLUSTERSUBTYPE = {
    new Tag(CONTINUOUS, "continuous"), new Tag(INTEGER, "integer") };

  /**
   * initializes the generator, sets the number of clusters to 0, since user has
   * to specify them explicitly
   */
  public SubspaceCluster() {
    super();
  }

  /**
   * Returns a string describing this data generator.
   * 
   * @return a description of the data generator suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String globalInfo() {
    return "A data generator that produces data points in hyperrectangular subspace clusters.";
  }

  /**
   * Returns an enumeration describing the available options.
   * 
   * @return an enumeration of all the available options
   */
  @Override
  public Enumeration<Option> listOptions() {
    Vector<Option> result = enumToVector(super.listOptions());

    result.addElement(new Option("\tA cluster definition of class '"
      + SubspaceClusterDefinition.class.getName().replaceAll(".*\\.", "")
      + "'\n" + "\t(definition needs to be quoted to be recognized as \n"
      + "\ta single argument).", "C", 1, "-C <cluster-definition>"));

    result.addElement(new Option("", "", 0, "\nOptions specific to "
      + SubspaceClusterDefinition.class.getName() + ":"));

    result.addElement(new Option("\tThe indices for boolean attributes.", "b",
            1, "-b <range>"));

    result.addElement(new Option("\tThe indices for nominal attributes.", "m",
            1, "-m <range>"));

    result.addAll(enumToVector(new SubspaceClusterDefinition(this).listOptions()));

    return result.elements();
  }

  /**
   * Parses a list of options for this object.
   * <p/>
   * 
   * <!-- options-start --> Valid options are:
   * <p/>
   * 
   * <pre>
   * -h
   *  Prints this help.
   * </pre>
   * 
   * <pre>
   * -o &lt;file&gt;
   *  The name of the output file, otherwise the generated data is
   *  printed to stdout.
   * </pre>
   * 
   * <pre>
   * -r &lt;name&gt;
   *  The name of the relation.
   * </pre>
   * 
   * <pre>
   * -d
   *  Whether to print debug informations.
   * </pre>
   * 
   * <pre>
   * -S
   *  The seed for random function (default 1)
   * </pre>
   * 
   * <pre>
   * -a &lt;num&gt;
   *  The number of attributes (default 1).
   * </pre>
   * 
   * <pre>
   * -c
   *  Class Flag, if set, the cluster is listed in extra attribute.
   * </pre>
   * 
   * <pre>
   * -b &lt;range&gt;
   *  The indices for boolean attributes.
   * </pre>
   * 
   * <pre>
   * -m &lt;range&gt;
   *  The indices for nominal attributes.
   * </pre>
   *
   * <pre>
   * -C &lt;cluster-definition&gt;
   *  A cluster definition of class 'SubspaceClusterDefinition'
   *  (definition needs to be quoted to be recognized as 
   *  a single argument).
   * </pre>
   * 
   * <pre>
   * Options specific to weka.datagenerators.clusterers.SubspaceClusterDefinition:
   * </pre>
   * 
   * <pre>
   * -A &lt;range&gt;
   *  Uses a random uniform distribution for the instances in the cluster.
   * </pre>
   *
   * <pre>
   * -U &lt;range&gt;
   *  Generates totally uniformly distributed instances in the cluster.
   * </pre>
   *
   * <pre>
   * -G &lt;range&gt;
   *  Uses a Gaussian distribution for the instances in the cluster.
   * </pre>
   * 
   * <pre>
   * -D &lt;num&gt;,&lt;num&gt;
   *  The attribute min/max (-A and -U) or mean/stddev (-G) for
   *  the cluster.
   * </pre>
   * 
   * <pre>
   * -N &lt;num&gt;..&lt;num&gt;
   *  The range of number of instances per cluster (default 1..50).
   * </pre>
   * 
   * <pre>
   * -I
   *  Uses integer instead of continuous values (default continuous).
   * </pre>
   * 
   * <!-- options-end -->
   * 
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {
    super.setOptions(options);

    String tmpStr = Utils.getOption('b', options);
    setBooleanIndices(tmpStr);
    m_booleanCols.setUpper(getNumAttributes() - 1);

    tmpStr = Utils.getOption('m', options);
    setNominalIndices(tmpStr);
    m_nominalCols.setUpper(getNumAttributes() - 1);

    // cluster definitions
    Vector<SubspaceClusterDefinition> list = new Vector<SubspaceClusterDefinition>();
    do {
      tmpStr = Utils.getOption('C', options);
      if (tmpStr.length() != 0) {
        SubspaceClusterDefinition cl = new SubspaceClusterDefinition(this);
        cl.setOptions(Utils.splitOptions(tmpStr));
        list.add(cl);
      }
    } while (tmpStr.length() != 0);

    // If list is empty, add default cluster definition, to be consistent with
    // initialisation of member variables in this class and to make generator work when
    // run from command-line without any explicit cluster specifications.
    if (list.size() == 0) {
      list.add(new SubspaceClusterDefinition(this));
    }

    m_Clusters = list.toArray(new ClusterDefinition[list.size()]);
   }

  /**
   * Sets which attributes are boolean
   *
   * @param rangeList a string representing the list of attributes. Since the
   *          string will typically come from a user, attributes are indexed
   *          from 1. <br/>
   *          eg: first-3,5,6-last
   * @throws IllegalArgumentException if an invalid range list is supplied
   */
  public void setBooleanIndices(String rangeList) {
    m_booleanCols.setRanges(rangeList);
  }

  /**
   * Sets which attributes are boolean.
   *
   * @param value the range to use
   */
  public void setBooleanCols(Range value) {
    m_booleanCols.setRanges(value.getRanges());
  }

  /**
   * returns the range of boolean attributes.
   *
   * @return the range of boolean attributes
   */
  public Range getBooleanCols() {
    return m_booleanCols;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String booleanColsTipText() {
    return "The range of attributes that are generated as boolean ones.";
  }

  /**
   * Sets which attributes are nominal
   *
   * @param rangeList a string representing the list of attributes. Since the
   *          string will typically come from a user, attributes are indexed
   *          from 1. <br/>
   *          eg: first-3,5,6-last
   * @throws IllegalArgumentException if an invalid range list is supplied
   */
  public void setNominalIndices(String rangeList) {
    m_nominalCols.setRanges(rangeList);
  }

  /**
   * Sets which attributes are nominal.
   *
   * @param value the range to use
   */
  public void setNominalCols(Range value) {
    m_nominalCols.setRanges(value.getRanges());
  }

  /**
   * returns the range of nominal attributes
   *
   * @return the range of nominal attributes
   */
  public Range getNominalCols() {

    return m_nominalCols;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String nominalColsTipText() {
    return "The range of attributes to generate as nominal ones.";
  }

  /**
   * check if attribute types are not contradicting
   *
   * @return empty string if no problem, otherwise error message
   */
  protected String checkIndices() {
    for (int i = 0; i < getNumAttributes(); i++) {
      if (m_booleanCols.isInRange(i) && m_nominalCols.isInRange(i)) {
        return "Error in attribute type: Attribute " + i + " is set to both boolean and nominal.";
      }
    }
    return "";
  }

  /**
   * Gets the current settings of the datagenerator.
   * 
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String[] getOptions() {

    Vector<String> result = new Vector<String>();

    Collections.addAll(result, super.getOptions());

    if (!getBooleanCols().toString().equalsIgnoreCase("empty")) {
      result.add("-b");
      result.add("" + getBooleanCols().getRanges());
    }

    if (!getNominalCols().toString().equalsIgnoreCase("empty")) {
      result.add("-m");
      result.add("" + getNominalCols().getRanges());
    }

    for (int i = 0; i < getClusters().length; i++) {
      result.add("-C");
      result.add(Utils.joinOptions(getClusters()[i].getOptions()));
    }

    return result.toArray(new String[result.size()]);
  }

  /**
   * returns the current cluster definitions, if necessary initializes them
   * 
   * @return the current cluster definitions
   */
  protected ClusterDefinition[] getClusters() {

    return m_Clusters;
  }

  /**
   * returns the default number of attributes
   * 
   * @return the default number of attributes
   */
  @Override
  protected int defaultNumAttributes() {
    return 1;
  }

  /**
   * returns the currently set clusters
   * 
   * @return the currently set clusters
   */
  public ClusterDefinition[] getClusterDefinitions() {
    return getClusters();
  }

  /**
   * sets the clusters to use
   * 
   * @param value the clusters do use
   * @throws Exception if clusters are not the correct class
   */
  public void setClusterDefinitions(ClusterDefinition[] value) throws Exception {

    String indexStr;

    indexStr = "";
    m_Clusters = value;
    for (int i = 0; i < getClusters().length; i++) {
      if (!(getClusters()[i] instanceof SubspaceClusterDefinition)) {
        if (indexStr.length() != 0) {
          indexStr += ",";
        }
        indexStr += "" + (i + 1);
      }
      getClusters()[i].setParent(this);
     }

    // any wrong classes encountered?
    if (indexStr.length() != 0) {
      throw new Exception("These cluster definitions are not '"
        + SubspaceClusterDefinition.class.getName() + "': " + indexStr);
    }
  }

  /**
   * Returns the tip text for this property
   * 
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String clusterDefinitionsTipText() {
    return "The clusters to use.";
  }

  /**
   * Checks, whether all attributes are covered by cluster definitions and
   * returns TRUE in that case.
   * 
   * @return whether all attributes are covered
   */
  protected boolean checkCoverage() {
    int i;
    int n;
    int[] count;
    Range r;
    String attrIndex;
    SubspaceClusterDefinition cl;

    // check whether all the attributes are covered
    count = new int[getNumAttributes()];
    for (i = 0; i < getNumAttributes(); i++) {
      if (m_nominalCols.isInRange(i)) {
        count[i]++;
      }
      if (m_booleanCols.isInRange(i)) {
        count[i]++;
      }
      for (n = 0; n < getClusters().length; n++) {
        cl = (SubspaceClusterDefinition) getClusters()[n];
        r = new Range(cl.getAttrIndexRange());
        r.setUpper(getNumAttributes());
        if (r.isInRange(i)) {
          count[i]++;
        }
      }
    }

    // list all indices that are not covered
    attrIndex = "";
    for (i = 0; i < count.length; i++) {
      if (count[i] == 0) {
        if (attrIndex.length() != 0) {
          attrIndex += ",";
        }
        attrIndex += (i + 1);
      }
    }

    if (attrIndex.length() != 0) {
      throw new IllegalArgumentException(
        "The following attributes are not covered by a cluster " + "definition: " + attrIndex + "\n");
    }

    return true;
  }

  /**
   * Gets the single mode flag.
   * 
   * @return true if methode generateExample can be used.
   */
  @Override
  public boolean getSingleModeFlag() {
    return false;
  }

  /**
   * Initializes the format for the dataset produced.
   * 
   * @return the output data format
   * @throws Exception data format could not be defined
   */

  @Override
  public Instances defineDataFormat() throws Exception {

    getBooleanCols().setUpper(getNumAttributes());
    getNominalCols().setUpper(getNumAttributes());

    // check indices
    String tmpStr = checkIndices();
    if (tmpStr.length() > 0) {
      throw new IllegalArgumentException(tmpStr);
    }

    checkCoverage();

    m_numValues = new int[getNumAttributes()];

    Random random = new Random(getSeed());
    setRandom(random);

    for (int i = 0; i < getClusters().length; i++) {
      SubspaceClusterDefinition cl = (SubspaceClusterDefinition) getClusters()[i];
      cl.setNumInstances(random);
      cl.setParent(this);
      cl.initialiseMemberVariables();
      cl.setValues();
    }

    ArrayList<Attribute> attributes = new ArrayList<Attribute>(3);

    // define dataset
    for (int i = 0; i < getNumAttributes(); i++) {
      // define boolean attribute
      if (m_booleanCols.isInRange(i)) {
        ArrayList<String> boolValues = new ArrayList<String>(2);
        boolValues.add("false");
        boolValues.add("true");
        attributes.add(new Attribute("B" + i, boolValues));
      } else if (m_nominalCols.isInRange(i)) {
        // define nominal attribute
        ArrayList<String> nomValues = new ArrayList<String>(m_numValues[i]);
        for (int j = 0; j < m_numValues[i]; j++) {
          nomValues.add("value-" + j);
        }
        attributes.add(new Attribute("N" + i, nomValues));
      } else {
        // numerical attribute
        attributes.add(new Attribute("X" + i));
      }
    }

    if (getClassFlag()) {
      ArrayList<String> classValues = new ArrayList<String>(getClusters().length);
      for (int i = 0; i < getClusters().length; i++) {
        classValues.add("c" + i);
      }
      attributes.add(new Attribute("class", classValues));
    }

    Instances dataset = new Instances(getRelationNameToUse(), attributes, 0);
    if (getClassFlag()) {
      dataset.setClassIndex(m_NumAttributes);
    }

    // set dataset format of this class
    setDatasetFormat(new Instances(dataset, 0));

    return dataset;
  }

  /**
   * Returns true if attribute is boolean
   * 
   * @param index of the attribute
   * @return true if the attribute is boolean
   */
  public boolean isBoolean(int index) {
    return m_booleanCols.isInRange(index);
  }

  /**
   * Returns true if attribute is nominal
   * 
   * @param index of the attribute
   * @return true if the attribute is nominal
   */
  public boolean isNominal(int index) {
    return m_nominalCols.isInRange(index);
  }

  /**
   * returns array that stores the number of values for a nominal attribute.
   * 
   * @return the array that stores the number of values for a nominal attribute
   */
  public int[] getNumValues() {
    return m_numValues;
  }

  /**
   * Generate an example of the dataset.
   * 
   * @return the instance generated
   * @throws Exception if format not defined or generating <br/>
   *           examples one by one is not possible, because voting is chosen
   */

  @Override
  public Instance generateExample() throws Exception {
    throw new Exception("Examples cannot be generated one by one.");
  }

  /**
   * Generate all examples of the dataset.
   * 
   * @return the instance generated
   * @throws Exception if format not defined
   */

  @Override
  public Instances generateExamples() throws Exception {
    Instances format = getDatasetFormat();
    Instance example = null;

    if (format == null) {
      throw new Exception("Dataset format not defined.");
    }

    // generate examples for one cluster after another
    for (int cNum = 0; cNum < getClusters().length; cNum++) {
      SubspaceClusterDefinition cl = (SubspaceClusterDefinition) getClusters()[cNum];

      // get the number of instances to create
      int instNum = cl.getNumInstances();

      // class value is c + cluster number
      String cName = "c" + cNum;

      switch (cl.getClusterType().getSelectedTag().getID()) {
      case (UNIFORM_RANDOM):
        for (int i = 0; i < instNum; i++) {
          // generate example
          example = generateExample(format, getRandom(), cl, cName);
          if (example != null) {
            format.add(example);
          }
        }
        break;
      case (TOTAL_UNIFORM):
        // generate examples
        if (!cl.isInteger()) {
          generateUniformExamples(format, instNum, cl, cName);
        } else {
          generateUniformIntegerExamples(format, instNum, cl, cName);
        }
        break;
      case (GAUSSIAN):
        // generate examples
        generateGaussianExamples(format, instNum, getRandom(), cl, cName);
        break;
      }
    }

    return format;
  }

  /**
   * Generate an example of the dataset.
   * 
   * @param format the dataset format
   * @param randomG the random number generator to use
   * @param cl the cluster definition
   * @param cName the class value
   * @return the generated instance
   */
  private Instance generateExample(Instances format, Random randomG, SubspaceClusterDefinition cl, String cName) {

    boolean makeInteger = cl.isInteger();
    int num = -1;
    int numAtts = m_NumAttributes;
    if (getClassFlag()) {
      numAtts++;
    }

    double[] values = new double[numAtts];
    boolean[] attributes = cl.getAttributes();
    double[] minValue = cl.getMinValue();
    double[] maxValue = cl.getMaxValue();
    double value;

    int clusterI = -1;
    for (int i = 0; i < m_NumAttributes; i++) {
      if (attributes[i]) {
        clusterI++;
        num++;
        // boolean or nominal attribute
        if (isBoolean(i) || isNominal(i)) {
          if (minValue[clusterI] == maxValue[clusterI]) {
            value = minValue[clusterI];
          } else {
            int numValues = (int) (maxValue[clusterI] - minValue[clusterI] + 1.0);
            value = randomG.nextInt(numValues);
            value += minValue[clusterI];
          }
        } else {
          // numeric attribute
          value = randomG.nextDouble() * (maxValue[num] - minValue[num]) + minValue[num];
          if (makeInteger) {
            value = Math.round(value);
          }
        }
        values[i] = value;
      } else {
        values[i] = Utils.missingValue();
      }
    }

    if (getClassFlag()) {
      values[format.classIndex()] = format.classAttribute().indexOfValue(cName);
    }

    DenseInstance example = new DenseInstance(1.0, values);
    example.setDataset(format);

    return example;
  }

  /**
   * Generate examples for a uniform cluster dataset.
   * 
   * @param format the dataset format
   * @param numInstances the number of instances to generator
   * @param cl the cluster definition
   * @param cName the class value
   */
  private void generateUniformExamples(Instances format, int numInstances,
    SubspaceClusterDefinition cl, String cName) {

    int numAtts = m_NumAttributes;
    if (getClassFlag()) {
      numAtts++;
    }
    boolean[] attributes = cl.getAttributes();
    double[] minValue = cl.getMinValue();
    double[] maxValue = cl.getMaxValue();
    double[] diff = new double[minValue.length];

    for (int i = 0; i < minValue.length; i++) {
      diff[i] = (maxValue[i] - minValue[i]);
    }

    int numStepsPerDimension = (int)Math.rint(Math.pow(numInstances, 1.0 / minValue.length));

    int[] countPerDimension = new int[minValue.length];

    for (int j = 0; j < numInstances; j++) {
      double[] values = new double[numAtts];
      int num = -1;
      for (int i = 0; i < m_NumAttributes; i++) {
        if (attributes[i]) {
          num++;
          values[i] = minValue[num] +
                  (diff[num] * ((double) (countPerDimension[num]) / (double) (numStepsPerDimension - 1)));
        } else {
          values[i] = Utils.missingValue();
        }
      }
      if (getClassFlag()) {
        values[format.classIndex()] = format.classAttribute().indexOfValue(cName);
      }

      DenseInstance example = new DenseInstance(1.0, values);
      example.setDataset(format);
      format.add(example);

      countPerDimension[0]++;
      for (int i = 0; i < minValue.length; i++) {
        if (countPerDimension[i] == numStepsPerDimension) {
          countPerDimension[i] = 0;
          if (i + 1 < minValue.length) {
            countPerDimension[i + 1]++;
          }
        }
      }
    }
  }

  /**
   * Generate examples for a uniform cluster dataset.
   * 
   * @param format the dataset format
   * @param numInstances the number of instances to generator
   * @param cl the cluster definition
   * @param cName the class value
   */
  private void generateUniformIntegerExamples(Instances format,
    int numInstances, SubspaceClusterDefinition cl, String cName) {

    double[] values = new double[getClassFlag() ? m_NumAttributes + 1 : m_NumAttributes];

    int[] minInt = new int[m_NumAttributes];
    int[] maxInt = new int[m_NumAttributes];

    int[] indices = new int[cl.getMaxValue().length];

    int num = 1;
    int index = 0;
    for (int i = 0; i < m_NumAttributes; i++) {
      if (cl.getAttributes()[i]) {
        minInt[i] = (int) Math.ceil(cl.getMinValue()[index]);
        maxInt[i] = (int) Math.floor(cl.getMaxValue()[index]);
        num *= (maxInt[i] - minInt[i] + 1);
        indices[index++] = i;
      }
    }
    int numEach = numInstances / num;
    int rest = numInstances - numEach * num;

    // initialize with smallest values combination
    for (int i = 0; i < m_NumAttributes; i++) {
      if (cl.getAttributes()[i]) {
        values[i] = minInt[i];
      } else {
        values[i] = Utils.missingValue();
      }
    }
    if (getClassFlag()) {
      values[format.classIndex()] = format.classAttribute().indexOfValue(cName);
    }

    int added = 0;
    while (added < numInstances) {
      DenseInstance example = new DenseInstance(1.0, values);
      // add all for one value combination
      for (int k = 0; k < numEach; k++) {
        format.add(example); // Instance will be copied here
        added++;
      }
      if (rest > 0) {
        format.add(example); // Instance will be copied here
        added++;
        rest--;
      }
      // switch to the next value combination
      values = example.toDoubleArray();
      values[indices[0]]++;
      for (int i = 0; i < indices.length; i++) {
        if (values[indices[i]] > maxInt[indices[i]]) {
          values[indices[i]] = minInt[indices[i]];
          if (i + 1 < indices.length) {
            values[indices[i + 1]]++;
          }
        }
      }
    }
  }

  /**
   * Generate examples for a uniform cluster dataset.
   * 
   * @param format the dataset format
   * @param numInstances the number of instances to generate
   * @param random the random number generator
   * @param cl the cluster definition
   * @param cName the class value
   */
  private void generateGaussianExamples(Instances format, int numInstances,
    Random random, SubspaceClusterDefinition cl, String cName) {

    boolean makeInteger = cl.isInteger();
    int numAtts = m_NumAttributes;
    if (getClassFlag()) {
      numAtts++;
    }

    boolean[] attributes = cl.getAttributes();
    double[] meanValue = cl.getMeanValue();
    double[] stddevValue = cl.getStddevValue();

    for (int j = 0; j < numInstances; j++) {
      double[] values = new double[numAtts];
      int num = -1;
      for (int i = 0; i < m_NumAttributes; i++) {
        if (attributes[i]) {
          num++;
          double value = meanValue[num] + (random.nextGaussian() * stddevValue[num]);
          if (makeInteger) {
            value = Math.round(value);
          }
          values[i] = value;
        } else {
          values[i] = Utils.missingValue();
        }
      }
      if (getClassFlag()) {
        values[format.classIndex()] = format.classAttribute().indexOfValue(cName);
      }

      DenseInstance example = new DenseInstance(1.0, values);
      example.setDataset(format);
      format.add(example);
    }
  }

  /**
   * Compiles documentation about the data generation after the generation
   * process
   * 
   * @return string with additional information about generated dataset
   * @throws Exception no input structure has been defined
   */
  @Override
  public String generateFinished() throws Exception {
    return "";
  }

  /**
   * Compiles documentation about the data generation before the generation
   * process
   * 
   * @return string with additional information
   */
  @Override
  public String generateStart() {
    StringBuffer docu = new StringBuffer();

    int sumInst = 0;
    for (int cNum = 0; cNum < getClusters().length; cNum++) {
      SubspaceClusterDefinition cl = (SubspaceClusterDefinition) getClusters()[cNum];
      docu.append("%\n");
      docu.append("% Cluster: c" + cNum + "   ");
      switch (cl.getClusterType().getSelectedTag().getID()) {
      case UNIFORM_RANDOM:
        docu.append("Uniform Random");
        break;
      case TOTAL_UNIFORM:
        docu.append("Total Random");
        break;
      case GAUSSIAN:
        docu.append("Gaussian");
        break;
      }
      if (cl.isInteger()) {
        docu.append(" / INTEGER");
      }

      docu.append("\n% ----------------------------------------------\n");
      docu.append("%" + cl.attributesToString());

      docu.append("\n% Number of Instances:            " + cl.getInstNums()
        + "\n");
      docu.append("% Generated Number of Instances:  " + cl.getNumInstances()
        + "\n");
      sumInst += cl.getNumInstances();
    }
    docu.append("%\n% ----------------------------------------------\n");
    docu.append("% Total Number of Instances: " + sumInst + "\n");
    docu.append("%                            in " + getClusters().length
      + " Cluster(s)\n%");

    return docu.toString();
  }

  /**
   * Returns the revision string.
   * 
   * @return the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision$");
  }

  /**
   * Main method for testing this class.
   * 
   * @param args should contain arguments for the data producer:
   */
  public static void main(String[] args) {
    runDataGenerator(new SubspaceCluster(), args);
  }
}
