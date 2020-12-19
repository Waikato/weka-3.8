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
 * ClusterGenerator.java
 * Copyright (C) 2000-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.datagenerators;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.core.Option;
import weka.core.Range;
import weka.core.Utils;

/**
 * Abstract class for cluster data generators.
 * <p/>
 * 
 * Example usage as the main of a datagenerator called RandomGenerator:
 * 
 * <pre>
 * public static void main(String[] args) {
 *   try {
 *     DataGenerator.makeData(new RandomGenerator(), args);
 *   } catch (Exception e) {
 *     e.printStackTrace();
 *     System.err.println(e.getMessage());
 *   }
 * }
 * </pre>
 * <p/>
 * 
 * @author Gabi Schmidberger (gabi@cs.waikato.ac.nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 */
public abstract class ClusterGenerator extends DataGenerator {

  /** for serialization */
  private static final long serialVersionUID = 6131722618472046365L;

  /** Number of attribute the dataset should have */
  protected int m_NumAttributes;

  /** class flag */
  protected boolean m_ClassFlag = false;

  /**
   * initializes the generator
   */
  public ClusterGenerator() {
    super();

    setNumAttributes(defaultNumAttributes());
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {
    Vector<Option> result = enumToVector(super.listOptions());

    result.addElement(new Option("\tThe number of attributes (default "
      + defaultNumAttributes() + ").", "a", 1, "-a <num>"));

    result.addElement(new Option(
      "\tClass Flag, if set, the cluster is listed in extra attribute.", "c",
      0, "-c"));

    return result.elements();
  }

  /**
   * Sets the options.
   *
   * @param options the options
   * @throws Exception if invalid option
   */
  @Override
  public void setOptions(String[] options) throws Exception {
    String tmpStr;

    super.setOptions(options);

    tmpStr = Utils.getOption('a', options);
    if (tmpStr.length() != 0) {
      setNumAttributes(Integer.parseInt(tmpStr));
    } else {
      setNumAttributes(defaultNumAttributes());
    }

    setClassFlag(Utils.getFlag('c', options));
  }

  /**
   * Gets the current settings of the classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String[] getOptions() {

    Vector<String> result = new Vector<String>();

    Collections.addAll(result, super.getOptions());

    result.add("-a");
    result.add("" + getNumAttributes());

    if (getClassFlag()) {
      result.add("-c");
    }

    return result.toArray(new String[result.size()]);
  }

  /**
   * returns the default number of attributes
   *
   * @return the default number of attributes
   */
  protected int defaultNumAttributes() {
    return 10;
  }

  /**
   * Sets the number of attributes the dataset should have.
   *
   * @param numAttributes the new number of attributes
   */
  public void setNumAttributes(int numAttributes) {
    m_NumAttributes = numAttributes;
   }

  /**
   * Gets the number of attributes that should be produced.
   *
   * @return the number of attributes that should be produced
   */
  public int getNumAttributes() {
    return m_NumAttributes;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String numAttributesTipText() {
    return "The number of attributes the generated data will contain.";
  }

  /**
   * Sets the class flag, if class flag is set, the cluster is listed as class
   * atrribute in an extra attribute.
   *
   * @param classFlag the new class flag
   */
  public void setClassFlag(boolean classFlag) {
    m_ClassFlag = classFlag;
  }

  /**
   * Gets the class flag.
   *
   * @return the class flag
   */
  public boolean getClassFlag() {
    return m_ClassFlag;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String classFlagTipText() {
    return "If set to TRUE, lists the cluster as an extra attribute.";
  }
}
