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

package weka.filters.unsupervised.attribute;

import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.AbstractFilterTest;
import weka.filters.Filter;

import junit.framework.Test;
import junit.framework.TestSuite;
import junit.textui.TestRunner;

public class FourierTransformTest extends AbstractFilterTest {

  public FourierTransformTest(String name) {
    super(name);
  }

  @Override
  public Filter getFilter() {
    return new FourierTransform();
  }

  @Override protected void setUp() throws Exception {
    super.setUp();

    m_Instances.deleteAttributeType(Attribute.STRING);
    m_Instances.deleteAttributeType(Attribute.RELATIONAL);
    m_Instances.deleteAttributeType(Attribute.NOMINAL);
    m_Instances.deleteAttributeType(Attribute.DATE);

    // class index
    m_Instances.setClassIndex(1);
    m_FilteredClassifier = null;
  }

  protected void performTest() {
    Instances icopy = new Instances(m_Instances);
    int numToExpect = m_Instances.numInstances() / ((FourierTransform) m_Filter).getSequenceLength();

    Instances result = null;
    try {
      m_Filter.setInputFormat(icopy);
    } catch (Exception ex) {
      ex.printStackTrace();
      fail("Exception thrown on setInputFormat(): \n" + ex.getMessage());
    }
    try {
      result = Filter.useFilter(icopy, m_Filter);
      assertNotNull(result);
    } catch (Exception ex) {
      ex.printStackTrace();
      fail("Exception thrown on useFilter(): \n" + ex.getMessage());
    }

    assertTrue(result.numInstances() == numToExpect ||
      result.numInstances() == numToExpect + 1);
  }

  public void testTypical() {
    m_Filter = getFilter();
    performTest();
  }

  /**
   * Returns a configures test suite.
   * 
   * @return a configured test suite
   */
  public static Test suite() {
    return new TestSuite(FourierTransformTest.class);
  }

  /**
   * For running the test from commandline.
   * 
   * @param args ignored
   */
  public static void main(String[] args) {
    TestRunner.run(suite());
  }
}
