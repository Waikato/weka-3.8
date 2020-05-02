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
 *    NormalizedPolyKernel.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.functions.supportVector;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;

/**
 <!-- globalinfo-start -->
 * The normalized polynomial kernel.<br/>
 * K(x,y) = &lt;x,y&gt;/sqrt(&lt;x,x&gt;&lt;y,y&gt;) where &lt;x,y&gt; = PolyKernel(x,y)
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -D
 *  Enables debugging output (if available) to be printed.
 *  (default: off)</pre>
 *
 * <pre> -C &lt;num&gt;
 *  The size of the cache (a prime number), 0 for full cache and 
 *  -1 to turn it off.
 *  (default: 250007)</pre>
 * 
 * <pre> -E &lt;num&gt;
 *  The Exponent to use.
 *  (default: 1.0)</pre>
 * 
 * <pre> -L
 *  Use lower-order terms.
 *  (default: no)</pre>
 * 
 <!-- options-end -->
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class NormalizedPolyKernel 
  extends PolyKernel {

  /** for serialization */
  static final long serialVersionUID = 1248574185532130851L;

  /** A cache for the diagonal of the dot product kernel */
  protected double[] m_diagDotproducts = null;

  /**
   * default constructor - does nothing
   */
  public NormalizedPolyKernel() {
    super();
  }
  
  /**
   * Creates a new <code>NormalizedPolyKernel</code> instance.
   *
   * @param dataset	the training dataset used.
   * @param cacheSize	the size of the cache (a prime number)
   * @param exponent	the exponent to use
   * @param lowerOrder	whether to use lower-order terms
   * @throws Exception	if something goes wrong
   */
  public NormalizedPolyKernel(Instances dataset, int cacheSize, 
      double exponent, boolean lowerOrder) throws Exception {
	
    super(dataset, cacheSize, exponent, lowerOrder);
  }

  /**
   * builds the kernel with the given data. Initializes the kernel cache. The
   * actual size of the cache in bytes is (64 * cacheSize).
   *
   * @param data the data to base the kernel on
   * @throws Exception if something goes wrong
   */
  @Override
  public void buildKernel(Instances data) throws Exception {

    super.buildKernel(data);
    m_diagDotproducts = new double[data.numInstances()];
    for (int i = 0; i < data.numInstances(); i++) {
      m_diagDotproducts[i] = dotProd(m_data.instance(i), m_data.instance(i));
    }
  }

  /**
   * Frees the cache used by the kernel.
   */
  @Override
  public void clean() {
    super.clean();
    m_diagDotproducts = null;
  }
  /**
   * Returns a string describing the kernel
   * 
   * @return a description suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String globalInfo() {
    return 
        "The normalized polynomial kernel.\n"
      + "K(x,y) = <x,y>/sqrt(<x,x><y,y>) where <x,y> = PolyKernel(x,y)";
  }

  /**
   *
   * @param id1 the index of instance 1
   * @param id2 the index of instance 2
   * @param inst1 the instance 1 object
   * @return the dot product
   * @throws Exception if something goes wrong
   */
  @Override
  protected double evaluate(int id1, int id2, Instance inst1) throws Exception {

    double result;
    if (id1 == id2) {
      return 1.0;
    } else {
      double numerator = dotProd(inst1, m_data.instance(id2));

      double denom1, denom2;
      if (m_diagDotproducts != null) {
        denom2 = m_diagDotproducts[id2];
        if (id1 < 0) {
          denom1 = dotProd(inst1, inst1);
        } else {
          denom1 = m_diagDotproducts[id1];
        }
      } else {
        denom1 = dotProd(inst1, inst1);
        denom2 = dotProd(m_data.instance(id2), m_data.instance(id2));
      }

      // Use lower order terms?
      if (m_lowerOrder) {
        numerator += 1.0;
        denom1 += 1.0;
        denom2 += 1.0;
      }
      double denominatorSquared = denom1 * denom2;
      if (denominatorSquared <= 0) {
        result = 0;
      } else {
        result = numerator / Math.sqrt(denominatorSquared);
      }
    }

    if (m_exponent != 1.0) {
      result = Math.pow(result, m_exponent);
    }
    return result;
  }
  
  /**
   * returns a string representation for the Kernel
   * 
   * @return 		a string representaiton of the kernel
   */
  public String toString() {
    String	result;
    
    if (getUseLowerOrder())
      result = "Normalized Poly Kernel with lower order: K(x,y) = (<x,y>+1)^" + getExponent() + "/" + 
      	       "((<x,x>+1)^" + getExponent() + "*" + "(<y,y>+1)^" + getExponent() + ")^(1/2)";
    else
      result = "Normalized Poly Kernel: K(x,y) = <x,y>^" + getExponent() + "/" + "(<x,x>^" + 
               getExponent() + "*" + "<y,y>^" + getExponent() + ")^(1/2)";
    
    return result;
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision$");
  }
}

