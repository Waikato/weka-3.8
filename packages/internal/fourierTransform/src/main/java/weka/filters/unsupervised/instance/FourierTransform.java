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
 *    FourierTransform.java
 *    Copyright (C) 2019 University of Waikato, Hamilton, New Zealand
 *    Original code by Anilkumar Patro at Worcester Polytechnic Institute
 *
 */

package weka.filters.unsupervised.instance;

import weka.core.*;
import weka.experiment.Stats;
import weka.filters.Filter;
import weka.filters.UnsupervisedFilter;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Vector;

/**
 * <!-- globalinfo-start --> Fourier Transform filter that computes FFT
 * magnitude spectrum over numeric sequences stored in consecutive attribute
 * values - i.e. each instance is expected to contain a sequence to be
 * transformed. Output consists of one attribute for each entry in the spectrum.
 * Provides an option to reduce dimensionality by binning the spectrum. In this
 * case, the value in a given output bin is the sum of the FFT magnitude values
 * that fall into that bin. <br>
 * <br>
 * <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 * 
 * <pre>
 * -width &lt;integer&gt;
 *  FFT bin width (default = 1).
 * </pre>
 * 
 * <pre>
 * -bin-name &lt;string&gt;
 *  Prefix for the bin names (default = 'bin').
 * </pre>
 * 
 * <pre>
 * -normalize
 *  Normalize FFT magnitude.
 * </pre>
 * 
 * <pre>
 * -keep
 *  Keep original numeric fields in output
 * </pre>
 * 
 * <pre>
 * -summary
 *  Add time-based summary attributes.
 * </pre>
 * 
 * <!-- options-end -->
 *
 * @author anilkpatro
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 */
public class FourierTransform extends Filter implements UnsupervisedFilter,
  OptionHandler {

  private static final long serialVersionUID = -1042901262886974910L;

  /**
   * Complex number helper for FTs
   */
  private AttFTHolder[] m_attsFT;

  /** Spectrum bin width */
  protected int m_binWidth = 1;

  /** Whether to add some time-based summary statistics for each sequence */
  protected boolean m_addTimeSummaryStats;

  /** Bin name prefix */
  protected String m_binNamePrefix = "bin";

  /** Whether to keep the original numeric fields in the output too */
  protected boolean m_keepOriginalNumerics;

  /** Whether to normalize the fft magnitude values */
  protected boolean m_normalizeMagnitude;

  /** Holds the computed number of bins */
  protected int m_numBins;

  /** Holds all the other attributes */
  protected List<Attribute> m_notConverted = new ArrayList<>();

  /** Class index (if set) */
  protected int m_classIndex = -1;

  /** Number of numeric attributes in the data */
  protected int m_numNumeric;

  /**
   * Set the bin width
   *
   * @param binSize the width of the bin
   */
  /**
   * Set the bin width
   *
   * @param binSize the bin width
   */
  @OptionMetadata(displayName = "Bin width",
    description = "Bin width for FFT magnitudes (default = 1, i.e. no binning). "
      + "Within each bin FFT magnitude values are summed.",
    commandLineParamName = "width",
    commandLineParamSynopsis = "-width <integer>", displayOrder = 1)
  public void setBinWidth(int binSize) {
    m_binWidth = binSize;
  }

  /**
   * Get the bin width
   *
   * @return the width of the bin
   */
  public int getBinWidth() {
    return m_binWidth;
  }

  /**
   * set whether to add time-based summary attributes
   *
   * @param addTimeSummaryStats true to add summary attributes
   */
  @OptionMetadata(
    displayName = "Add time-based summary stats",
    description = "Add min, max, mean, stdev of the original values in each sequence "
      + "as additional attributes.", commandLineParamSynopsis = "-summary",
    commandLineParamIsFlag = true, displayOrder = 4)
  public
    void setAddTimeSummaryStats(boolean addTimeSummaryStats) {
    m_addTimeSummaryStats = addTimeSummaryStats;
  }

  /**
   * Get whether to add time-based summary attributes
   *
   * @return true if summary attributes will be added
   */
  public boolean getAddTimeSummaryStats() {
    return m_addTimeSummaryStats;
  }

  /**
   * Set whether to keep original numeric values in the output
   *
   * @param keepOriginalNumerics true to keep original values in the output
   */
  @OptionMetadata(
    displayName = "Keep original numeric values",
    description = "Keep the original numeric sequence values in the output (in "
      + "addition to the FFT transformed versions",
    commandLineParamName = "keep", commandLineParamSynopsis = "-keep",
    commandLineParamIsFlag = true, displayOrder = 5)
  public
    void setKeepOriginalNumerics(boolean keepOriginalNumerics) {
    m_keepOriginalNumerics = keepOriginalNumerics;
  }

  /**
   * Get whether to keep original numeric values in the output
   *
   * @return true if original numeric values will be kept in the output
   */
  public boolean getKeepOriginalNumerics() {
    return m_keepOriginalNumerics;
  }

  /**
   * Set whether to normalize magnitude values by by N * 2
   *
   * @param normalizeMagnitude true if magnitude is to be normalized
   */
  /**
   * Set whether to normalize magnitude values by by N * 2
   *
   * @param normalizeMagnitude true if magnitude is to be normalized
   */
  @OptionMetadata(
    displayName = "Normalize FFT magnitude values",
    description = "Normalized magnitude values by N * 2. Scaled by 2 as we only "
      + "use half the (symmetric) FFT spectrum",
    commandLineParamName = "normalize",
    commandLineParamSynopsis = "-normalize", commandLineParamIsFlag = true,
    displayOrder = 3)
  public
    void setNormalizeMagnitude(boolean normalizeMagnitude) {
    m_normalizeMagnitude = normalizeMagnitude;
  }

  /**
   * Get whether to normalize magnitude values by N * 2
   *
   * @return true if magnitude is to be normalized
   */
  public boolean getNormalizeMagnitude() {
    return m_normalizeMagnitude;
  }

  /**
   * Set a name prefix for the bin attributes
   *
   * @param binNamePrefix name prefix to use
   */
  @OptionMetadata(displayName = "Bin name prefix",
    description = "Prefix for the binned FFT attributes (default = 'bin')",
    commandLineParamName = "bin-name",
    commandLineParamSynopsis = "-bin-name <string>", displayOrder = 2)
  public void setBinNamePrefix(String binNamePrefix) {
    m_binNamePrefix = binNamePrefix;
  }

  /**
   * Get the name prefix for the bin attributes
   *
   * @return the name prefix to use
   */
  public String getBinNamePrefix() {
    return m_binNamePrefix;
  }

  /**
   * Returns the Capabilities of this filter.
   *
   * @return the capabilities of this object
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capabilities.Capability.MISSING_VALUES);

    // class
    result.enableAllClasses();
    result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

    result.enable(Capabilities.Capability.NO_CLASS);

    return result;
  }

  /**
   * Sets the format of the input instances. If the filter is able to determine
   * the output format before seeing any input instances, it does so here. This
   * default implementation clears the output format and output queue, and the
   * new batch flag is set. Overriders should call
   * <code>super.setInputFormat(Instances)</code>
   *
   * @param instanceInfo an Instances object containing the input instance
   *          structure (any instances contained in the object are ignored -
   *          only the structure is required).
   * @return true if the outputFormat may be collected immediately
   * @throws Exception if the inputFormat can't be set successfully
   */
  @Override
  public boolean setInputFormat(Instances instanceInfo) throws Exception {
    super.setInputFormat(instanceInfo);
    m_numBins = 0;
    m_numNumeric = 0;
    m_notConverted.clear();

    m_classIndex = instanceInfo.classIndex();
    for (int i = 0; i < instanceInfo.numAttributes(); i++) {
      if (instanceInfo.attribute(i).isNominal() || i == m_classIndex) {
        m_notConverted.add(instanceInfo.attribute(i));
      } else {
        m_numNumeric++;
      }
    }

    // Create the output buffer
    setOutputFormat();
    return true;
  }

  /**
   * Construct the output format
   */
  protected void setOutputFormat() {
    ArrayList<Attribute> newAtts = new ArrayList<>();

    int nearestPower2;
    for (nearestPower2 = 1; m_numNumeric > nearestPower2; nearestPower2 <<= 1)
      ;

    m_numBins = nearestPower2 / 2 / m_binWidth; // fft is symmetric - only need
                                                // half

    // bins
    for (int i = 0; i < m_numBins; i++) {
      newAtts.add(new Attribute(m_binNamePrefix + "_" + i));
    }

    // summary
    if (getAddTimeSummaryStats()) {
      newAtts.add(new Attribute("min"));
      newAtts.add(new Attribute("max"));
      newAtts.add(new Attribute("mean"));
      newAtts.add(new Attribute("std"));
    }

    if (getKeepOriginalNumerics()) {
      for (int i = 0; i < getInputFormat().numAttributes(); i++) {
        if (getInputFormat().attribute(i).isNumeric() && i != m_classIndex) {
          newAtts.add((Attribute) getInputFormat().attribute(i).copy());
        }
      }
    }

    // add any nominals class (unchanged)
    for (Attribute a : m_notConverted) {
      newAtts.add(a.copy(a.name()));
    }

    Instances outF =
      new Instances(getInputFormat().relationName() + "_fft", newAtts,
        getInputFormat().numInstances());
    if (m_classIndex >= 0) {
      String className = getInputFormat().attribute(m_classIndex).name();
      outF.setClass(outF.attribute(className));
    }
    setOutputFormat(outF);
  }

  /**
   * Compute FFT magnitudes for a batch of instances
   *
   * @return true if there are more instances left to process
   */
  @Override
  public boolean batchFinished() {
    Instances input = getInputFormat();
    double seqLen = m_numNumeric;

    int nearestPower2;
    for (nearestPower2 = 1; m_numNumeric > nearestPower2; nearestPower2 <<= 1)
      ;

    // can m_re-use for each instance
    AttFTHolder holder = new AttFTHolder();
    holder.re = new double[nearestPower2];
    holder.im = new double[nearestPower2];

    for (int i = 0; i < input.numInstances(); i++) {
      Instance current = input.instance(i);
      int counter = 0;
      for (int j = 0; j < input.numAttributes(); j++) {
        if (input.attribute(j).isNumeric() && j != m_classIndex) {
          holder.re[counter] = current.isMissing(j) ? 0 : current.value(j);
          if (!current.isMissing(j)) {
            holder.m_stats.add(current.value(j));
          }
          holder.im[counter++] = 0;
        }
      }
      while (counter < nearestPower2) {
        holder.re[counter] = 0;
        holder.im[counter++] = 0;
      }
      // inplace fft
      computeFFT(holder.re, holder.im);

      // make output instance
      push(makeOutputInstance(holder, current));
    }

    flushInput();
    m_NewBatch = true;
    return numPendingOutput() != 0;
  }

  /**
   * Construct an output instance given a sequence
   *
   * @param holder contains the sequence to compute the FFT for
   * @param current The current input instance
   * @return an output instance
   */
  protected Instance makeOutputInstance(AttFTHolder holder, Instance current) {
    double[] vals = new double[outputFormatPeek().numAttributes()];

    double numPoints = holder.re.length;
    int arrayPos = 0;
    for (int i = 0; i < m_numBins; i++) {
      double v = 0;
      for (int j = 0; j < m_binWidth; j++) {
        // abs val of a complex number
        double temp =
          (holder.re[arrayPos] * holder.re[arrayPos])
            + (holder.im[arrayPos] * holder.im[arrayPos]);
        temp = Math.sqrt(temp);
        if (m_normalizeMagnitude) {
          temp = (temp / numPoints) * 2; // * 2 as we are only using one side of
                                         // the fft
        }
        v += temp;
        arrayPos++;
      }
      vals[i] = v;
    }

    int start = m_numBins;
    if (getAddTimeSummaryStats()) {
      holder.m_stats.calculateDerived();
      vals[start++] = holder.m_stats.min;
      vals[start++] = holder.m_stats.max;
      vals[start++] = holder.m_stats.mean;
      vals[start++] = holder.m_stats.stdDev;
    }

    if (getKeepOriginalNumerics()) {
      for (int i = 0; i < getInputFormat().numAttributes(); i++) {
        if (getInputFormat().attribute(i).isNumeric() && i != m_classIndex) {
          vals[start++] = current.value(i);
        }
      }
    }

    // remaining nominal vals (if any)
    for (Attribute a : m_notConverted) {
      vals[start++] = current.value(a.index());
    }

    return new DenseInstance(1.0, vals);
  }

  /**
   * This computes an in-place complex-to-complex FFT x and y are the real and
   * imaginary arrays of 2^m points.
   *
   * @param x real array
   * @param y imaginary array
   */
  private void computeFFT(double[] x, double[] y) {
    int numPoints = x.length;
    int logPoints = (int) (Math.log(numPoints) / Math.log(2));

    // Do the bit reversal
    int halfPoints = numPoints / 2;
    int rev = 0;
    for (int i = 0; i < numPoints - 1; i++) {
      if (i < rev) {
        // swap the numbers
        double tx = x[i];
        double ty = y[i];
        x[i] = x[rev];
        y[i] = y[rev];
        x[rev] = tx;
        y[rev] = ty;
      }
      int mask = halfPoints;
      while (mask <= rev) {
        rev -= mask;
        mask >>= 1;
      }
      rev += mask;
    }

    // Compute the FFT
    double c1 = -1.0;
    double c2 = 0.0;
    int step = 1;
    for (int level = 0; level < logPoints; level++) {
      int increm = step * 2;
      double u1 = 1.0;
      double u2 = 0.0;
      for (int j = 0; j < step; j++) {
        for (int i = j; i < numPoints; i += increm) {
          // Butterfly
          double t1 = u1 * x[i + step] - u2 * y[i + step];
          double t2 = u1 * y[i + step] + u2 * x[i + step];
          x[i + step] = x[i] - t1;
          y[i + step] = y[i] - t2;
          x[i] += t1;
          y[i] += t2;
        }
        // U = exp ( - 2 PI j / 2 ^ level )
        double z = u1 * c1 - u2 * c2;
        u2 = u1 * c2 + u2 * c1;
        u1 = z;
      }
      c2 = Math.sqrt((1.0 - c1) / 2.0);
      c1 = Math.sqrt((1.0 + c1) / 2.0);

      step *= 2;
    }
  }

  public String globalInfo() {
    return "Fourier Transform filter that computes FFT magnitude spectrum over numeric "
      + "sequences stored in consecutive attribute values - i.e. each instance is "
      + "expected to contain a sequence to be transformed. Output consists of one "
      + "attribute for each entry in the spectrum. Provides an option to reduce "
      + "dimensionality by binning the spectrum. In this case, the value in a given "
      + "output bin is the sum of the FFT magnitude values that fall into that bin.\n\n"
      + "Based on original code by Anilkumar Patro at Worcester Polytechnic "
      + "Institute";
  }

  /**
   * Entry point for testing filter
   */
  public static void main(String[] args) {
    runFilter(new FourierTransform(), args);
  }

  /**
   * Helper class
   */
  private static class AttFTHolder {
    public double[] re;
    public double[] im;
    public Stats m_stats = new Stats();
  }
}
