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

package weka.filters.unsupervised.attribute;

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
 * magnitude spectrum over sequences stored in numeric attributes. Attributes
 * are expected to be be time series, and FFT is computed over consecutive
 * sequences of N data points/instances (as specified by the Sequence length
 * option). Output consists of one attribute for each entry in the spectrum for
 * each numeric attribute. So, N instances in the original format become 1
 * instance in the output. Provides an option to reduce dimensionality by
 * binning the spectrum. In this case, the value in a given output bin is the
 * sum of the FFT magnitude values that fall into that bin. <br>
 * <br>
 * <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 * 
 * <pre>
 * -seq &lt;length&gt;
 *  Sequence length.
 * </pre>
 * 
 * <pre>
 * -width &lt;integer&gt;
 *  FFT bin width (default = 1).
 * </pre>
 * 
 * <pre>
 * -normalize
 *  Normalize FFT magnitude.
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

  private static final long serialVersionUID = 7521475436573700495L;

  /**
   * Complex number helper for FTs
   */
  private AttFTHolder[] m_attsFT;

  /** The length of the sequences to process */
  protected int m_sequenceLength = 8;

  /** Spectrum bin width */
  protected int m_binWidth = 1;

  /** Whether to normalize the fft magnitude values */
  protected boolean m_normalizeMagnitude;

  /** Whether to add some time-based summary stats */
  protected boolean m_addTimeSummaryStats;

  /** Holds the computed number of bins */
  protected int m_numBins;

  /** Holds the number of numeric attributes in the input data */
  protected int m_numNumeric;

  /**
   * Set the length of sequences to process
   *
   * @param sequenceLength the length of the sequences to process
   */
  @OptionMetadata(displayName = "Sequence length",
    description = "The length (N) of sequence to FFT transform. consecutive "
      + "sets of N points for a numeric attribute are converted.",
    commandLineParamName = "seq", commandLineParamSynopsis = "-seq <integer>",
    displayOrder = 1)
  public void setSequenceLength(int sequenceLength) {
    m_sequenceLength = sequenceLength;
  }

  /**
   * Get the length of sequences to process
   *
   * @return the length of sequences to process
   */
  public int getSequenceLength() {
    return m_sequenceLength;
  }

  /**
   * Set the bin width
   *
   * @param binSize the bin width
   */
  @OptionMetadata(displayName = "Bin width",
    description = "Bin width for FFT magnitudes (default = 1, i.e. no binning). "
      + "Within each bin FFT magnitude values are summed.",
    commandLineParamName = "width",
    commandLineParamSynopsis = "-width <integer>", displayOrder = 2)
  public void setBinWidth(int binSize) {
    m_binWidth = binSize;
  }

  /**
   * Get the bin width
   *
   * @return the bin width
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
   * Set whether to normalize magnitude values by by N * 2
   *
   * @param normalizeMagnitude true if magnitude is to be normalized
   */
  @OptionMetadata(
    displayName = "Normalize FFT magnitude values",
    description = "Normalized magnitude values by N * 2. Scaled by 2 as we only "
      + "use half the (symmetric) FFT spectrum",
    commandLineParamName = "normalize",
    commandLineParamSynopsis = "-normalize",
    commandLineParamIsFlag = true,
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
    result.enable(Capabilities.Capability.MISSING_VALUES);

    // class
    result.enableAllClasses();
    result.enable(Capabilities.Capability.NO_CLASS);
    result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

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

    for (int i = 0; i < instanceInfo.numAttributes(); ++i) {
      if (!instanceInfo.attribute(i).isNumeric()) {
        throw new UnsupportedAttributeTypeException(
          "All attributes must be numeric");
      }
    }

    // Create the output buffer
    // setOutputFormat();
    setOutputFormat();
    return true;
  }

  /**
   * Construct the output format
   */
  protected void setOutputFormat() {
    ArrayList<Attribute> newAtts = new ArrayList<>();

    int nearestPower2;
    for (nearestPower2 = 1; m_sequenceLength > nearestPower2; nearestPower2 <<=
      1)
      ;

    m_numBins = nearestPower2 / 2 / m_binWidth;
    // m_numBins = m_sequenceLength / 2 / m_binWidth; // fft is symmetric - only
    // need half
    for (int i = 0; i < getInputFormat().numAttributes(); i++) {
      Attribute current = getInputFormat().attribute(i);
      if (current.isNumeric()) {
        m_numNumeric++;
        String attPrefix = current.name() + "_ft_";
        for (int j = 0; j < m_numBins; j++) {
          Attribute newAtt = new Attribute(attPrefix + j);
          newAtts.add(newAtt);
        }

        if (m_addTimeSummaryStats) {
          newAtts.add(new Attribute(current.name() + "_min"));
          newAtts.add(new Attribute(current.name() + "_max"));
          newAtts.add(new Attribute(current.name() + "_mean"));
          newAtts.add(new Attribute(current.name() + "_std"));
        }
      }
    }

    Instances outF =
      new Instances(getInputFormat().relationName() + "_fft", newAtts,
        getInputFormat().numInstances() / m_sequenceLength);
    setOutputFormat(outF);
  }

  /**
   * Signify that this batch of input to the filter is finished. If the filter
   * requires all instances prior to filtering, output() may now be called to
   * retrieve the filtered instances. Any subsequent instances filtered should
   * be filtered based on setting obtained from the first batch (unless the
   * inputFormat has been m_re-assigned or new options have been set). This
   * default implementation assumes all instance processing occurs during
   * inputFormat() and input().
   *
   * @return true if there are instances pending output
   * @throws NullPointerException if no input structure has been defined,
   * @throws Exception if there was a problem finishing the batch.
   */
  @Override
  public boolean batchFinished() throws Exception {
    if (getInputFormat() == null) {
      throw new IllegalStateException("No input instance format defined");
    }

    Instances input = getInputFormat();
    if (m_sequenceLength > input.numInstances()) {
      throw new Exception("Sequence length must be <= number of instances");
    }

    AttFTHolder[] holder = new AttFTHolder[m_numNumeric];

    int nearestPower2;
    for (nearestPower2 = 1; m_sequenceLength > nearestPower2; nearestPower2 <<=
      1)
      ;

    for (int i = 0; i < holder.length; i++) {
      holder[i] = new AttFTHolder();
      holder[i].m_re = new double[nearestPower2];
      holder[i].m_im = new double[nearestPower2];
    }

    int counter = 0;
    for (int k = 0; k < input.numInstances(); k++) {
      int numCounter = 0;
      if (counter < m_sequenceLength) {
        for (int i = 0; i < input.numAttributes(); i++) {
          if (input.attribute(i).isNumeric()) {
            if (counter < m_sequenceLength) {
              holder[numCounter].m_re[counter] =
                input.instance(k).isMissing(i) ? 0 : input.instance(k).value(i);
              if (!input.instance(k).isMissing(i)) {
                holder[numCounter].m_stats.add(input.instance(k).value(i));
              }
            } else {
              holder[numCounter].m_re[counter] = 0;
            }
            holder[numCounter++].m_im[counter] = 0;
          }
        }
        counter++;
      } else {
        // pad out with zeros
        while (counter < nearestPower2) {
          for (int i = 0; i < holder.length; i++) {
            holder[i].m_re[counter] = 0;
            holder[i].m_im[counter] = 0;
          }
          counter++;
        }
        k--;
      }

      if (counter == nearestPower2) {
        // - e.g. mean, std. dev, skewness, kurtoses. Padding will affect this.
        for (int i = 0; i < holder.length; i++) {
          // inplace FT
          computeFFT(holder[i].m_re, holder[i].m_im);
        }

        // System.err.println("*** Making output instance");
        push(makeOutputInstance(holder));
        counter = 0;
      }
    }

    // Last buffer - pad or discard entirely?
    if (counter > nearestPower2 / 2) {
      for (int i = counter; i < nearestPower2; i++) {
        for (int j = 0; j < holder.length; j++) {
          holder[j].m_re[i] = 0;
          holder[j].m_im[i] = 0;
        }
      }

      for (int j = 0; j < holder.length; j++) {
        // inplace FT
        computeFFT(holder[j].m_re, holder[j].m_im);
      }
      push(makeOutputInstance(holder));
    }

    flushInput();
    m_NewBatch = true;
    return numPendingOutput() != 0;
  }

  /**
   * Construct an output instance given an array of input sequences
   *
   * @param holders the input sequences to process into one output instance
   * @return an output instance
   */
  protected Instance makeOutputInstance(AttFTHolder[] holders) {
    int numTimeSummary = m_numNumeric * (getAddTimeSummaryStats() ? 4 : 0);
    double[] vals = new double[m_numBins * m_numNumeric + numTimeSummary];
    /*
     * for (int i = 0; i < holders[0].m_re.length; i++) {
     * System.out.println(holders[0].m_re[i]); } System.out.println();
     */

    int pos = 0;
    for (int i = 0; i < m_numNumeric; i++) {
      AttFTHolder current = holders[i];
      double numPoints = current.m_re.length;
      int arrayPos = 0;
      for (int j = 0; j < m_numBins; j++) {
        double v = 0;
        for (int k = 0; k < m_binWidth; k++) {
          // abs value of a complex number
          double temp =
            (current.m_re[arrayPos] * current.m_re[arrayPos])
              + (current.m_im[arrayPos] * current.m_im[arrayPos]);
          temp = Math.sqrt(temp);
          if (getNormalizeMagnitude()) {
            temp = (temp / numPoints) * 2; // * 2 as we are only using one side
                                           // of the fft
          }
          v += temp;
          arrayPos++;
        }
        vals[pos++] = v;
      }
      if (getAddTimeSummaryStats()) {
        current.m_stats.calculateDerived();
        vals[pos++] = current.m_stats.min;
        vals[pos++] = current.m_stats.max;
        vals[pos++] = current.m_stats.mean;
        vals[pos++] = current.m_stats.stdDev;
      }
    }
    for (int i = 0; i < holders.length; i++) {
      holders[i].m_stats = new Stats();
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
    return "Fourier Transform filter that computes FFT magnitude spectrum over sequences "
      + "stored in numeric attributes. Attributes are expected to be be time series, "
      + "and FFT is computed over consecutive sequences of N data points/instances (as "
      + "specified by the Sequence length option). Output consists of one attribute "
      + "for each entry in the spectrum for each numeric attribute. So, N instances in "
      + "the original format become 1 instance in the output. Provides an option to "
      + "reduce dimensionality by binning the spectrum. In this case, the value in a "
      + "given output bin is the sum of the FFT magnitude values that fall into that "
      + "bin.\n\nBased on original code by Anilkumar Patro at Worcester Polytechnic "
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
    public double[] m_re;
    public double[] m_im;
    public Stats m_stats = new Stats();
  }
}
