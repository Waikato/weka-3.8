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
 *    HotSpot.java
 *    Copyright (C) 2008 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.associations;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.CapabilitiesHandler;
import weka.core.CapabilitiesIgnorer;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.SingleIndex;
import weka.core.Utils;

/**
 <!-- globalinfo-start -->
 * HotSpot learns a set of rules (displayed in a tree-like structure) that maximize/minimize a target variable/value of interest. With a nominal target, one might want to look for segments of the data where there is a high probability of a minority value occuring (given the constraint of a minimum support). For a numeric target, one might be interested in finding segments where this is higher on average than in the whole data set. For example, in a health insurance scenario, find which health insurance groups are at the highest risk (have the highest claim ratio), or, which groups have the highest average insurance payout.  This algorithm is similar in spirit to the PRIM bump hunting algorithm described by Friedman and Fisher (1999).
 * <br><br>
 <!-- globalinfo-end -->
 *
 <!-- options-start -->
 * Valid options are: <p>
 * 
 * <pre> -c &lt;num | first | last | attribute name&gt;
 *  The target index. (default = last)</pre>
 * 
 * <pre> -V &lt;num | first | last&gt;
 *  The target value (nominal target only, default = first)</pre>
 * 
 * <pre> -L
 *  Minimize rather than maximize.</pre>
 * 
 * <pre> -S &lt;num&gt;
 *  Minimum value count (nominal target)/segment size (numeric target).
 *  Values between 0 and 1 are 
 *  interpreted as a percentage of 
 *  the total population (numeric) or total target value
 *  population size (nominal); values &gt; 1 are 
 *  interpreted as an absolute number of 
 *  instances (default = 0.3)</pre>
 * 
 * <pre> -M &lt;num&gt;
 *  Maximum branching factor (default = 2)</pre>
 * 
 * <pre> -length &lt;num&gt;
 *  Maximum rule length (default = -1, i.e. no maximum)</pre>
 * 
 * <pre> -sum
 *  Operate on sum, rather than average, for numeric target. Note, this mode
 *  can only operate on nominal attributes.</pre>
 * 
 * <pre> -direct
 *  Consider direct max/min sum of target rather than sum relative to that expected across
 *  categories of a nominal attribute. Use in conjunction with -sum.</pre>
 * 
 * <pre> -I &lt;num&gt;
 *  Minimum improvement in target value in order 
 *  to add a new branch/test (default = 0.01 (1%))</pre>
 * 
 * <pre> -Z
 *  Treat zero (first value) as missing for nominal attributes</pre>
 * 
 * <pre> -R
 *  Output a set of rules instead of a tree structure</pre>
 * 
 * <pre> -D
 *  Output debugging info (duplicate rule lookup 
 *  hash table stats)</pre>
 * 
 <!-- options-end -->
 *
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}org
 * @version $Revision$
 */
public class HotSpot implements Associator, OptionHandler, RevisionHandler,
  CapabilitiesHandler, CapabilitiesIgnorer, Drawable, AssociationRulesProducer,
  Serializable {

  static final long serialVersionUID = 42972325096347677L;

  /** index of the target attribute */
  protected SingleIndex m_targetSI = new SingleIndex("last");
  protected int m_target;

  /** Holds the user-supplied support value (fraction or absolute num instances) */
  protected String m_supportString = "0.33";

  /** Support as a fraction of the total training set */
  protected double m_support;

  /** Support as an instance count */
  private int m_supportCount;

  /** The global value of the attribute of interest (mean or probability) */
  protected double m_globalTarget;

  /** The minimum improvement necessary to justify adding a test */
  protected double m_minImprovement;

  /** Actual global support of the target value (discrete target only) */
  protected int m_globalSupport;

  /** For discrete target, the index of the value of interest */
  protected SingleIndex m_targetIndexSI = new SingleIndex("first");
  protected int m_targetIndex;

  /** At each level of the tree consider at most this number extensions */
  protected int m_maxBranchingFactor;

  /** The maximum depth of a path in the tree (or length of a rule) */
  protected int m_maxRuleLength = 3;

  /** Attempt to force the maximum rule length by reducing support */
  protected boolean m_forceMaxRuleLength;

  /**
   * Use a heuristic (sqrt(num instances)) to set the minimum segment size for
   * for direct and relative sum trees (numeric target only)
   */
  protected boolean m_heuristic;

  /**
   * Treat zero as missing for nominal attributes (useful for basket data in
   * sparse format)
   */
  protected boolean m_treatZeroAsMissing;

  /** Number of instances in the full data */
  protected int m_numInstances;

  /** Number of instances with non-missing target values in the full data */
  protected int m_numNonMissingTarget;

  /** The head of the tree */
  protected HotNode m_head;

  /** Header of the training data */
  protected Instances m_header;

  /** Debugging stuff */
  protected int m_lookups = 0;
  protected int m_insertions = 0;
  protected int m_hits = 0;

  protected boolean m_debug;

  /** Minimize, rather than maximize the target */
  protected boolean m_minimize;

  /**
   * Whether to work with the sum (rather than average) for a numeric target. In this case,
   * only nominal attributes can be considered and merit is based on relative improvement
   * over the average/expected sum of the categories of a nominal attribute under consideration -
   * i.e. high/low concentration/volume of sum relative to expected.
   */
  protected boolean m_sumForNumericTarget;

  /**
   * When true (in conjunction with -sum), then just look for those nominal categories with
   * highest/smallest direct sum of target values (i.e. max/min volume) rather than
   * relative to the expected sum
   */
  protected boolean m_directSum;

  /** Error messages relating to too large/small support values */
  protected String m_errorMessage;

  /** Rule lookup table */
  protected HashMap<HotSpotHashKey, String> m_ruleLookup;

  /** True if a set of rules is to be output instead of a tree structure */
  protected boolean m_outputRules = false;

  /** Whether capabilities checking should be disabled */
  protected boolean m_doNotCheckCapabilities;

  /**
   * Constructor
   */
  public HotSpot() {
    resetOptions();
  }

  /**
   * Returns a string describing this classifier
   *
   * @return a description of the classifier suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String globalInfo() {
    return "HotSpot learns a set of rules (displayed in a tree-like structure) "
      + "that maximize/minimize a target variable/value of interest. "
      + "With a nominal target, one might want to look for segments of the "
      + "data where there is a high probability of a minority value occuring ("
      + "given the constraint of a minimum support). For a numeric target, "
      + "one might be interested in finding segments where this is higher "
      + "on average than in the whole data set. For example, in a health "
      + "insurance scenario, find which health insurance groups are at "
      + "the highest risk (have the highest claim ratio), or, which groups "
      + "have the highest average insurance payout.  This algorithm "
      + "is similar in spirit to the PRIM bump hunting algorithm described by Friedman "
      + "and Fisher (1999).";
  }

  /**
   * Returns default capabilities of HotSpot
   *
   * @return the capabilities of HotSpot
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = new Capabilities(this);
    result.disableAll();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.NO_CLASS);
    result.enable(Capability.NUMERIC_CLASS);
    result.enable(Capability.NOMINAL_CLASS);

    return result;
  }

  /**
   * Hash key class for sets of attribute, value tests
   */
  protected class HotSpotHashKey implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = 3962829200560373755L;

    // split values, one for each attribute (0 indicates att not used).
    // for nominal indexes, 1 is added so that 0 can indicate not used.
    protected double[] m_splitValues;

    // 0 = not used, 1 = "<=", 2 = "=", 3 = ">"
    protected byte[] m_testTypes;

    protected boolean m_computed = false;
    protected int m_key;

    public HotSpotHashKey(double[] splitValues, byte[] testTypes) {
      m_splitValues = splitValues.clone();
      m_testTypes = testTypes.clone();
    }

    @Override
    public boolean equals(Object b) {
      if ((b == null) || !(b.getClass().equals(this.getClass()))) {
        return false;
      }
      HotSpotHashKey comp = (HotSpotHashKey) b;
      boolean ok = true;
      for (int i = 0; i < m_splitValues.length; i++) {
        if (m_splitValues[i] != comp.m_splitValues[i]
          || m_testTypes[i] != comp.m_testTypes[i]) {
          ok = false;
          break;
        }
      }
      return ok;
    }

    @Override
    public int hashCode() {

      if (m_computed) {
        return m_key;
      } else {
        int hv = 0;
        for (int i = 0; i < m_splitValues.length; i++) {
          hv += (m_splitValues[i] * 5 * i);
          hv += (m_testTypes[i] * i * 3);
        }
        m_computed = true;

        m_key = hv;
      }
      return m_key;
    }
  }

  protected int getSupportForInstances(Instances insts) {

    int globalSupport = insts.attribute(m_target).isNominal() ?
      insts.attributeStats(m_target).nominalCounts[m_targetIndex] : 0;

    int result = 0;
    if (insts.attribute(m_target).isNumeric()
      && m_heuristic && (m_directSum || m_sumForNumericTarget)) {
      result = (int) Math.sqrt(insts.numInstances()) + 2;
    } else {
      result = insts.attribute(m_target).isNumeric() ?
        (int) Math.floor((m_support * insts.numInstances()) + 0.5d) :
        (int) Math.floor((m_support * globalSupport) + 0.5d);
    }

    if (result < 1) {
      result = 1;
    }

    return result;
  }

  /**
   * Build the tree
   *
   * @param instances the training instances
   * @throws Exception if something goes wrong
   */
  @Override
  public void buildAssociations(Instances instances) throws Exception {

    // can associator handle the data?
    getCapabilities().testWithFail(instances);

    if (m_supportString == null || m_supportString.length() == 0) {
      throw new Exception("No support value provided!");
    }
    m_support = Double.parseDouble(m_supportString);

    m_errorMessage = null;
    try {
      m_targetSI.setUpper(instances.numAttributes() - 1);
      m_target = m_targetSI.getIndex();
    } catch (Exception ex) {
      // try to match the string against an attribute name
      String value = m_targetSI.getSingleIndex();
      int index = -1;
      for (int i = 0; i < instances.numAttributes(); i++) {
        if (instances.attribute(i).name().indexOf(value) > -1) {
          index = i;
          break;
        }
      }

      if (index == -1) {
        throw new Exception("Can't find an attribute containing the string "
          + value);
      }

      m_target = index;
    }

    Instances inst = new Instances(instances);
    inst.setClassIndex(m_target);
    m_header = new Instances(inst, 0);
    // inst.deleteWithMissingClass();

    if (inst.attribute(m_target).isNominal()) {
      m_targetIndexSI.setUpper(inst.attribute(m_target).numValues() - 1);
      m_targetIndex = m_targetIndexSI.getIndex();

      if (m_sumForNumericTarget || m_directSum) {
        // Turn these off, as they only apply to a numeric target
        m_sumForNumericTarget = m_directSum = false;
      }
    } else {
      m_targetIndexSI.setUpper(1); // just to stop this SingleIndex from moaning
    }

    if (inst.attribute(m_target).isNumeric()) {
      if (m_sumForNumericTarget || m_directSum) {
        m_globalTarget = inst.attributeStats(m_target).numericStats.sum;
      } else {
        m_globalTarget = inst.meanOrMode(m_target);
      }
      m_numNonMissingTarget = inst.numInstances()
        - inst.attributeStats(m_target).missingCount;

      if (m_supportCount > m_numInstances) {
        m_errorMessage =
          "Error: support set to more instances than there are in the data!";
        return;
      }
      if ((m_sumForNumericTarget || m_directSum) && !inst.checkForAttributeType(Attribute.NOMINAL)) {
        m_errorMessage = "Aggregation type sum requires at least one nominal attribute in the data";
        return;
      }
    } else {
      double[] probs =
        new double[inst.attributeStats(m_target).nominalCounts.length];
      for (int i = 0; i < probs.length; i++) {
        probs[i] = inst.attributeStats(m_target).nominalCounts[i];
      }
      m_globalSupport = (int) probs[m_targetIndex];
      // check that global support is greater than min support
      if (m_globalSupport < m_supportCount) {
        m_errorMessage = "Error: minimum support " + m_supportCount
          + " is too high. Target value "
          + inst.attribute(m_target).value(m_targetIndex) + " has support "
          + m_globalSupport + ".";
      }

      // Utils.normalize(probs);
      for (int i = 0; i < probs.length; i++) {
        probs[i] /= inst.numInstances();
      }
      m_globalTarget = probs[m_targetIndex];
      /*
       * System.err.println("Global target " + m_globalTarget);
       * System.err.println("Min support count " + m_supportCount);
       */
    }

    if (m_support <= 0) {
      throw new Exception("Support must be greater than zero.");
    }

    m_numInstances = inst.numInstances();
    if (m_support >= 1) {
      m_supportCount = (int) m_support;
      if (inst.attribute(m_target).isNumeric()) {
        m_support = m_support / m_numInstances;
      } else {
        // fraction of the global support in the case of a nominal target
        m_support = m_support / m_globalSupport;
      }
    }

    m_supportCount = getSupportForInstances(instances);
    /* m_supportCount = inst.attribute(m_target).isNumeric() ? (int) Math
      .floor((m_support * m_numInstances) + 0.5d) : (int) Math
      .floor((m_support * m_globalSupport) + 0.5d); */

    // m_supportCount = (int)(m_support * m_numInstances);

    if (m_supportCount < 1) {
      m_supportCount = 1;
    }

    m_ruleLookup = new HashMap<HotSpotHashKey, String>();
    double[] splitVals = new double[m_header.numAttributes()];
    byte[] tests = new byte[m_header.numAttributes()];

    m_head = new HotNode(inst, m_globalTarget, splitVals, tests, 0, m_supportCount);
    // m_head = new HotNode(inst, m_globalTarget);
  }

  /**
   * Return the tree as a string
   *
   * @return a String containing the tree
   */
  @Override
  public String toString() {
    StringBuffer buff = new StringBuffer();
    buff.append("\nHot Spot\n========");
    if (m_errorMessage != null) {
      buff.append("\n\n" + m_errorMessage + "\n\n");
      return buff.toString();
    }
    if (m_head == null) {
      buff.append("No model built!");
      return buff.toString();
    }
    buff.append("\nMode: ").append(m_minimize ? "minimise" : "maximise");
    buff.append("\nTotal population: ");
    buff.append("" + m_numInstances + " instances");
    buff.append("\nTarget attribute: " + m_header.attribute(m_target).name());
    if (m_header.attribute(m_target).isNominal()) {
      buff.append("\nTarget value: "
        + m_header.attribute(m_target).value(m_targetIndex));
      buff.append(" [value count in total population: " + m_globalSupport
        + " instances (" + Utils.doubleToString((m_globalTarget * 100.0), 2)
        + "%)]");

      buff.append("\nMinimum value count for segments: ");
    } else {
      String aggType = m_sumForNumericTarget || m_directSum ? "sum" : "average";
      buff.append("\nTarget " + aggType + " in total population: "
        + Utils.doubleToString(m_globalTarget, 3));
      buff.append("\nMinimum segment size: ");
    }
    double supportPercentage = 0;
    if (m_header.attribute(m_target).isNumeric() && m_heuristic && (m_directSum || m_sumForNumericTarget)) {
      supportPercentage = ((double) m_supportCount / m_numInstances) * 100.0;
    } else {
      supportPercentage = m_support * 100.0;
    }
    buff
      .append("" + m_supportCount + " instances ("
        + Utils.doubleToString(supportPercentage, 2)
        + "% of " +
        (m_header.attribute(m_target).isNominal() ? "target value " : "")
        + "total population)");
    buff.append("\nMaximum branching factor: " + m_maxBranchingFactor);
    buff.append("\nMaximum rule length: "
      + (m_maxRuleLength < 0 ? "unbounded" : "" + m_maxRuleLength));
    if (!m_directSum) {
      buff.append("\nMinimum improvement: ");
    }
    if (m_header.attribute(m_target).isNumeric() && m_sumForNumericTarget || m_directSum) {
      if (!m_directSum) {
        double xsNeeded = m_minimize ? 1.0 - m_minImprovement : 1.0 + m_minImprovement;
        double minImprovPercent = m_minImprovement * 100.0;
        buff.append(Utils.doubleToString(minImprovPercent, 2))
          .append("% (").append(Utils.doubleToString(xsNeeded, 2))
          .append("x) ").append((m_minimize ? "below " : "above ")).append("(of) the {expected sum} ")
          .append("over the values of a splitting attribute");
      }
    } else {
      String targetType = m_header.attribute(m_target).isNominal() ? " in probability" : " of average";
      String objective = m_minimize ? "reduction" : "increase";
      buff.append(Utils.doubleToString((m_minImprovement * 100.0), 2) + "% ");
      buff.append(objective).append(targetType);
    }

    buff.append("\n\n");
    if (!m_outputRules) {
      buff.append(m_header.attribute(m_target).name());
      if (m_header.attribute(m_target).isNumeric()) {
        buff.append(" (" + Utils.doubleToString(m_globalTarget, 4) + ")");
      } else {
        buff.append("=" + m_header.attribute(m_target).value(m_targetIndex)
          + " (");
        buff.append("" + Utils.doubleToString((m_globalTarget * 100.0), 2)
          + "% [");
        buff.append("" + m_globalSupport + "/" + m_numInstances + "])");
      }
    }

    if (!m_outputRules) {
      m_head.dumpTree(0, buff);
    } else {
      List<AssociationRule> rules = new ArrayList<AssociationRule>();
      try {
        m_head.getRules(rules, new ArrayList<Item>());
        Collections.sort(rules);
        for (AssociationRule r : rules) {
          buff.append(r.toString() + "\n");
        }
      } catch (Exception e) {
        e.printStackTrace();
      }
    }
    buff.append("\n");

    if (m_debug) {
      buff.append("\n=== Duplicate rule lookup hashtable stats ===\n");
      buff.append("Insertions: " + m_insertions);
      buff.append("\nLookups : " + m_lookups);
      buff.append("\nHits: " + m_hits);
      buff.append("\n");
    }

    return buff.toString();
  }

  @Override
  public String graph() throws Exception {
    m_head.assignIDs(-1);

    StringBuffer text = new StringBuffer();

    text.append("digraph HotSpot {\n");
    text.append("rankdir=LR;\n");
    text.append("N0 [label=\"" + m_header.attribute(m_target).name());

    if (m_header.attribute(m_target).isNumeric()) {
      text.append("\\n(" + Utils.doubleToString(m_globalTarget, 4) + ")");
    } else {
      text.append("=" + m_header.attribute(m_target).value(m_targetIndex)
        + "\\n(");
      text.append("" + Utils.doubleToString((m_globalTarget * 100.0), 2)
        + "% [");
      text.append("" + m_globalSupport + "/" + m_numInstances + "])");
    }
    text.append("\" shape=plaintext]\n");

    m_head.graphHotSpot(text);

    text.append("}\n");
    return text.toString();
  }

  public Map<String, Object> graphAsMap() throws Exception {
    m_head.assignIDs(-1);
    Map<String, Object> graph = new LinkedHashMap<String, Object>();
    String details = "";
    if (m_header.attribute(m_target).isNumeric()) {
      details = " (" + m_globalTarget + ")";
    } else {
      details = " = " + m_header.attribute(m_target).value(m_targetIndex)
        + " (" + (m_globalTarget * 100.0) + "% [" + m_globalSupport + "/"
        + m_numInstances + "])";
    }
    graph.put("target", m_header.attribute(m_target).name() + details);
    String aggregationType = "";
    if (m_header.attribute(m_target).isNumeric()) {
      if (m_sumForNumericTarget || m_directSum) {
        aggregationType = "sum";
      } else {
        aggregationType = "average";
      }
    } else {
      aggregationType = "probability";
    }
    String objective = m_minimize ? "minimize" : "maximize";
    graph.put("aggregation", aggregationType);
    graph.put("objective", objective);
    if (m_errorMessage != null) {
      graph.put("error", m_errorMessage);
    }

    if (m_head != null) {
      m_head.graphAsMap(graph);
    }
    return graph;
  }

  /**
   * Inner class representing a node/leaf in the tree
   */
  protected class HotNode implements Serializable {

    /**
     * For serialization
     */
    private static final long serialVersionUID = -4665984155566279901L;

    /**
     * An inner class holding data on a particular attribute value test
     */
    protected class HotTestDetails implements Comparable<HotTestDetails>,
      Serializable {

      /**
       * For serialization
       */
      private static final long serialVersionUID = -8403762320170624616L;
      public double m_merit;
      public int m_supportLevel;
      public int m_subsetSize;
      public int m_splitAttIndex;
      public double m_splitValue;
      public boolean m_lessThan;

      public double m_targetSum; // only when target is numeric and aggregation is sum
      // only when target is numeric, split is nominal and aggregation is sum (relative)
      public double m_expectedTargetSumPerCategory = -1;

      // holds the distribution of the target (across all nominal labels for a nominal test)
      public double[] m_distributionForNominalTest;
      // holds the distribution of counts (across all nominal labels for a nominal test)
      public int[] m_countsForNominalTest;

      public HotTestDetails(int attIndex, double splitVal, boolean lessThan,
        int support, int subsetSize, double merit) {
        m_merit = merit;
        m_splitAttIndex = attIndex;
        m_splitValue = splitVal;
        m_lessThan = lessThan;
        m_supportLevel = support;
        m_subsetSize = subsetSize;
      }

      public HotTestDetails(int attIndex, double splitVal, int support, double targetSum, double merit) {
        this(attIndex, splitVal, false, support, support, merit);
        m_targetSum = targetSum;
      }

      // reverse order for maximize as PriorityQueue has the least element at
      // the head
      @Override
      public int compareTo(HotTestDetails comp) {
        int result = 0;
        if (m_minimize) {
          if (m_merit == comp.m_merit) {
            // larger support is better
            if (m_supportLevel == comp.m_supportLevel) {
            } else if (m_supportLevel > comp.m_supportLevel) {
              result = -1;
            } else {
              result = 1;
            }
          } else if (m_merit < comp.m_merit) {
            result = -1;
          } else {
            result = 1;
          }
        } else {
          if (m_merit == comp.m_merit) {
            // larger support is better
            if (m_supportLevel == comp.m_supportLevel) {
            } else if (m_supportLevel > comp.m_supportLevel) {
              result = -1;
            } else {
              result = 1;
            }
          } else if (m_merit < comp.m_merit) {
            result = 1;
          } else {
            result = -1;
          }
        }
        return result;
      }

      public String toString() {
        StringBuilder b = new StringBuilder();
        if (m_insts.attribute(m_splitAttIndex).isNominal()) {
          b.append(m_insts.attribute(m_splitAttIndex).name()).append(" = ")
            .append(m_insts.attribute(m_splitAttIndex).value((int) m_splitValue))
            .append(": " + m_merit);
        }
        return b.toString();
      }
    }

    // the instances at this node
    protected Instances m_insts;

    // the value (to beat) of the target for these instances
    protected double m_targetValue;

    // child nodes
    protected HotNode[] m_children;
    protected HotTestDetails[] m_testDetails;

    public int m_id;

    /**
     * Constructor
     *
     * @param insts the instances at this node
     * @param targetValue the target value
     * @param splitVals the values of attributes split on so far down this
     *          branch
     * @param tests the types of tests corresponding to the split values (<=, =,
     *          >)
     */
    public HotNode(Instances insts, double targetValue, double[] splitVals,
      byte[] tests, int depth, int supportCount) {

      if (depth == m_maxRuleLength) {
        return;
      }

      m_insts = insts;
      m_targetValue = targetValue;
      PriorityQueue<HotTestDetails> splitQueue =
        new PriorityQueue<HotTestDetails>();

      // Consider each attribute
      for (int i = 0; i < m_insts.numAttributes(); i++) {
        if (i != m_target) {
          if (m_insts.attribute(i).isNominal()) {
            evaluateNominal(i, splitQueue, supportCount);
          } else {
            if (!(m_sumForNumericTarget || m_directSum)) {
              evaluateNumeric(i, splitQueue, supportCount);
            }
            // skip numeric attributes if using the sum, as finding
            // the best split point does not make sense in this case
          }
        }
      }

      boolean queueHasEntries = splitQueue.size() > 0;
      if (queueHasEntries) {
        processQueue(insts, splitQueue, splitVals, tests, depth, supportCount);
      }

      // if there was nothing in the queue, or all candidates (in the case of nominal only splits)
      // had already been tested higher in the branch (children.length = 0) and we are forcing
      // max rule length then try again with a smaller support
      if (m_forceMaxRuleLength && (!queueHasEntries || m_children.length == 0)) {
        // reduce support as row count for this subset and try again
        int reducedSupport = getSupportForInstances(insts);
        if (reducedSupport < 2 && !(m_sumForNumericTarget || m_directSum)) {
          reducedSupport = 2;
        }

        // Consider each attribute
        for (int i = 0; i < m_insts.numAttributes(); i++) {
          if (i != m_target) {
            if (m_insts.attribute(i).isNominal()) {
              evaluateNominal(i, splitQueue, reducedSupport);
            } else {
              if (!(m_sumForNumericTarget || m_directSum)) {
                evaluateNumeric(i, splitQueue, reducedSupport);
              }
              // skip numeric attributes if using the sum, as finding
              // the best split point does not make sense in this case
            }
          }
        }

        if (splitQueue.size() > 0) {
          processQueue(insts, splitQueue, splitVals, tests, depth, reducedSupport);
        }
      }
      m_insts = new Instances(m_insts, 0);
    }

    protected void processQueue(Instances insts, PriorityQueue<HotTestDetails> splitQueue, double[] splitVals,
      byte[] tests, int depth, int supportCount) {
      int queueSize = splitQueue.size();

      // count how many of the potential children are unique
      ArrayList<HotTestDetails> newCandidates =
        new ArrayList<HotTestDetails>();
      ArrayList<HotSpotHashKey> keyList = new ArrayList<HotSpotHashKey>();

      for (int i = 0; i < queueSize; i++) {
        if (newCandidates.size() < m_maxBranchingFactor) {
          HotTestDetails temp = splitQueue.poll();
          double[] newSplitVals = splitVals.clone();
          byte[] newTests = tests.clone();
          newSplitVals[temp.m_splitAttIndex] = temp.m_splitValue + 1;
          newTests[temp.m_splitAttIndex] = (m_header
            .attribute(temp.m_splitAttIndex).isNominal()) ? (byte) 2 // ==
            : (temp.m_lessThan) ? (byte) 1 : (byte) 3;
          HotSpotHashKey key = new HotSpotHashKey(newSplitVals, newTests);
          m_lookups++;
          if (!m_ruleLookup.containsKey(key)) {
            // insert it into the hash table
            m_ruleLookup.put(key, "");
            newCandidates.add(temp);
            keyList.add(key);
            m_insertions++;
          } else {
            m_hits++;
          }
        } else {
          break;
        }
      }

      m_children =
        new HotNode[(newCandidates.size() < m_maxBranchingFactor) ? newCandidates
          .size()
          : m_maxBranchingFactor];

      // save the details of the tests at this node
      m_testDetails = new HotTestDetails[m_children.length];
      for (int i = 0; i < m_children.length; i++) {
        m_testDetails[i] = newCandidates.get(i);
      }

      // save memory
      splitQueue = null;
      newCandidates = null;

      // process the children
      for (int i = 0; i < m_children.length; i++) {
        Instances subset = subset(insts, m_testDetails[i]);
        HotSpotHashKey tempKey = keyList.get(i);
        m_children[i] = new HotNode(subset, m_testDetails[i].m_merit,
          tempKey.m_splitValues, tempKey.m_testTypes, depth + 1, supportCount);
      }
    }

    /**
     * Create a subset of instances that correspond to the supplied test details
     *
     * @param insts the instances to create the subset from
     * @param test the details of the split
     */
    private Instances subset(Instances insts, HotTestDetails test) {
      Instances sub = new Instances(insts, insts.numInstances());
      for (int i = 0; i < insts.numInstances(); i++) {
        Instance temp = insts.instance(i);
        if (!temp.isMissing(test.m_splitAttIndex)) {
          if (insts.attribute(test.m_splitAttIndex).isNominal()) {
            if (temp.value(test.m_splitAttIndex) == test.m_splitValue) {
              sub.add(temp);
            }
          } else {
            if (test.m_lessThan) {
              if (temp.value(test.m_splitAttIndex) <= test.m_splitValue) {
                sub.add(temp);
              }
            } else {
              if (temp.value(test.m_splitAttIndex) > test.m_splitValue) {
                sub.add(temp);
              }
            }
          }
        }
      }
      sub.compactify();
      return sub;
    }

    /**
     * Evaluate a numeric attribute for a potential split
     *
     * @param attIndex the index of the attribute
     * @param pq the priority queue of candidtate splits
     * @param supportCount the support constraint (minimum number of instances)
     *                     by which to add candidate tests
     */
    private void
      evaluateNumeric(int attIndex, PriorityQueue<HotTestDetails> pq, int supportCount) {
      Instances tempInsts = m_insts;
      tempInsts.sort(attIndex);

      // target sums/counts
      double targetLeft = 0;
      double targetRight = 0;

      int numMissingAtt = 0;
      int numMissingTarget = 0;
      // count missing values and sum/counts for the initial right subset
      for (int i = tempInsts.numInstances() - 1; i >= 0; i--) {
        if (!tempInsts.instance(i).isMissing(attIndex)) {
          if (!tempInsts.instance(i).isMissing(m_target)) {
            targetRight +=
              (tempInsts.attribute(m_target).isNumeric()) ? (tempInsts
                .instance(i).value(m_target)) : ((tempInsts.instance(i).value(
                m_target) == m_targetIndex) ? 1 : 0);
          } else {
            numMissingTarget++;
          }
        } else {
          numMissingAtt++;
        }
      }

      // are there still enough instances?
      if (tempInsts.numInstances() - numMissingAtt - numMissingTarget <= supportCount) {
        return;
      }

      double bestMerit = 0.0;
      double bestSplit = 0.0;
      double bestSupport = 0.0;
      double bestSubsetSize = 0;
      boolean lessThan = true;

      // denominators
      double leftCount = 0;
      double rightCount = tempInsts.numInstances() - numMissingAtt - numMissingTarget;

      /*
       * targetRight = (tempInsts.attribute(m_target).isNumeric()) ?
       * tempInsts.attributeStats(m_target).numericStats.sum :
       * tempInsts.attributeStats(m_target).nominalCounts[m_targetIndex];
       */
      // targetRight =
      // tempInsts.attributeStats(attIndexnominalCounts[m_targetIndex];

      // consider all splits
      for (int i = 0; i < tempInsts.numInstances() - numMissingAtt; i++) {
        Instance inst = tempInsts.instance(i);

        if (!inst.isMissing(m_target)) {
          if (tempInsts.attribute(m_target).isNumeric()) {
            targetLeft += inst.value(m_target);
            targetRight -= inst.value(m_target);
          } else {
            if ((int) inst.value(m_target) == m_targetIndex) {
              targetLeft++;
              targetRight--;
            }
          }
          leftCount++;
          rightCount--;
        }

        // move to the end of any ties
        if (i < tempInsts.numInstances() - 1
          && inst.value(attIndex) == tempInsts.instance(i + 1)
          .value(attIndex)) {
          continue;
        }

        // evaluate split
        if (tempInsts.attribute(m_target).isNominal()) {
          if (targetLeft >= supportCount) {
            double delta =
              (m_minimize) ? (bestMerit - (targetLeft / leftCount))
                : ((targetLeft / leftCount) - bestMerit);
            // if (targetLeft / leftCount > bestMerit) {
            if (delta > 0) {
              bestMerit = targetLeft / leftCount;
              bestSplit = inst.value(attIndex);
              bestSupport = targetLeft;
              bestSubsetSize = leftCount;
              lessThan = true;
              // } else if (targetLeft / leftCount == bestMerit) {
            } else if (delta == 0) {
              // break ties in favour of higher support
              if (targetLeft > bestSupport) {
                bestMerit = targetLeft / leftCount;
                bestSplit = inst.value(attIndex);
                bestSupport = targetLeft;
                bestSubsetSize = leftCount;
                lessThan = true;
              }
            }
          }

          if (targetRight >= supportCount) {
            double delta =
              (m_minimize) ? (bestMerit - (targetRight / rightCount))
                : ((targetRight / rightCount) - bestMerit);
            // if (targetRight / rightCount > bestMerit) {
            if (delta > 0) {
              bestMerit = targetRight / rightCount;
              bestSplit = inst.value(attIndex);
              bestSupport = targetRight;
              bestSubsetSize = rightCount;
              lessThan = false;
              // } else if (targetRight / rightCount == bestMerit) {
            } else if (delta == 0) {
              // break ties in favour of higher support
              if (targetRight > bestSupport) {
                bestMerit = targetRight / rightCount;
                bestSplit = inst.value(attIndex);
                bestSupport = targetRight;
                bestSubsetSize = rightCount;
                lessThan = false;
              }
            }
          }
        } else {
          if (leftCount >= supportCount) {
            double delta =
              (m_minimize) ? (bestMerit - (targetLeft / leftCount))
                : ((targetLeft / leftCount) - bestMerit);
            // if (targetLeft / leftCount > bestMerit) {
            if (delta > 0) {
              bestMerit = targetLeft / leftCount;
              bestSplit = inst.value(attIndex);
              bestSupport = leftCount;
              bestSubsetSize = leftCount;
              lessThan = true;
              // } else if (targetLeft / leftCount == bestMerit) {
            } else if (delta == 0) {
              // break ties in favour of higher support
              if (leftCount > bestSupport) {
                bestMerit = targetLeft / leftCount;
                bestSplit = inst.value(attIndex);
                bestSupport = leftCount;
                bestSubsetSize = leftCount;
                lessThan = true;
              }
            }
          }

          if (rightCount >= supportCount) {
            double delta =
              (m_minimize) ? (bestMerit - (targetRight / rightCount))
                : ((targetRight / rightCount) - bestMerit);
            // if (targetRight / rightCount > bestMerit) {
            if (delta > 0) {
              bestMerit = targetRight / rightCount;
              bestSplit = inst.value(attIndex);
              bestSupport = rightCount;
              bestSubsetSize = rightCount;
              lessThan = false;
              // } else if (targetRight / rightCount == bestMerit) {
            } else if (delta == 0) {
              // break ties in favour of higher support
              if (rightCount > bestSupport) {
                bestMerit = targetRight / rightCount;
                bestSplit = inst.value(attIndex);
                bestSupport = rightCount;
                bestSubsetSize = rightCount;
                lessThan = false;
              }
            }
          }
        }
      }

      double delta = (m_minimize) ? m_targetValue - bestMerit : bestMerit
        - m_targetValue;

      // Have we found a candidate split?
      if (bestSupport > 0 && (delta / m_targetValue >= m_minImprovement)) {
        /*
         * System.err.println("Evaluating " +
         * tempInsts.attribute(attIndex).name()); System.err.println("Merit : "
         * + bestMerit); System.err.println("Support : " + bestSupport);
         */
        // double suppFraction = bestSupport / m_numInstances;

        HotTestDetails newD = new HotTestDetails(attIndex, bestSplit, lessThan,
          (int) bestSupport, (int) bestSubsetSize, bestMerit);
        pq.add(newD);
      }
    }

    /**
     * Evaluate a nominal attribute for a potential split
     *
     * @param attIndex the index of the attribute
     * @param pq the priority queue of candidtate splits
     * @param supportCount the support constraint (minimum number of instances)
     *                     by which to add candidate tests
     */
    private void
      evaluateNominal(int attIndex, PriorityQueue<HotTestDetails> pq, int supportCount) {
      if (m_insts.attribute(attIndex).numValues() < 2) {
        return;
      }
      /* int baseSupportCount = m_insts.attribute(m_target).isNumeric() && m_heuristic && (m_directSum || m_sumForNumericTarget)
        ? (int) Math.sqrt(supportCount) + 2 : supportCount; // use a heuristic to set the minimum segment size for sum trees */

      int[] counts = m_insts.attributeStats(attIndex).nominalCounts;
      boolean ok = false;
      // only consider attribute values that result in subsets that meet/exceed
      // min support
      int offset = (getTreatZeroAsMissing() ? 1 : 0);
      for (int i = 0 + offset; i < m_insts.attribute(attIndex).numValues(); i++) {
        if (counts[i] >= supportCount) {
          ok = true;
          break;
        }
      }

      if (ok) {
        double[] subsetMerit = new double[m_insts.attribute(attIndex)
          .numValues()];

        for (int i = 0; i < m_insts.numInstances(); i++) {
          Instance temp = m_insts.instance(i);
          boolean missingAtt =
            (temp.isMissing(attIndex) || (getTreatZeroAsMissing() ? (int) temp
              .value(attIndex) == 0 : false));
          // if (!temp.isMissing(attIndex) && !temp.isMissing(m_target)) {
          if (!missingAtt && !temp.isMissing(m_target)) {
            int attVal = (int) temp.value(attIndex);
            if (m_insts.attribute(m_target).isNumeric()) {
              subsetMerit[attVal] += temp.value(m_target);
            } else {
              subsetMerit[attVal] +=
                ((int) temp.value(m_target) == m_targetIndex) ? 1.0
                  : 0;
            }
          }
        }

        double expectedSumPerCategory = 0;
        if (m_sumForNumericTarget) {
          double sumOfMerits = Utils.sum(subsetMerit);
          expectedSumPerCategory = sumOfMerits / subsetMerit.length;
        }

        // add to queue if it meets min support and exceeds the merit for the
        // full set
        for (int i = 0; i < m_insts.attribute(attIndex).numValues(); i++) {
          // does the subset based on this value have enough instances, and,
          // furthermore,
          // does the target value (nominal only) occur enough times to exceed
          // min support
          if (counts[i] >= supportCount
            && ((m_insts.attribute(m_target).isNominal()) ? (subsetMerit[i] >= supportCount) // nominal
                                                                                               // only
                                                                                               // test
              : true)) {
            double merit = 0;
            if ((m_sumForNumericTarget || m_directSum) && m_insts.attribute(m_target).isNumeric()) {
              if (m_directSum) {
                merit = subsetMerit[i];
              } else {
                // should always be rows here as long as supportCount >= 1; however, need to check for sum (subsetMerit) == 0
                if (m_minimize) {
                  merit = subsetMerit[i] == 0 ? 0 : expectedSumPerCategory / subsetMerit[i];
                  if (merit < 1) {
                    // only interested in sums that are less than the expected (we may have already
                    // eliminated the ones that are because they don't have necessary support)
                    continue;
                  }
                } else {
                  merit = subsetMerit[i] / expectedSumPerCategory;
                  if (merit < 1) {
                    continue;
                  }
                }
              }
            } else {
              merit = subsetMerit[i] / counts[i];
            }

            /* double merit = m_sumForNumericTarget && m_insts.attribute(m_target).isNumeric()
              ? subsetMerit[i] / expectedSumPerCategory
              : subsetMerit[i] / counts[i]; // subsetMerit[i][1]; */
            double delta = 0;
            if ((m_sumForNumericTarget || m_directSum) && m_insts.attribute(m_target).isNumeric()) {

              delta = merit;

              double minImprovement = 0;
              if (m_sumForNumericTarget && !m_directSum) {
                double minIFrac = m_minImprovement;
                if (m_minimize && minIFrac >= 1) {
                  minIFrac = 0.99; // minimum improvement can't be more than a 99% reduction of the expected value
                }
                minImprovement = m_minimize ? 1.0 / (1.0 - minIFrac) : 1 + minIFrac;
              }

              // min improvement is ignored/makes no sense for absolute sum
              if (delta > minImprovement) {
                double support = counts[i];
                HotTestDetails newD =
                  new HotTestDetails(attIndex, i, (int) support, subsetMerit[i], !m_directSum && m_minimize ? -merit : merit);
                newD.m_distributionForNominalTest = subsetMerit;
                newD.m_countsForNominalTest = counts;
                if (!m_directSum) {
                  newD.m_expectedTargetSumPerCategory = expectedSumPerCategory;
                }
                pq.add(newD);
              }
            } else {
              delta =
                m_minimize ? m_targetValue - merit : merit - m_targetValue;

              if (delta / m_targetValue >= m_minImprovement) {
                double support = (m_insts.attribute(m_target).isNominal()) ?
                  subsetMerit[i] :
                  counts[i];
                HotTestDetails newD =
                  new HotTestDetails(attIndex, i, false, (int) support, counts[i], merit);
                newD.m_distributionForNominalTest = subsetMerit;
                pq.add(newD);
              }
            }
          }
        }
      }
    }

    public int assignIDs(int lastID) {
      int currentLastID = lastID + 1;
      m_id = currentLastID;
      if (m_children != null) {
        for (HotNode element : m_children) {
          currentLastID = element.assignIDs(currentLastID);
        }
      }
      return currentLastID;
    }

    private void addNodeDetails(StringBuffer buff, int i, String spacer) {
      buff.append(m_header.attribute(m_testDetails[i].m_splitAttIndex).name());
      if (m_header.attribute(m_testDetails[i].m_splitAttIndex).isNumeric()) {
        if (m_testDetails[i].m_lessThan) {
          buff.append(" <= ");
        } else {
          buff.append(" > ");
        }
        buff.append(Utils.doubleToString(m_testDetails[i].m_splitValue, 4));
      } else {
        buff.append(" = "
          + m_header.attribute(m_testDetails[i].m_splitAttIndex).value(
            (int) m_testDetails[i].m_splitValue));
      }

      if (m_header.attribute(m_target).isNumeric()) {
        buff.append(spacer + "(");
        if (m_sumForNumericTarget) {
          buff.append(Utils.doubleToString(m_testDetails[i].m_targetSum, 4));
          if (!m_directSum) {
            buff.append(" {").append(Utils.doubleToString(m_testDetails[i].m_expectedTargetSumPerCategory, 4)).append("}");
            // merit is the # x's relative to expected sum
            double merit = m_testDetails[i].m_merit < 0 ? -m_testDetails[i].m_merit : m_testDetails[i].m_merit;
            if (m_minimize) {
              merit = 1.0 / merit;
            }
            buff.append(" : ")
              .append(Utils.doubleToString(merit, 2)).append("x ");
          }
        } else {
          // here merit *is* the average target value
          buff.append(Utils.doubleToString(m_testDetails[i].m_merit, 4));
          double xsComparedToNode = m_testDetails[i].m_merit / m_targetValue;
          buff.append(" : ").append(Utils.doubleToString(xsComparedToNode, 2)).append("x");
        }
        buff.append(" [").append(
          m_testDetails[i].m_supportLevel).append("])");
      } else {
        buff.append(spacer + "("
          + Utils.doubleToString((m_testDetails[i].m_merit * 100.0), 2) + "% ["
          + m_testDetails[i].m_supportLevel + "/"
          + m_testDetails[i].m_subsetSize + "])");
      }
    }

    private void graphHotSpot(StringBuffer text) {
      if (m_children != null) {
        for (int i = 0; i < m_children.length; i++) {
          text.append("N" + m_children[i].m_id);
          text.append(" [label=\"");
          addNodeDetails(text, i, "\\n");
          text.append("\" shape=plaintext]\n");
          m_children[i].graphHotSpot(text);
          text.append("N" + m_id + "->" + "N" + m_children[i].m_id + "\n");
        }
      }
    }

    protected void addNodeDetails(Map<String, Object> node, int i) {
      node.put("id", m_id);
      node.put("name", m_header.attribute(m_testDetails[i].m_splitAttIndex).name());
      boolean isNumeric = m_header.attribute(m_testDetails[i].m_splitAttIndex).isNumeric();
      boolean targetIsNominal = m_header.attribute(m_targetIndex).isNominal();
      node.put("isNumeric", isNumeric);
      String comparison = "";
      Object value = null;
      double merit = 0;
      double support = 0;
      double numericTargetSum = 0;
      int subsetSize = m_testDetails[i].m_subsetSize;
      if (isNumeric) {
        if (m_testDetails[i].m_lessThan) {
          comparison = "<=";
        } else {
          comparison = ">";
        }
        value = m_testDetails[i].m_splitValue;
        merit = m_testDetails[i].m_merit * (targetIsNominal ? 100.0 : 1.0);
        if (targetIsNominal) {
          support = m_testDetails[i].m_supportLevel;
        }
      } else {
        comparison = "=";
        value = m_header.attribute(m_testDetails[i].m_splitAttIndex).value(
          (int) m_testDetails[i].m_splitValue);

        merit = m_testDetails[i].m_merit * (targetIsNominal ? 100.0 : 1.0);
        if (targetIsNominal) {
          support = m_testDetails[i].m_supportLevel;
        }
        if (!targetIsNominal && (m_sumForNumericTarget || m_directSum)) {
          numericTargetSum = m_testDetails[i].m_targetSum;
        }
      }
      node.put("comparison", comparison);
      node.put("value", value);
      node.put("target", !targetIsNominal && (m_sumForNumericTarget || m_directSum) ? numericTargetSum : merit); // target value

      if (m_minimize && !targetIsNominal && (m_sumForNumericTarget || m_directSum)
        && merit < 0) { // reverse the sign used if minimising in the sum cases
        merit = -merit;
      }

      if (!targetIsNominal && !m_directSum) {
        if (m_sumForNumericTarget) {
          // in this case, merit holds the number of x's that this node's category is above the expected sum
          // or the number of x's the expected sum is above this node's category (if minimizing)
          if (m_minimize) {
            merit = 1.0 / merit;
          }
          node.put("expectedSum", m_testDetails[i].m_expectedTargetSumPerCategory);
        } else {
          // here the merit *is* the average of the target value - need to compute x's
          merit = m_testDetails[i].m_merit / m_targetValue;
        }
        node.put("merit", merit);
      }
      node.put("subsetSize", subsetSize);
      if (support > 0) {
        node.put("support", support); // nominal target value only
      }
      if (m_testDetails[i].m_distributionForNominalTest != null) {
        node.put("distribution", m_testDetails[i].m_distributionForNominalTest);
        node.put("counts", m_testDetails[i].m_countsForNominalTest);
      }
    }

    protected void graphAsMap(Map<String, Object> graph) {
      // details of this node
      if (m_children != null && m_children.length > 0) {
        List<Map<String, Object>> childList = new ArrayList<Map<String, Object>>();
        graph.put("children", childList);
        for (int i = 0; i < m_children.length; i++) {
          Map<String, Object> childNode = new LinkedHashMap<String, Object>();
          childList.add(childNode);
          addNodeDetails(childNode, i);
          m_children[i].graphAsMap(childNode);
        }
      }
    }

    /**
     * Traverse the tree to create a string description
     *
     * @param depth the depth at this point in the tree
     * @param buff the string buffer to append node details to
     */
    protected void dumpTree(int depth, StringBuffer buff) {
      if (m_children == null) {
        // buff.append("\n");
      } else {
        for (int i = 0; i < m_children.length; i++) {
          buff.append("\n  ");
          for (int j = 0; j < depth; j++) {
            buff.append("|   ");
          }
          addNodeDetails(buff, i, " ");

          m_children[i].dumpTree(depth + 1, buff);
        }
      }
    }

    private void addTestToRule(List<Item> currentPremise, int i)
      throws Exception {
      if (m_header.attribute(m_testDetails[i].m_splitAttIndex).isNumeric()) {
        NumericItem.Comparison comp =
          (m_testDetails[i].m_lessThan) ? NumericItem.Comparison.LESS_THAN_OR_EQUAL_TO
            : NumericItem.Comparison.GREATER_THAN;

        NumericItem newItem = new NumericItem(
          m_header.attribute(m_testDetails[i].m_splitAttIndex),
          m_testDetails[i].m_splitValue, comp);

        currentPremise.add(newItem);
      } else {
        NominalItem newItem = new NominalItem(
          m_header.attribute(m_testDetails[i].m_splitAttIndex),
          (int) m_testDetails[i].m_splitValue);
        currentPremise.add(newItem);
      }
    }

    private class HotSpotNumericTargetRule extends AssociationRule implements
      Serializable {

      private static final long serialVersionUID = -1028053590504776204L;

      Collection<Item> m_premise;
      Collection<Item> m_consequence;
      boolean m_numericTarget = true;
      int m_totalSupport;
      int m_consequenceSupport;
      int m_totalTransactions;

      double m_averageTarget;

      DefaultAssociationRule m_delegateForDiscreteTarget;

      public HotSpotNumericTargetRule(Collection<Item> premise,
        Collection<Item> consequence, int totalSupport, int consequenceSupport,
        int totalTransactions, double averageTarget) {
        m_premise = premise;
        m_consequence = consequence;
        m_totalSupport = totalSupport;
        m_consequenceSupport = consequenceSupport;
        m_totalTransactions = totalTransactions;
        m_averageTarget = averageTarget;
      }

      public HotSpotNumericTargetRule(Collection<Item> premise,
        Collection<Item> consequence, int premiseSupport,
        int consequenceSupport, int totalSupport, int totalTransactions,
        double averageTarget) {

        m_numericTarget = false;
        m_premise = premise;
        m_consequence = consequence;
        if (m_numericTarget) {
          m_totalSupport = totalSupport;
          m_consequenceSupport = consequenceSupport;
          m_totalTransactions = totalTransactions;
          m_averageTarget = averageTarget;
        } else {
          m_delegateForDiscreteTarget =
            new DefaultAssociationRule(premise,
              consequence, DefaultAssociationRule.METRIC_TYPE.CONFIDENCE,
              premiseSupport, consequenceSupport, totalSupport,
              totalTransactions);
        }
      }

      @Override
      public Collection<Item> getPremise() {
        return m_premise;
      }

      @Override
      public Collection<Item> getConsequence() {
        return m_consequence;
      }

      @Override
      public String getPrimaryMetricName() {
        return (m_numericTarget) ? "AverageTarget"
          : m_delegateForDiscreteTarget.getPrimaryMetricName();
      }

      @Override
      public double getPrimaryMetricValue() {
        return (m_numericTarget) ? m_averageTarget
          : m_delegateForDiscreteTarget.getPrimaryMetricValue();
      }

      @Override
      public double getNamedMetricValue(String metricName) throws Exception {
        if (m_numericTarget) {
          if (metricName.equals("AverageTarget")) {
            return getPrimaryMetricValue();
          }
          return Utils.missingValue();
        } else {
          if (metricName.equals("AverageTarget")) {
            return Utils.missingValue();
          }
          return m_delegateForDiscreteTarget.getNamedMetricValue(metricName);
        }
      }

      @Override
      public int getNumberOfMetricsForRule() {
        return DefaultAssociationRule.METRIC_TYPE.values().length + 1;
      }

      @Override
      public String[] getMetricNamesForRule() {
        String[] result = new String[getNumberOfMetricsForRule()];
        result[0] = "AverageTarget";
        for (int i = 0; i < DefaultAssociationRule.TAGS_SELECTION.length; i++) {
          result[i + 1] = DefaultAssociationRule.TAGS_SELECTION[i]
            .getReadable();
        }
        return result;
      }

      @Override
      public double[] getMetricValuesForRule() throws Exception {
        double[] result = new double[getNumberOfMetricsForRule()];
        result[0] = (m_numericTarget) ? getPrimaryMetricValue() : Utils
          .missingValue();

        for (int i = 0; i < DefaultAssociationRule.TAGS_SELECTION.length; i++) {
          if (m_numericTarget) {
            result[i + 1] = Utils.missingValue();
          } else {
            result[i + 1] = m_delegateForDiscreteTarget
              .getNamedMetricValue(DefaultAssociationRule.TAGS_SELECTION[i]
                .getReadable());
          }
        }

        return result;
      }

      @Override
      public int getPremiseSupport() {
        return (m_numericTarget) ? m_totalSupport : m_delegateForDiscreteTarget
          .getPremiseSupport();
      }

      @Override
      public int getConsequenceSupport() {
        return (m_numericTarget) ? m_consequenceSupport
          : m_delegateForDiscreteTarget.getConsequenceSupport();
      }

      @Override
      public int getTotalSupport() {
        return (m_numericTarget) ? m_totalSupport : m_delegateForDiscreteTarget
          .getTotalSupport();
      }

      @Override
      public int getTotalTransactions() {
        return (m_numericTarget) ? m_totalTransactions
          : m_delegateForDiscreteTarget.getTotalTransactions();
      }

      @Override
      public String toString() {
        StringBuffer result = new StringBuffer();

        if (m_numericTarget) {
          result.append(m_premise.toString() + " ==> "
            + m_consequence.toString() + ": " + m_totalSupport + "   ");
        } else {
          result.append(m_delegateForDiscreteTarget.toString());
        }
        return result.toString();
      }

      @Override
      public int compareTo(AssociationRule other) {
        int result = super.compareTo(other);
        if (m_minimize) {
          result = -result;
        }
        return result;
      }
    }

    protected void getRules(List<AssociationRule> rules,
      ArrayList<Item> currentPremise) throws Exception {
      if (m_children == null) {

      } else {
        for (int i = 0; i < m_children.length; i++) {

          // first clone the current rule
          @SuppressWarnings("unchecked")
          ArrayList<Item> newPremise = (ArrayList<Item>) currentPremise.clone();

          // add the child details
          addTestToRule(newPremise, i);

          if (m_header.attribute(m_target).isNominal()) {
            NominalItem consequenceItem = new NominalItem(
              m_header.attribute(m_target), m_targetIndex);
            List<Item> consequence = new ArrayList<Item>();
            consequence.add(consequenceItem);

            HotSpotNumericTargetRule newRule = new HotSpotNumericTargetRule(
              newPremise, consequence, m_testDetails[i].m_subsetSize,
              m_globalSupport, m_testDetails[i].m_supportLevel, m_numInstances,
              Utils.missingValue());

            // add the rule to the list
            rules.add(newRule);

          } else {
            NumericItem consequenceItem = new NumericItem(
              m_header.attribute(m_target), m_testDetails[i].m_merit,
              NumericItem.Comparison.NONE);
            List<Item> consequence = new ArrayList<Item>();
            consequence.add(consequenceItem);

            HotSpotNumericTargetRule newRule = new HotSpotNumericTargetRule(
              newPremise, consequence, m_testDetails[i].m_supportLevel,
              m_numNonMissingTarget, m_numInstances, m_testDetails[i].m_merit);

            rules.add(newRule);
          }

          // recurse
          m_children[i].getRules(rules, newPremise);
        }
      }
    }
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String targetTipText() {
    return "The target attribute of interest (\"first\", \"last\","
      + "<index> or <attribute name> are valid values).";
  }

  /**
   * Set the target index
   *
   * @param target the target index as a string (1-based)
   */
  public void setTarget(String target) {
    m_targetSI.setSingleIndex(target);
  }

  /**
   * Get the target index as a string
   *
   * @return the target index (1-based)
   */
  public String getTarget() {
    return m_targetSI.getSingleIndex();
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String targetIndexTipText() {
    return "The value of the target (nominal attributes only) of interest.";
  }

  /**
   * For a nominal target, set the index of the value of interest (1-based)
   *
   * @param index the index of the nominal value of interest
   */
  public void setTargetIndex(String index) {
    m_targetIndexSI.setSingleIndex(index);
  }

  /**
   * For a nominal target, get the index of the value of interest (1-based)
   *
   * @return the index of the nominal value of interest
   */
  public String getTargetIndex() {
    return m_targetIndexSI.getSingleIndex();
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String minimizeTargetTipText() {
    return "Minimize rather than maximize the target.";
  }

  /**
   * Set whether to minimize the target rather than maximize
   *
   * @param m true if target is to be minimized
   */
  public void setMinimizeTarget(boolean m) {
    m_minimize = m;
  }

  /**
   * Get whether to minimize the target rather than maximize
   *
   * @return true if target is to be minimized
   */
  public boolean getMinimizeTarget() {
    return m_minimize;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String useSumForNumericTargetTipText() {
    return "For a numeric target use the sum of the target (relative to the expected sum) "
      + "as the aggregation rather than the average. In this mode, only nominal attributes "
      + "are considered for splitting in the tree. The metric favours categorical values "
      + "where the sum of the target, within the category, is markedly higher (or lower, "
      + "if minimizing) than the expected sum.";
  }

  /**
   * Set whether to use sum of target (relative to expected sum) as the aggregation
   * rather than average for a numeric target. In this mode, only nominal attributes
   * are considered for the splits of the tree. The metric finds categorical values where
   * the sum of the target, within the category, is markedly higher (or lower, if minimizing)
   * than the expected sum.
   *
   * @param relativeSumForNumericTarget true to use relative sum as the aggregation
   */
  public void setUseSumForNumericTarget(boolean relativeSumForNumericTarget) {
    m_sumForNumericTarget = relativeSumForNumericTarget;
  }

  /**
   * Get whether to use sum of target (relative to expected sum) as the aggregation
   * rather than average for a numeric target. In this mode, only nominal attributes
   * are considered for the splits of the tree. The metric finds categorical values where
   * the sum of the target, within the category, is markedly higher (or lower, if minimizing)
   * than the expected sum.
   *
   * @return true if using relative sum as the aggregation
   */
  public boolean getUseSumForNumericTarget() {
    return m_sumForNumericTarget;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String useDirectRatherThanRelativeSumForNumericTargetTipText() {
    return "Look for nominal categories with directly high/low concentration/volume of sum "
      + "of target, rather than sum relative to the expected sum across categories. Only "
      + "has an effect when using sum (rather than average) as the aggregation for a numeric "
      + "target";
  }

  /**
   * Look for nominal categories with directly high/low concentration/volume of sum
   * of the target, rather than sum relative to the expected sum across categories. Only has
   * an effect when using sum (rather than average) as the aggregation of a numreic target
   *
   * @param useAbsolute use the direct, rather than relative, sum as the goodness metric
   */
  public void setUseDirectRatherThanRelativeSumForNumericTarget(boolean useAbsolute) {
    m_directSum = useAbsolute;
  }

  /**
   * Look for nominal categories with directly high/low concentration/volume of sum
   * of the target, rather than sum relative to the expected sum across categories. Only has
   * an effect when using sum (rather than average) as the aggregation of a numreic target
   *
   * @return true if using the direct, rather than relative, sum as the goodness metric
   */
  public boolean getUseDirectRatherThanRelativeSumForNumericTarget() {
    return m_directSum;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String supportTipText() {
    return "The minimum support. Values between 0 and 1 are interpreted "
      + "as a percentage of the total population; values > 1 are "
      + "interpreted as an absolute number of instances";
  }

  /**
   * Get the minimum support
   *
   * @return the minimum support
   */
  public String getSupport() {
    return m_supportString;
  }

  /**
   * Set the minimum support
   *
   * @param s the minimum support
   */
  public void setSupport(String s) {
    m_supportString = s;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String maxBranchingFactorTipText() {
    return "Maximum branching factor. The maximum number of children "
      + "to consider extending each node with.";
  }

  /**
   * Set the maximum branching factor
   *
   * @param b the maximum branching factor
   */
  public void setMaxBranchingFactor(int b) {
    m_maxBranchingFactor = b;
  }

  /**
   * Get the maximum branching factor
   *
   * @return the maximum branching factor
   */
  public int getMaxBranchingFactor() {
    return m_maxBranchingFactor;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String maxRuleLengthTipText() {
    return "Bound the length of a rule/path in the tree. "
      + "-1 means unbounded";
  }

  /**
   * Set the maximum rule length
   *
   * @param l the maximum rule length
   */
  public void setMaxRuleLength(int l) {
    m_maxRuleLength = l;
  }

  /**
   * Get the maximum rule length
   *
   * @return the maximum rule length
   */
  public int getMaxRuleLength() {
    return m_maxRuleLength;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String forceMaxRuleLengthTipText() {
    return "Ignore min value count setting and try to force the maximum rule "
      + "length";
  }

  /**
   * Set whether to try and force the max rule length by ignoring the
   * minimum value count (support) setting
   *
   * @param force true to force maximum rule length
   */
  public void setForceMaxRuleLength(boolean force) {
    m_forceMaxRuleLength = force;
  }

  /**
   * Get whether to try and force the max rule length by ignoring the
   * minimum value count (support) setting
   *
   * @return true to force maximum rule length
   */
  public boolean getForceMaxRuleLength() {
    return m_forceMaxRuleLength;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String useHeuristicMinSegmentSizeTipText() {
    return "Set the minimum segment size/support for direct and relative sum "
      + "trees via a heuristic. Overrides any manually specified support.";
  }

  /**
   * Set whether to use a heuristic to set the minimum support for direct and
   * relative sum trees
   *
   * @param useHeuristicMinSegmentSize true to use the heuristic
   */
  public void setUseHeuristicMinSegmentSize(boolean useHeuristicMinSegmentSize) {
    m_heuristic = useHeuristicMinSegmentSize;
  }

  /**
   * Get whether to use a heuristic to set the minimum support for direct and
   * relative sum trees
   *
   * @return true if using the heuristic
   */
  public boolean getUseHeuristicMinSegmentSize() {
    return m_heuristic;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String treatZeroAsMissingTipText() {
    return "Treat zero (first value) for nominal attributes "
      + "the same way as missing value (i.e. ignore). This is useful "
      + "for market basket data.";
  }

  /**
   * Set whether to treat zero as missing.
   *
   * @param t true if zero (first value) for nominal attributes is to be treated
   *          like missing value.
   */
  public void setTreatZeroAsMissing(boolean t) {
    m_treatZeroAsMissing = t;
  }

  /**
   * Get whether to treat zero as missing.
   *
   * @return true if zero (first value) for nominal attributes is to be treated
   *         like missing value.
   */
  public boolean getTreatZeroAsMissing() {
    return m_treatZeroAsMissing;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String minImprovementTipText() {
    return "Minimum improvement in target value in order to "
      + "consider adding a new branch/test";
  }

  /**
   * Set the minimum improvement in the target necessary to add a test
   *
   * @param i the minimum improvement
   */
  public void setMinImprovement(double i) {
    m_minImprovement = i;
  }

  /**
   * Get the minimum improvement in the target necessary to add a test
   *
   * @return the minimum improvement
   */
  public double getMinImprovement() {
    return m_minImprovement;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String debugTipText() {
    return "Output debugging info (duplicate rule lookup hash table stats).";
  }

  /**
   * Set whether debugging info is output
   *
   * @param d true to output debugging info
   */
  public void setDebug(boolean d) {
    m_debug = d;
  }

  /**
   * Get whether debugging info is output
   *
   * @return true if outputing debugging info
   */
  public boolean getDebug() {
    return m_debug;
  }

  public void setOutputRules(boolean r) {
    m_outputRules = r;
  }

  public boolean getOutputRules() {
    return m_outputRules;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String outputRulesTipText() {
    return "Output a rule set instead of a tree";
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String doNotCheckCapabilitiesTipText() {
    return "If set, associator capabilities are not checked before associator is built"
      + " (Use with caution to reduce runtime).";
  }

  /**
   * Set whether capabilities checking is turned off.
   *
   * @param doNotCheck true if capabilities checking is turned off.
   */
  @Override
  public void setDoNotCheckCapabilities(boolean doNotCheck) {
    m_doNotCheckCapabilities = doNotCheck;
  }

  /**
   * Get whether capabilities checking is turned off.
   *
   * @return true if capabilities checking is turned off.
   */
  @Override
  public boolean getDoNotCheckCapabilities() {
    return m_doNotCheckCapabilities;
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {
    Vector<Option> newVector = new Vector<Option>();
    newVector.addElement(new Option("\tThe target index. (default = last)",
      "c", 1, "-c <num | first | last | attribute name>"));
    newVector.addElement(new Option(
      "\tThe target value (nominal target only, default = first)", "V", 1,
      "-V <num | first | last>"));
    newVector.addElement(new Option("\tMinimize rather than maximize.", "L", 0,
      "-L"));
    newVector.addElement(new Option(
      "\tMinimum value count (nominal target)/segment size "
        + "(numeric target)." + "\n\tValues between 0 and 1 are "
        + "\n\tinterpreted as a percentage of "
        + "\n\tthe total population (numeric) or total target value\n\t"
        + "population size (nominal); values > 1 are "
        + "\n\tinterpreted as an absolute number of "
        + "\n\tinstances (default = 0.3)", "S", 1, "-S <num>"));
    newVector.addElement(new Option("\tUse a heuristic to set the minimum segment size "
      + "\n\tfor numeric target direct and relative sum trees.", "heuristic", 0,
      "-heuristic"));
    newVector.addElement(new Option("\tMaximum branching factor (default = 2)",
      "M", 1, "-M <num>"));
    newVector.addElement(new Option(
      "\tMaximum rule length (default = 3)", "length", 1,
      "-length <num>"));
    newVector.addElement(new Option("\tForce maximum rule length (i.e. ignore minimum value count)",
      "force", 0, "-force"));
    newVector.addElement(new Option(
      "\tOperate on sum, rather than average, for numeric target. Note, this mode\n\t"
        + "can only operate on nominal attributes.", "sum", 0, "-sum"));
    newVector.addElement(new Option(
      "\tConsider direct max/min sum of target rather than sum relative to that expected across\n\t"
        + "categories of a nominal attribute. Use in conjunction with -sum.",
      "direct", 0, "-direct"));
    newVector.addElement(new Option(
      "\tMinimum improvement in target value in order "
        + "\n\tto add a new branch/test (default = 0.01 (1%))", "I", 1,
      "-I <num>"));
    newVector.addElement(new Option(
      "\tTreat zero (first value) as missing for nominal attributes", "Z", 0,
      "-Z"));
    newVector.addElement(new Option(
      "\tOutput a set of rules instead of a tree structure", "R", 0, "-R"));
    newVector.addElement(new Option(
      "\tOutput debugging info (duplicate rule lookup "
        + "\n\thash table stats)", "D", 0, "-D"));
    return newVector.elements();
  }

  /**
   * Reset options to their defaults
   */
  public void resetOptions() {
    m_support = 0.33;
    m_supportString = "0.33";
    m_minImprovement = 0.01; // 1%
    m_maxBranchingFactor = 2;
    m_maxRuleLength = 3;
    m_minimize = false;
    m_debug = false;
    m_outputRules = false;
    setTarget("last");
    setTargetIndex("first");
    m_errorMessage = null;
    m_forceMaxRuleLength = false;
    m_heuristic = false;
  }

  /**
   * Parses a given list of options.
   * <p/>
   *
   <!-- options-start -->
   * Valid options are: <p>
   * 
   * <pre> -c &lt;num | first | last | attribute name&gt;
   *  The target index. (default = last)</pre>
   * 
   * <pre> -V &lt;num | first | last&gt;
   *  The target value (nominal target only, default = first)</pre>
   * 
   * <pre> -L
   *  Minimize rather than maximize.</pre>
   * 
   * <pre> -S &lt;num&gt;
   *  Minimum value count (nominal target)/segment size (numeric target).
   *  Values between 0 and 1 are 
   *  interpreted as a percentage of 
   *  the total population (numeric) or total target value
   *  population size (nominal); values &gt; 1 are 
   *  interpreted as an absolute number of 
   *  instances (default = 0.3)</pre>
   * 
   * <pre> -M &lt;num&gt;
   *  Maximum branching factor (default = 2)</pre>
   * 
   * <pre> -length &lt;num&gt;
   *  Maximum rule length (default = -1, i.e. no maximum)</pre>
   * 
   * <pre> -sum
   *  Operate on sum, rather than average, for numeric target. Note, this mode
   *  can only operate on nominal attributes.</pre>
   * 
   * <pre> -direct
   *  Consider direct max/min sum of target rather than sum relative to that expected across
   *  categories of a nominal attribute. Use in conjunction with -sum.</pre>
   * 
   * <pre> -I &lt;num&gt;
   *  Minimum improvement in target value in order 
   *  to add a new branch/test (default = 0.01 (1%))</pre>
   * 
   * <pre> -Z
   *  Treat zero (first value) as missing for nominal attributes</pre>
   * 
   * <pre> -R
   *  Output a set of rules instead of a tree structure</pre>
   * 
   * <pre> -D
   *  Output debugging info (duplicate rule lookup 
   *  hash table stats)</pre>
   * 
   <!-- options-end -->
   * 
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {
    resetOptions();

    String tempString = Utils.getOption('c', options);
    if (tempString.length() != 0) {
      setTarget(tempString);
    }

    tempString = Utils.getOption('V', options);
    if (tempString.length() != 0) {
      setTargetIndex(tempString);
    }

    setMinimizeTarget(Utils.getFlag('L', options));

    tempString = Utils.getOption('S', options);
    if (tempString.length() != 0) {
      setSupport(tempString);
    }

    tempString = Utils.getOption('M', options);
    if (tempString.length() != 0) {
      setMaxBranchingFactor(Integer.parseInt(tempString));
    }

    tempString = Utils.getOption("length", options);
    if (tempString.length() > 0) {
      setMaxRuleLength(Integer.parseInt(tempString));
    }

    tempString = Utils.getOption('I', options);
    if (tempString.length() != 0) {
      setMinImprovement(Double.parseDouble(tempString));
    }

    setDebug(Utils.getFlag('D', options));
    setOutputRules(Utils.getFlag('R', options));
    setTreatZeroAsMissing(Utils.getFlag('Z', options));
    setUseSumForNumericTarget(Utils.getFlag("sum", options));
    setUseDirectRatherThanRelativeSumForNumericTarget(Utils.getFlag("direct", options));
    setForceMaxRuleLength(Utils.getFlag("force", options));
    setUseHeuristicMinSegmentSize(Utils.getFlag("heuristic", options));
  }

  /**
   * Gets the current settings of HotSpot.
   * 
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String[] getOptions() {
    String[] options = new String[20];
    int current = 0;

    options[current++] = "-c";
    options[current++] = getTarget();
    options[current++] = "-V";
    options[current++] = getTargetIndex();
    if (getMinimizeTarget()) {
      options[current++] = "-L";
    }
    options[current++] = "-S";
    options[current++] = "" + getSupport();
    options[current++] = "-M";
    options[current++] = "" + getMaxBranchingFactor();
    options[current++] = "-length";
    options[current++] = "" + getMaxRuleLength();
    options[current++] = "-I";
    options[current++] = "" + getMinImprovement();
    if (getDebug()) {
      options[current++] = "-D";
    }

    if (getOutputRules()) {
      options[current++] = "-R";
    }

    if (getTreatZeroAsMissing()) {
      options[current++] = "-Z";
    }

    if (getUseSumForNumericTarget()) {
      options[current++] = "-sum";
    }

    if (getUseDirectRatherThanRelativeSumForNumericTarget()) {
      options[current++] = "-direct";
    }

    if (getForceMaxRuleLength()) {
      options[current++] = "-force";
    }

    if (getUseHeuristicMinSegmentSize()) {
      options[current++] = "-heuristic";
    }

    while (current < options.length) {
      options[current++] = "";
    }

    return options;
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
   * Returns the type of graph this scheme represents.
   * 
   * @return Drawable.TREE
   */
  @Override
  public int graphType() {
    return Drawable.TREE;
  }

  /**
   * Gets the list of mined association rules.
   * 
   * @return the list of association rules discovered during mining. Returns
   *         null if mining hasn't been performed yet.
   */
  @Override
  public AssociationRules getAssociationRules() {
    List<AssociationRule> rulesToReturn = new ArrayList<AssociationRule>();
    try {
      m_head.getRules(rulesToReturn, new ArrayList<Item>());
      Collections.sort(rulesToReturn);
    } catch (Exception e) {
      e.printStackTrace();
    }

    return new AssociationRules(rulesToReturn, this);
  }

  /**
   * Returns true if this AssociationRulesProducer can actually produce rules.
   * Most implementing classes will always return true from this method
   * (obviously :-)). However, an implementing class that actually acts as a
   * wrapper around things that may or may not implement
   * AssociationRulesProducer will want to return false if the thing they wrap
   * can't produce rules.
   * 
   * @return true if this producer can produce rules in its current
   *         configuration
   */
  @Override
  public boolean canProduceRules() {
    return true;
  }

  /**
   * Gets a list of the names of the metrics output for each rule. This list
   * should be the same (in terms of the names and order thereof) as that
   * produced by AssociationRule.getMetricNamesForRule().
   * 
   * @return an array of the names of the metrics available for each rule
   *         learned by this producer.
   */
  @Override
  public String[] getRuleMetricNames() {
    String[] metricNames =
      new String[DefaultAssociationRule.TAGS_SELECTION.length + 1];
    metricNames[0] = "AverageTarget";

    for (int i = 0; i < DefaultAssociationRule.TAGS_SELECTION.length; i++) {
      metricNames[i + 1] = DefaultAssociationRule.TAGS_SELECTION[i]
        .getReadable();
    }

    return metricNames;
  }

  /**
   * Main method for testing this class.
   * 
   * @param args the options
   */
  public static void main(String[] args) {
    try {
      AbstractAssociator.runAssociator(new HotSpot(), args);
    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }
}

