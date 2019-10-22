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
 *    KerasZooClassifier.java
 *    Copyright (C) 2019 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.keras;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.rules.ZeroR;
import weka.core.Attribute;
import weka.core.BatchPredictor;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Environment;
import weka.core.EnvironmentHandler;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.LogHandler;
import weka.core.OptionMetadata;
import weka.core.Utils;
import weka.core.WekaException;
import weka.core.converters.CSVLoader;
import weka.gui.FilePropertyMetadata;
import weka.gui.Logger;
import weka.gui.knowledgeflow.KFGUIConsts;
import weka.python.PythonSession;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

/**
 <!-- globalinfo-start -->
 * Wrapper classifier for Keras zoo models.
 * <br><br>
 <!-- globalinfo-end -->
 *
 <!-- options-start -->
 * Valid options are: <p>
 * 
 * <pre> -seed &lt;integer&gt;
 *  Random seed</pre>
 * 
 * <pre> -zoo [Xception | VGG16 | VGG19 | ResNet | ResNetV2 | ResNeXt | InceptionV3 | InceptionResNetV2 | MobileNet | MobileNetV2 | DenseNet | NASNet]
 *  The zoo model to use</pre>
 * 
 * <pre> -weights [imagenet | None]
 *  Pre-trained or not</pre>
 * 
 * <pre> -epochs &lt;interger&gt;
 *  Number of epochs to perform</pre>
 * 
 * <pre> -max-queue &lt;integer&gt;
 *  Integer: Maximum size for the image generator queue.</pre>
 * 
 * <pre> -workers &lt;integer&gt;
 *  Number of workers to use for populating the generator queue</pre>
 * 
 * <pre> -optimizer [SGD | RMSProp | Adam | Adamax | Nadam | Adagrad | Adadelta]
 *  Optimizer to use when training full network from scratch, orthe top dense layers when transfer learning using pre-trained weights.</pre>
 * 
 * <pre> -optimizer-opts &lt;string&gt;
 *  Options for the selected optimizer (comma separated)</pre>
 * 
 * <pre> -top-5
 *  Output the top-5 accuracy when iterating</pre>
 * 
 * <pre> -fcl &lt;comma separated list of layer sizes&gt;
 *  Number (and size) of fully connected layers as a comma-separated list to add to the end of the network</pre>
 * 
 * <pre> -gap
 *  Add a GAP layer before fully connected layers (replaces AP layer if set)</pre>
 * 
 * <pre> -ap
 *  Add an AP layer before fully connected layers</pre>
 * 
 * <pre> -pool-size &lt;int,int&gt;
 *  Pool size (comma separated)</pre>
 * 
 * <pre> -dropout
 *  Whether to add a Dropout layer after each fully connected connected layer</pre>
 * 
 * <pre> -dropout-rate &lt;number&gt;
 *  Rate (between 0 and 1) for dropout</pre>
 * 
 * <pre> -finetune
 *  Whether to fine-tune top convolutional layers</pre>
 * 
 * <pre> -continue
 *  Continue training using network/weights loaded from model file (don't set this if performing a cross-validation)</pre>
 * 
 * <pre> -print-layer-indexes
 *  Print this info instead of model summary. This can help in deciding which layers to fine tune.</pre>
 * 
 * <pre> -initial-epoch &lt;integer&gt;
 *  When continuing training, start from this epoch; num epochs is added to this to determine the final epoch number</pre>
 * 
 * <pre> -load &lt;path to .hdf5 file&gt;
 *  Path to load network structure and weights from (i.e. to continue training an existing model)</pre>
 * 
 * <pre> -finetune-index &lt;index&gt;
 *  Fine-tune all top layers above and including this layer</pre>
 * 
 * <pre> -save &lt;path to .hdf5 file&gt;
 *  Path to save the final trained network (structure and weights) to</pre>
 * 
 * <pre> -finetune-optimizer [SGD | RMSProp | Adam | Adamax | Nadam | Adagrad | Adadelta]
 *  Optimizer to use for fine-tuning the network</pre>
 * 
 * <pre> -log &lt;log file path&gt;
 *  File to write training performance to while iterating</pre>
 * 
 * <pre> -dont-use-model-specific
 *  Don't apply the preprocess_input function associated with the selected zoo model. This function is applied in conjunction with manually specified image processing options.</pre>
 * 
 * <pre> -finetune-optimizer-opts &lt;string&gt;
 *  Options for the fine-tune optimizer</pre>
 * 
 * <pre> -images &lt;directory&gt;
 *  Directory containing images</pre>
 * 
 * <pre> -finetune-use-lr-schedule
 *  Override the learning rate in the fine-tune optimizer options with those set by the LR schedule callback (if defined)</pre>
 * 
 * <pre> -validation-split &lt;number between 0 and 1&gt;
 *  Amount of training data/images to use for validation as a fraction between 0 and 1. 0 = no validation split.</pre>
 * 
 * <pre> -validation-data &lt;path to arff or csv file&gt;
 *  Separate file for validation (arff or csv)</pre>
 * 
 * <pre> -mini-batch &lt;integer&gt;
 *  Size of the mini batches to train with</pre>
 * 
 * <pre> -width &lt;integer&gt;
 *  The target width of the images</pre>
 * 
 * <pre> -height &lt;integer&gt;
 *  The target image height</pre>
 * 
 * <pre> -samplewise-center
 *  Samplewise center</pre>
 * 
 * <pre> -samplewise-normalization
 *  Samplewise std. normalization</pre>
 * 
 * <pre> -rotation &lt;integer&gt;
 *  Degree range for random rotations</pre>
 * 
 * <pre> -width-shift-range &lt;number&gt;
 *  Fraction of width if &lt; 1; range [-x, x) if x &gt;= 1</pre>
 * 
 * <pre> -height-shift-range &lt;number&gt;
 *  Fraction of height if &lt;1; [-x, x) if x &gt;= 1</pre>
 * 
 * <pre> -rescale [number | expression]
 *  Rescaling factor (0/None for no rescaling)</pre>
 * 
 * <pre> -shear-range &lt;number&gt;
 *  Shear Intensity (Shear angle in counter-clockwise direction in degrees)</pre>
 * 
 * <pre> -zoom-range &lt;number&gt;
 *  Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range]</pre>
 * 
 * <pre> -channel-shift &lt;number&gt;
 *  Range for random channel shifts</pre>
 * 
 * <pre> -horizontal-flip
 *  Randomly flip inputs horizontally</pre>
 * 
 * <pre> -vertical-flip
 *  Randomly flip inputs vertically</pre>
 * 
 * <pre> -fill-mode [constant | nearest | reflect | wrap]
 *  One of {'constant', 'nearest', 'reflect' or 'wrap'}</pre>
 * 
 * <pre> -cval &lt;number&gt;
 *   Float or Int. Value used for points outside the boundaries when fill_mode = 'constant'</pre>
 * 
 * <pre> -reduce-lr
 *  Reduce learning rate when loss has stopped improving</pre>
 * 
 * <pre> -reduce-lr-factor &lt;number between 0 and 1&gt;
 *  Factor by which the learning rate will be reduced. new_lr = lr * factor</pre>
 * 
 * <pre> -reduce-lr-patience &lt;integer num epochs&gt;
 *  Number of epochs with no improvement after which learning rate will be reduced.</pre>
 * 
 * <pre> -reduce-lr-min-delta &lt;number&gt;
 *  Threshold for measuring the new optimum, to only focus on significant changes.</pre>
 * 
 * <pre> -reduce-lr-cooldown &lt;integer num epochs&gt;
 *  number of epochs to wait before resuming normal operation after lr has been reduced.</pre>
 * 
 * <pre> -reduce-lr-min-lr &lt;number&gt;
 *  Lower bound on the learning rate.</pre>
 * 
 * <pre> -lr-schedule
 *  Use an epoch-drive if-then-else learning rate schedule</pre>
 * 
 * <pre> -lr-schedule-def &lt;condition:rate,condition:rate,...,rate&gt;
 *  Definition of if-then-else schedule - format epoch:lr rate, epoch:lr, ..., lr, interpreted as if epoch # &lt; epoch then lr; else if ...; else lr</pre>
 * 
 * <pre> -checkpoints
 *  Save model after each epoch</pre>
 * 
 * <pre> -checkpoint-path &lt;path string&gt;
 *  Path to save checkpoint models to</pre>
 * 
 * <pre> -checkpoint-period &lt;integer&gt;
 *  How often (in epochs) to save a checkpoint model</pre>
 * 
 * <pre> -checkpoint-monitor &lt;[loss | val_loss]&gt;
 *  Monitor this metric - prevents best saved model from being overwritten, unless outperformed (loss, val_loss)</pre>
 * 
 * <pre> -gpus &lt;integer&gt;
 *  Number of GPUs to use (for best results, mini-batch size should be divisible by number of GPUs)</pre>
 * 
 * <pre> -gpu-turn-off-cpu-merge
 *  Multi-gpu: Do not force merging of model weights under the scope of the CPU (useful when NVLink is available)</pre>
 * 
 * <pre> -output-debug-info
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 * <pre> -do-not-check-capabilities
 *  If set, classifier capabilities are not checked before classifier is built
 *  (use with caution).</pre>
 * 
 * <pre> -num-decimal-places
 *  The number of decimal places for the output of numbers in the model (default 2).</pre>
 * 
 * <pre> -batch-size
 *  The desired batch size for batch prediction  (default 100).</pre>
 * 
 <!-- options-end -->
 *
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 */
public class KerasZooClassifier extends AbstractClassifier implements
  BatchPredictor, CapabilitiesHandler, EnvironmentHandler, LogHandler {

  private static final long serialVersionUID = 8553901105036380079L;

  /** Seed for attempting to achieve determinism */
  protected String m_seed = "1";

  // ------ image stuff ---------
  /**
   * True to turn off the use of the preprocess_input function associated with
   * the selected zoo model (in conjunction with manually specified image
   * preprocessing options
   */
  protected boolean m_dontUseModelSpecificImageProcFunc;

  /** Directory for finding images */
  protected File m_imageDirectory = new File(System.getProperty("user.dir"));

  /** Validation split fraction */
  protected String m_validationSplitPercent = "0.0"; // default - no validation
                                                     // split
  /** Separate validation set */
  protected File m_validationFile = new File("-NONE-");
  protected boolean m_separateValidationDataSetInPython;
  protected int m_separateValidationDataSetNumInstances = -1;

  protected boolean m_samplewiseCenter;
  protected boolean m_samplewiseStdNormalization;
  protected boolean m_zcaWhitening;
  protected String m_zcaEpsilon = "1e-06";
  protected String m_rotationRange = "0"; // integer
  protected String m_widthShiftRange = "0.0"; // percentage
  protected String m_heightShiftRange = "0.0"; // percentage
  // protected String m_brightnessRange = None // or list of two floats
  protected String m_rescale = "None"; // "1./255";
  protected String m_shearRange = "0.0"; // angle - counterclockwise in degrees
  protected String m_zoomRange = "0.0"; // between 0 and 1
  protected String m_channelShiftRange = "0.0";
  protected boolean m_horizontalFlip;
  protected boolean m_verticalFlip;
  protected String m_fillMode = "nearest"; // could change this to an enum
                                           // (nearest, constant, reflect,
                                           // wrap)
  protected String m_cval; // constant value when fillMode is set to constant

  // target_size = (w,h)
  protected String m_targetWidth = "224";
  protected String m_targetHeight = "224";
  // protected String m_colorMode = "'rgb'"; // rgb, grayscale

  /**
   * Whether to include an GlobalAveragePooling2D layer (overrides average
   * pooling layer)
   */
  protected boolean m_globalAveragePoolingLayer;

  /** Whether to include an average pooling layer */
  protected boolean m_averagePoolingLayer;
  /** Size of pool */
  protected String m_poolingSize = "2,2";

  /** Fully connected layers to add when transfer learning */
  protected String m_fcl = "512,512";

  /** Whether to put dropout between FCL */
  protected boolean m_dropOutsBetweenFCL;

  /** Rate for dropout */
  protected String m_dropOutRate = "0.5";

  /** Zoo model to use */
  protected ZooModel m_model = ZooModel.Xception;

  /** Weights to use. If None, then include_top is set to false */
  protected WeightsType m_weightsType = WeightsType.None;

  /**
   * Optimizer to use when training full network (no pretrained weights), or the
   * dense output layers when transfer learning
   */
  protected Optimizer m_optimizer = Optimizer.RMSprop;

  /** Options for the optimizer */
  protected String m_optimizerOptions = "";

  /** Whether to output top k (5) accuracy during iteration */
  protected boolean m_topKMetric;

  /** Number of epochs to run */
  protected String m_numEpochs = "10";

  /** steps_per_epoch = num_instances / batch size */
  protected String m_batchSize = "48";

  /** Maximum size of the queue for the generator */
  protected String m_maxQueueSize = "10";

  /** Number of workers to spin up for populating the generator queue */
  protected String m_workers = "1";

  /**
   * Path to load the .hdf5 model file from in python.
   */
  protected File m_modelLoadPath = new File("-NONE-");

  /** Path to save the .hdf5 model file to in python */
  protected File m_modelSavePath = new File("-NONE-");

  // ---- callbacks
  /** Whether to use a ReduceLROnPlateau callback */
  protected boolean m_reduceLRCallback;

  /** Factor by which LR will be reduced */
  protected String m_reduceLRfactor = "0.1";

  /** Number of epochs with no improvement, after which LR will be reduced */
  protected String m_reduceLRpatience = "10";

  /**
   * Threshold for measuring the new optimum, to only focus on significant
   * changes
   */
  protected String m_reduceLRMinDelta = "0.0001";

  /**
   * The number of epochs to wait before resuming normal operation after LR has
   * been reduced
   */
  protected String m_reduceLRCooldown = "0";

  /** Lower bound on the learning rate */
  protected String m_reduceLRMinLR = "0";

  /** Use a simple epoch-driven learning rate schedule */
  protected boolean m_learningRateSchedule;

  /**
   * Schedule definition if-elseif-else. condition:value, condition:value, ...,
   * default condition. Eg if epoch less than 10 then lr=0.01, elseif epoch less
   * than 20 then lr=0.001, else lr=0.0005
   */
  protected String m_lrScheduleDefinition = "10:0.01,20:0.001,0.0005";

  /** True to save models after every epoch */
  protected boolean m_useModelCheckpoints;

  /** Path for checkpoint models */
  protected String m_modelCheckpointPath =
    "${user.home}/model-{epoch:02d}-{loss:.2f}.hdf5";

  /** How often (in epochs) to checkpoint the model */
  protected String m_modelCheckpointPeriod = "1";

  /**
   * The metric to monitor - the best model saved according to this metric will
   * not be overwritten
   */
  protected String m_modelCheckpointMonitor = "loss";

  /** variable name prefix for model in python */
  protected String m_pythonModelPrefix = "";

  // Other stuff
  /** Zero R classifier */
  protected ZeroR m_zeroR;

  /** Hash code for making python-side stuff unique */
  protected String m_modelHash;

  /** Class priors */
  protected double[] m_classPriors;

  /**
   * True for nominal class labels that don't occur in the training data
   */
  protected boolean[] m_nominalEmptyClassIndexes;

  /** Environment variables to use */
  protected transient Environment m_env = Environment.getSystemWide();

  /** Log to use (if any) */
  protected transient Logger m_log;

  /** True to continue training (as long as model path .hdf5 file is valid... */
  protected boolean m_continueTraining;

  /**
   * Epoch to resume training at. numEpochs is added to this in order to
   * determine the final epoch number.
   */
  protected String m_initialEpoch = "0";

  /**
   * Where we will output training performance metrics so that we can tail it to
   * the log (if necessary)
   */
  protected File m_trainingLogFile = new File(
    "${user.home}/trainingProgress.txt");

  /**
   * Whether to print layer indexes and names, rather than the standard model
   * summary output to the log.
   */
  protected boolean m_printLayerIndexes;

  /** True to finetune top layers of the network if performing transfer learning */
  protected boolean m_finetuneNetwork;

  /**
   * Index from which to fine tune layers. All layers below this are frozen.
   * This results in a further training phase after training transfer layers.
   */
  protected String m_finetuneTopLayersIndex = "";

  /** Optimizer to use for fine-tuning */
  protected Optimizer m_finetuneOptimizer = Optimizer.SGD;

  /** Options for the fine-tune optimizer */
  protected String m_finetuneOptimizerOptions = "lr=1e-5, momentum=0.5";

  /**
   * Whether to apply the LR schedule callback to the fine tuning phase of
   * training
   */
  protected boolean m_finetuneUseLRScheduleIfSet;

  /**
   * Runs in a separate thread to monitor lines written to the csv file for
   * epoch stats
   */
  protected transient LogMonitor m_logMonitor;

  /** True if python is available */
  protected boolean m_pythonAvailable;

  /** True if Keras is installed and available */
  protected boolean m_kerasInstalled;

  /**
   * Number of GPUs (if GPU(s) are available) - multi-gpu training when >=2.
   */
  protected String m_numGPUs = "1";

  /** Turn off weight merging on CPU - again use this with NVLink */
  protected boolean m_gpusDoNotMergeOnCPU;

  /** True if multi-gpu parallel wrapper is being used */
  protected boolean m_parallelWrapper;

  /** Holds the number of GPUs actually available to use. -1 indicates we haven't checked yet */
  protected int m_availableGPUs = -1;

  /**
   * Set the seed
   *
   * @param seed the seed to use
   */
  @OptionMetadata(displayName = "Random seed", description = "Random seed",
    commandLineParamName = "seed",
    commandLineParamSynopsis = "-seed <integer>", displayOrder = 0)
  public void setSeed(String seed) {
    m_seed = seed;
  }

  /**
   * Get the seed
   *
   * @return the seed
   */
  public String getSeed() {
    return m_seed;
  }

  /**
   * Set the zoo model to use
   *
   * @param type the zoo model to use
   */
  @OptionMetadata(displayName = "Zoo model to use",
    description = "The zoo model to use", commandLineParamName = "zoo",
    commandLineParamSynopsis = "-zoo [Xception | VGG16 | VGG19 | ResNet | "
      + "ResNetV2 | ResNeXt | InceptionV3 | InceptionResNetV2 | MobileNet | "
      + "MobileNetV2 | DenseNet | NASNet]", displayOrder = 1)
  public void setZooModelType(ZooModel type) {
    m_model = type;
  }

  /**
   * Get the zoo model to use
   *
   * @return the zoo model
   */
  public ZooModel getZooModelType() {
    return m_model;
  }

  /**
   * Set the weights to use with the zoo model - either imagenet or None
   *
   * @param type the weights type to use
   */
  @OptionMetadata(displayName = "Weights", description = "Pre-trained or not",
    commandLineParamName = "weights",
    commandLineParamSynopsis = "-weights [imagenet | None]", displayOrder = 2)
  public void setWeightsType(WeightsType type) {
    m_weightsType = type;
  }

  /**
   * Get the weights to use with the zoo model - either imagenet or None
   *
   * @return the weights type to use
   */
  public WeightsType getWeightsType() {
    return m_weightsType;
  }

  /**
   * Set the number of epochs
   *
   * @param numEpochs number of epochs
   */
  @OptionMetadata(displayName = "Number of epochs",
    description = "Number of epochs to perform",
    commandLineParamName = "epochs",
    commandLineParamSynopsis = "-epochs <interger>", displayOrder = 3)
  public void setNumEpochs(String numEpochs) {
    m_numEpochs = numEpochs;
  }

  /**
   * Get the number of epochs
   *
   * @return the number of epochs
   */
  public String getNumEpochs() {
    return m_numEpochs;
  }

  /**
   * Set the size for the image generator queue
   *
   * @param queueSize the size of the queue
   */
  @OptionMetadata(displayName = "Max queue size",
    description = "Integer: Maximum size for the image generator queue.",
    commandLineParamName = "max-queue",
    commandLineParamSynopsis = "-max-queue <integer>", displayOrder = 4)
  public void setMaxQueueSize(String queueSize) {
    m_maxQueueSize = queueSize;
  }

  /**
   * Get the size of the image generator queue
   *
   * @return the size of the queue
   */
  public String getMaxQueueSize() {
    return m_maxQueueSize;
  }

  /**
   * Set the number of workers to use for feeding the generator queue
   *
   * @param workers the number of workers to use
   */
  @OptionMetadata(
    displayName = "Workers",
    description = "Number of workers to use for populating the generator queue",
    commandLineParamName = "workers",
    commandLineParamSynopsis = "-workers <integer>", displayOrder = 4)
  public
    void setWorkers(String workers) {
    m_workers = workers;
  }

  /**
   * Get the number of workers to use for feeding the generator queue
   *
   * @return the number of workers
   */
  public String getWorkers() {
    return m_workers;
  }

  /**
   * Set the optimizer to use for training the network
   *
   * @param optimizer the optimizer to use
   */
  @OptionMetadata(
    displayName = "Optimizer",
    description = "Optimizer to use when training full network from scratch, or"
      + "the top dense layers when transfer learning using pre-trained weights.",
    commandLineParamName = "optimizer",
    commandLineParamSynopsis = "-optimizer [SGD | RMSProp | Adam | Adamax | Nadam | Adagrad | Adadelta]",
    displayOrder = 5)
  public
    void setOptimizer(Optimizer optimizer) {
    m_optimizer = optimizer;
  }

  /**
   * Get the optimizer to use for training the network
   *
   * @return the optimizer to use
   */
  public Optimizer getOptimizer() {
    return m_optimizer;
  }

  /**
   * Set the options to use with the optimizer
   *
   * @param options options to use with the optimizer
   */
  @OptionMetadata(displayName = "Optimizer options",
    description = "Options for the selected optimizer (comma separated)",
    commandLineParamName = "optimizer-opts",
    commandLineParamSynopsis = "-optimizer-opts <string>", displayOrder = 5)
  public void setOptimizerOptions(String options) {
    m_optimizerOptions = options;
  }

  /**
   * Get the options to use with the optimizer
   *
   * @return the options to use with the optimizer
   */
  public String getOptimizerOptions() {
    return m_optimizerOptions;
  }

  /**
   * Set whether to output the top-5 accuracy to the log during training
   *
   * @param top5Accuracy true to output top-5 accuracy during training
   */
  @OptionMetadata(displayName = "Output top-5 accuracy",
    description = "Output the top-5 accuracy when iterating",
    commandLineParamName = "top-5", commandLineParamSynopsis = "-top-5",
    commandLineParamIsFlag = true, displayOrder = 5)
  public void setOutputTop5Accuracy(boolean top5Accuracy) {
    m_topKMetric = top5Accuracy;
  }

  /**
   * Get whether to output the top-5 accuracy to the log during training
   *
   * @return true if top-5 accuracy is to be output
   */
  public boolean getOutputTop5Accuracy() {
    return m_topKMetric;
  }

  /**
   * Set the number and size of fully connected layers to add to the end of the
   * network when performing transfer learning
   *
   * @param fcLayers a comma-separated list of layer sizes
   */
  @OptionMetadata(displayName = "Fully connected layers (list) to add",
    description = "Number (and size) of fully connected layers as a "
      + "comma-separated list to add to the end of the network",
    commandLineParamName = "fcl",
    commandLineParamSynopsis = "-fcl <comma separated list of layer sizes>",
    displayOrder = 6, category = "Transfer learning")
  public void setFCLayers(String fcLayers) {
    m_fcl = fcLayers;
  }

  /**
   * Get the number and size of fully connected layers to add to the end of the
   * network when performing transfer learning
   *
   * @return a comma-separated list of layer sizes
   */
  public String getFCLayers() {
    return m_fcl;
  }

  /**
   * Set whether to include a global average pooling layer when transfer
   * learning. Overrides the AP layer option.
   *
   * @param gapLayer true to include a GAP layer
   */
  @OptionMetadata(displayName = "Include global average pooling layer",
    description = "Add a GAP layer before fully connected layers (replaces AP "
      + "layer if set)", commandLineParamName = "gap",
    commandLineParamSynopsis = "-gap", commandLineParamIsFlag = true,
    displayOrder = 6, category = "Transfer learning")
  public void setIncludeGAPLayer(boolean gapLayer) {
    m_globalAveragePoolingLayer = gapLayer;
  }

  /**
   * Get whether to include a global average pooling layer when transfer
   * learning. Overrides the AP layer option.
   *
   * @return true if including a GAP layer
   */
  public boolean getIncludeGAPLayer() {
    return m_globalAveragePoolingLayer;
  }

  /**
   * Set whether to include an average pooling layer when transfer learning.
   *
   * @param apLayer true to include an AP layer
   */
  @OptionMetadata(displayName = "Include average pooling layer",
    description = "Add an AP layer before fully connected layers",
    commandLineParamName = "ap", commandLineParamSynopsis = "-ap",
    commandLineParamIsFlag = true, displayOrder = 7,
    category = "Transfer learning")
  public void setIncludeAPLayer(boolean apLayer) {
    m_averagePoolingLayer = apLayer;
  }

  /**
   * Get whether to include an average pooling layer when transfer learning.
   *
   * @return true to include an AP layer
   */
  public boolean getIncludeAPLayer() {
    return m_averagePoolingLayer;
  }

  /**
   * Set the size of the pool to use with the AP layer.
   *
   * @param poolingSize the size of the pool - two comma-separated values
   */
  @OptionMetadata(displayName = "Pool size",
    description = "Pool size (comma separated)",
    commandLineParamName = "pool-size",
    commandLineParamSynopsis = "-pool-size <int,int>", displayOrder = 8,
    category = "Transfer learning")
  public void setPoolSize(String poolingSize) {
    m_poolingSize = poolingSize;
  }

  /**
   * Get the size of the pool to use with the AP layer.
   *
   * @return the size of the pool - two comma-separated values
   */
  public String getPoolSize() {
    return m_poolingSize;
  }

  /**
   * Set whether to add a dropout layer between FC layers
   *
   * @param dropouts true to add a dropout layer between FC layers
   */
  @OptionMetadata(displayName = "Add dropout layers between FC layers",
    description = "Whether to add a Dropout layer after each fully connected "
      + "connected layer", commandLineParamName = "dropout",
    commandLineParamSynopsis = "-dropout", commandLineParamIsFlag = true,
    displayOrder = 9, category = "Transfer learning")
  public void setAddDropoutBetweenFCLayers(boolean dropouts) {
    m_dropOutsBetweenFCL = dropouts;
  }

  /**
   * Get whether to add a dropout layer between FC layers
   *
   * @return true if adding a dropout layer between FC layers
   */
  public boolean getAddDropoutBetweenFCLayers() {
    return m_dropOutsBetweenFCL;
  }

  /**
   * Set the rate for dropout
   *
   * @param dropoutRate the rate for dropout
   */
  @OptionMetadata(displayName = "Dropout rate",
    description = "Rate (between 0 and 1) for dropout",
    commandLineParamName = "dropout-rate",
    commandLineParamSynopsis = "-dropout-rate <number>", displayOrder = 9,
    category = "Transfer learning")
  public void setDropoutRate(String dropoutRate) {
    m_dropOutRate = dropoutRate;
  }

  /**
   * Get the rate for dropout
   *
   * @return the rate for dropout
   */
  public String getDropoutRate() {
    return m_dropOutRate;
  }

  /**
   * Set whether to perform fine tuning on some number of top layers when
   * performing transfer learning
   *
   * @param finetuneNetwork true to perform fine tuning
   */
  @OptionMetadata(displayName = "Fine-tune top layers",
    description = "Whether to fine-tune top convolutional layers",
    commandLineParamName = "finetune", commandLineParamSynopsis = "-finetune",
    commandLineParamIsFlag = true, displayOrder = 10,
    category = "Transfer learning")
  public void setFinetuneNetwork(boolean finetuneNetwork) {
    m_finetuneNetwork = finetuneNetwork;
  }

  /**
   * Get whether to perform fine tuning on some number of top layers when
   * performing transfer learning
   *
   * @return true to perform fine tuning
   */
  public boolean getFinetuneNetwork() {
    return m_finetuneNetwork;
  }

  /**
   * Set whether to print network layer indices and names to the log
   *
   * @param printLayerIndexes true to print layer indices and names
   */
  @OptionMetadata(displayName = "Print layer indices and names to log",
    description = "Print this info instead of model summary. This can help in "
      + "deciding which layers to fine tune.",
    commandLineParamName = "print-layer-indexes",
    commandLineParamSynopsis = "-print-layer-indexes",
    commandLineParamIsFlag = true, displayOrder = 11,
    category = "Transfer learning")
  public void setFinetunePrintLayerIndexes(boolean printLayerIndexes) {
    m_printLayerIndexes = printLayerIndexes;
  }

  /**
   * Get whether to print network layer indices and names to the log
   *
   * @return true to print layer indices and names
   */
  public boolean getFinetunePrintLayerIndexes() {
    return m_printLayerIndexes;
  }

  /**
   * Set the layer index at which to unfreeze layers for fine tuning. All top
   * layers above, and including, this layer get unfrozen for fine tuning.
   *
   * @param layerStartIndex the layer index at which to unfreeze layers
   */
  @OptionMetadata(displayName = "Start fine-tuning top layers from this index",
    description = "Fine-tune all top layers above and including this layer",
    commandLineParamName = "finetune-index",
    commandLineParamSynopsis = "-finetune-index <index>", displayOrder = 12,
    category = "Transfer learning")
  public void setFinetuneLayerStartIndex(String layerStartIndex) {
    m_finetuneTopLayersIndex = layerStartIndex;
  }

  /**
   * Get the layer index at which to unfreeze layers for fine tuning. All top
   * layers above, and including, this layer get unfrozen for fine tuning.
   *
   * @return the layer index at which to unfreeze layers
   */
  public String getFinetuneLayerStartIndex() {
    return m_finetuneTopLayersIndex;
  }

  /**
   * The optimizer to use during fine tuning.
   *
   * @param finetuneOptimizer the optimizer to use during fine tuning.
   */
  @OptionMetadata(
    displayName = "Optimizer for fine-tuning",
    description = "Optimizer to use for fine-tuning the network",
    commandLineParamName = "finetune-optimizer",
    commandLineParamSynopsis = "-finetune-optimizer [SGD | RMSProp | Adam | Adamax | Nadam | Adagrad | Adadelta]",
    displayOrder = 13, category = "Transfer learning")
  public
    void setFinetuneOptimizer(Optimizer finetuneOptimizer) {
    m_finetuneOptimizer = finetuneOptimizer;
  }

  /**
   * The optimizer to use during fine tuning.
   *
   * @return the optimizer to use during fine tuning.
   */
  public Optimizer getFinetuneOptimizer() {
    return m_finetuneOptimizer;
  }

  /**
   * Set the options to use with the fine tuning optimizer
   *
   * @param finetuneOptimizerOpts the options to use with the fine tune
   *          optimizer
   */
  @OptionMetadata(displayName = "Options for the fine-tune optimizer",
    description = "Options for the fine-tune optimizer",
    commandLineParamName = "finetune-optimizer-opts",
    commandLineParamSynopsis = "-finetune-optimizer-opts <string>",
    displayOrder = 14, category = "Transfer learning")
  public void setFinetuneOptimizerOpts(String finetuneOptimizerOpts) {
    m_finetuneOptimizerOptions = finetuneOptimizerOpts;
  }

  /**
   * Get the options to use with the fine tuning optimizer
   *
   * @return the options to use with the fine tune optimizer
   */
  public String getFinetuneOptimizerOpts() {
    return m_finetuneOptimizerOptions;
  }

  /**
   * Set whether to use the primary learning rate scheduler (if set) for the
   * fine tuning training phase. This will then override any learning rate
   * specified in the fine tuning optimizer's options
   *
   * @param useLRSchedulerIfSet true to use the primary learning rate schedule
   *          (if set) for fine tuning
   */
  @OptionMetadata(
    displayName = "Use LR scheduler (if defined)",
    description = "Override the learning rate in the fine-tune optimizer options "
      + "with those set by the LR schedule callback (if defined)",
    commandLineParamName = "finetune-use-lr-schedule",
    commandLineParamSynopsis = "-finetune-use-lr-schedule",
    commandLineParamIsFlag = true, displayOrder = 15,
    category = "Transfer learning")
  public
    void setFinetuneUseLRSchedulerIfSet(boolean useLRSchedulerIfSet) {
    m_finetuneUseLRScheduleIfSet = useLRSchedulerIfSet;
  }

  /**
   * Get whether to use the primary learning rate scheduler (if set) for the
   * fine tuning training phase. This will then override any learning rate
   * specified in the fine tuning optimizer's options
   *
   * @return true if using the primary learning rate schedule (if set) for fine
   *         tuning
   */
  public boolean getFinetuneUseLRSchedulerIfSet() {
    return m_finetuneUseLRScheduleIfSet;
  }

  /**
   * Set whether to continue training using network/weights loaded from the
   * specified model file. Note: don't set this if performing cross-validation!
   *
   * @param continueTraining true to continue training a previously saved (hdf5)
   *          network
   */
  @OptionMetadata(displayName = "Continue training",
    description = "Continue training using network/weights loaded from "
      + "model file (don't set this if performing a cross-validation)",
    commandLineParamName = "continue", commandLineParamSynopsis = "-continue",
    commandLineParamIsFlag = true, displayOrder = 10,
    category = "Model and log paths")
  public void setContinueTraining(boolean continueTraining) {
    m_continueTraining = continueTraining;
  }

  /**
   * Get whether to continue training using network/weights loaded from the
   * specified model file. Note: don't set this if performing cross-validation!
   *
   * @return true if continuing training a previously saved (hdf5) network
   */
  public boolean getContinueTraining() {
    return m_continueTraining;
  }

  /**
   * Set the initial epoch when resuming training. The num epochs parameter then
   * specifies how many epochs to perform from this starting point (i.e.
   * determines the final epoch number).
   *
   * @param initialEpoch the initial epoch to resume/continue training from
   */
  @OptionMetadata(
    displayName = "Initial epoch to continue training from",
    description = "When continuing training, start from this epoch; num epochs "
      + "is added to this to determine the final epoch number",
    commandLineParamName = "initial-epoch",
    commandLineParamSynopsis = "-initial-epoch <integer>", displayOrder = 11,
    category = "Model and log paths")
  public
    void setInitialEpoch(String initialEpoch) {
    m_initialEpoch = initialEpoch;
  }

  /**
   * Get the initial epoch when resuming training. The num epochs parameter then
   * specifies how many epochs to perform from this starting point (i.e.
   * determines the final epoch number).
   *
   * @return the initial epoch to resume/continue training from
   */
  public String getInitialEpoch() {
    return m_initialEpoch;
  }

  /**
   * Set the path to load a saved (.hdf5) model from
   *
   * @param modelLoadPath the path to load the model from
   */
  @FilePropertyMetadata(fileChooserDialogType = KFGUIConsts.OPEN_DIALOG,
    directoriesOnly = false)
  @OptionMetadata(displayName = "Load model path",
    description = "Path to load network structure and weights from (i.e. to "
      + "continue training an existing model)", commandLineParamName = "load",
    commandLineParamSynopsis = "-load <path to .hdf5 file>", displayOrder = 11,
    category = "Model and log paths")
  public void setModelLoadPath(File modelLoadPath) {
    m_modelLoadPath = modelLoadPath;
  }

  /**
   * Get the path to load a saved (.hdf5) model from
   *
   * @return the path to load the model from
   */
  public File getModelLoadPath() {
    return m_modelLoadPath;
  }

  /**
   * Set the path to save (.hdf5) the current model to.
   *
   * @param modelSavePath the path to save the current model to.
   */
  @FilePropertyMetadata(fileChooserDialogType = KFGUIConsts.SAVE_DIALOG,
    directoriesOnly = false)
  @OptionMetadata(
    displayName = "Save model path",
    description = "Path to save the final trained network (structure and weights) to",
    commandLineParamName = "save",
    commandLineParamSynopsis = "-save <path to .hdf5 file>", displayOrder = 12,
    category = "Model and log paths")
  public
    void setModelSavePath(File modelSavePath) {
    m_modelSavePath = modelSavePath;
  }

  /**
   * Get the path to save (.hdf5) the current model to.
   *
   * @return the path to save the current model to.
   */
  public File getModelSavePath() {
    return m_modelSavePath;
  }

  /**
   * Set the file to save the epoch stats to
   *
   * @param trainingLogFile the file to save the epoch stats to
   */
  @FilePropertyMetadata(fileChooserDialogType = KFGUIConsts.SAVE_DIALOG,
    directoriesOnly = false)
  @OptionMetadata(displayName = "Training log file",
    description = "File to write training performance to while iterating",
    commandLineParamName = "log",
    commandLineParamSynopsis = "-log <log file path>", displayOrder = 13,
    category = "Model and log paths")
  public void setTrainingLogFile(File trainingLogFile) {
    m_trainingLogFile = trainingLogFile;
  }

  /**
   * Get the file to save the epoch stats to
   *
   * @return the file to save the epoch stats to
   */
  public File getTrainingLogFile() {
    return m_trainingLogFile;
  }

  /**
   * Set whether to turn off the zoo model-specific image preprocessing
   *
   * @param dontUseZooModelSpecificImageProcessingFunction true to turn off the
   *          use of zoo model-specific preprocessing
   */
  @OptionMetadata(
    displayName = "Turn off zoo-model specific preprocessing",
    description = "Don't apply the preprocess_input function associated with the "
      + "selected zoo model. This function is applied in conjunction with manually specified "
      + "image processing options.",
    commandLineParamName = "dont-use-model-specific",
    commandLineParamSynopsis = "-dont-use-model-specific",
    commandLineParamIsFlag = true, category = "Image processing",
    displayOrder = 13)
  public
    void setDontUseZooModelSpecificImageProcessingFunction(
      boolean dontUseZooModelSpecificImageProcessingFunction) {
    m_dontUseModelSpecificImageProcFunc =
      dontUseZooModelSpecificImageProcessingFunction;
  }

  /**
   * Get whether to turn off the zoo model-specific image preprocessing
   *
   * @return true if turning off the use of zoo model-specific preprocessing
   */
  public boolean getDontUseZooModelSpecificImageProcessingFunction() {
    return m_dontUseModelSpecificImageProcFunc;
  }

  /**
   * Set the directory in which to find images
   *
   * @param location the directory in which to find images
   */
  @FilePropertyMetadata(fileChooserDialogType = KFGUIConsts.OPEN_DIALOG,
    directoriesOnly = true)
  @OptionMetadata(displayName = "Images location",
    description = "Directory containing images",
    commandLineParamName = "images",
    commandLineParamSynopsis = "-images <directory>", displayOrder = 14,
    category = "Image processing")
  public void setImagesLocation(File location) {
    m_imageDirectory = location;
  }

  /**
   * Get the directory in which to find images
   *
   * @return the directory in which to find images
   */
  public File getImagesLocation() {
    return m_imageDirectory;
  }

  /**
   * Set the fraction of the training data to use for validation. 0 = no
   * validation split.
   *
   * @param validationSplitPercentage the fraction (between 0 and 1) of the
   *          training data to use for validation
   */
  @OptionMetadata(
    displayName = "Validation split percent",
    description = "Amount of training data/images to use for validation as a fraction "
      + "between 0 and 1. 0 = no validation split.",
    commandLineParamName = "validation-split",
    commandLineParamSynopsis = "-validation-split <number between 0 and 1>",
    displayOrder = 15, category = "Image processing")
  public
    void setValidationSplitPercentage(String validationSplitPercentage) {
    m_validationSplitPercent = validationSplitPercentage;
  }

  /**
   * Get the fraction of the training data to use for validation. 0 = no
   * validation split.
   *
   * @return the fraction (between 0 and 1) of the training data to use for
   *         validation
   */
  public String getValidationSplitPercentage() {
    return m_validationSplitPercent;
  }

  /**
   * Set the path to a separate file containing validation data (csv or arff).
   * Keras then performs validation on this data rather than holding out a
   * percentage of the training data for validation. Setting this option
   * overrides the validation percent.
   *
   * @param validationFile the
   */
  @FilePropertyMetadata(fileChooserDialogType = KFGUIConsts.OPEN_DIALOG,
    directoriesOnly = false)
  @OptionMetadata(displayName = "Separate validation data",
    description = "Separate file for validation (arff or csv)",
    commandLineParamName = "validation-data",
    commandLineParamSynopsis = "-validation-data <path to arff or csv file>",
    displayOrder = 15, category = "Image processing")
  public void setValidationFile(File validationFile) {
    m_validationFile = validationFile;
  }

  public File getValidationFile() {
    return m_validationFile;
  }

  /**
   * Set the size of the mini-batch to use
   *
   * @param batchSize the size of the mini-batch
   */
  @OptionMetadata(displayName = "Mini batch size",
    description = "Size of the mini batches to train with",
    commandLineParamName = "mini-batch",
    commandLineParamSynopsis = "-mini-batch <integer>", displayOrder = 15,
    category = "Image processing")
  public void setMiniBatchSize(String batchSize) {
    m_batchSize = batchSize;
  }

  /**
   * Get the size of the mini-batch to use
   *
   * @return the size of the mini-batch
   */
  public String getMiniBatchSize() {
    return m_batchSize;
  }

  /**
   * Set the image width. If not specified, then the default size for the
   * selected zoo model will be used.
   *
   * @param imageWidth the image width
   */
  @OptionMetadata(displayName = "Image width",
    description = "The target width of the images",
    commandLineParamName = "width",
    commandLineParamSynopsis = "-width <integer>", displayOrder = 16,
    category = "Image processing")
  public void setImageWidth(String imageWidth) {
    m_targetWidth = imageWidth;
  }

  /**
   * Get the image width. If not specified, then the default size for the
   * selected zoo model will be used.
   *
   * @return the image width
   */
  public String getImageWidth() {
    return m_targetWidth;
  }

  /**
   * Set the image height. If not specified, then the default size for the
   * selected zoo model will be used.
   *
   * @param imageHeight the image height
   */
  @OptionMetadata(displayName = "Image height",
    description = "The target image height", commandLineParamName = "height",
    commandLineParamSynopsis = "-height <integer>", displayOrder = 17,
    category = "Image processing")
  public void setImageHeight(String imageHeight) {
    m_targetHeight = imageHeight;
  }

  /**
   * Get the image height. If not specified, then the default size for the
   * selected zoo model will be used.
   *
   * @return the image height
   */
  public String getImageHeight() {
    return m_targetHeight;
  }

  /**
   * Set whether to perform samplewise centering
   *
   * @param samplewiseCenter true if performing samplewise centering
   */
  @OptionMetadata(displayName = "Samplewise center",
    description = "Samplewise center",
    commandLineParamName = "samplewise-center",
    commandLineParamSynopsis = "-samplewise-center",
    commandLineParamIsFlag = true, displayOrder = 20,
    category = "Image processing")
  public void setSamplewiseCenter(boolean samplewiseCenter) {
    m_samplewiseCenter = samplewiseCenter;
  }

  /**
   * Get whether to perform samplewise centering
   *
   * @return true if perform samplewise centering
   */
  public boolean getSamplewiseCenter() {
    return m_samplewiseCenter;
  }

  /**
   * Set whether to perform samplewise standard normalization
   *
   * @param samplewiseStdNormalization true to perform samplewise standard
   *          normalization
   */
  @OptionMetadata(displayName = "Samplewise std. normalization",
    description = "Samplewise std. normalization",
    commandLineParamName = "samplewise-normalization",
    commandLineParamSynopsis = "-samplewise-normalization",
    commandLineParamIsFlag = true, displayOrder = 22,
    category = "Image processing")
  public void setSamplewiseStdNormalization(boolean samplewiseStdNormalization) {
    m_samplewiseStdNormalization = samplewiseStdNormalization;
  }

  /**
   * Get whether to perform samplewise standard normalization
   *
   * @return true if performing samplewise standard normalization
   */
  public boolean getSamplewiseStdNormalization() {
    return m_samplewiseStdNormalization;
  }

  /**
   * Set the rotation range for images in degrees
   *
   * @param rotationRange the range in degrees
   */
  @OptionMetadata(displayName = "Rotation range",
    description = "Degree range for random rotations",
    commandLineParamName = "rotation",
    commandLineParamSynopsis = "-rotation <integer>", displayOrder = 25,
    category = "Image processing")
  public void setRotationRange(String rotationRange) {
    m_rotationRange = rotationRange;
  }

  /**
   * Get the rotation range for images in degrees
   *
   * @return the range in degrees
   */
  public String getRotationRange() {
    return m_rotationRange;
  }

  /**
   * Set the amount by which to randomly shift width. Is a fraction of width if
   * < 1; otherwise a range [-x, x) if x greater than, or equal to, 1.
   *
   * @param widthShiftRange the width shift
   */
  @OptionMetadata(displayName = "Width shift range",
    description = "Fraction of width if < 1; range [-x, x) if x >= 1",
    commandLineParamName = "width-shift-range",
    commandLineParamSynopsis = "-width-shift-range <number>",
    displayOrder = 26, category = "Image processing")
  public void setWidthShiftRange(String widthShiftRange) {
    m_widthShiftRange = widthShiftRange;
  }

  /**
   * Get the amount by which to randomly shift width. Is a fraction of width if
   * < 1; otherwise a range [-x, x) if x greater that or equal to 1.
   *
   * @return the width shift
   */
  public String getWidthShiftRange() {
    return m_widthShiftRange;
  }

  /**
   * Set the amount by which to randomly shift height. Is a fraction of height
   * if < 1; otherwise a range [-x, x) if x greater than, or equal to, 1.
   *
   * @param heightShiftRange the height shift
   */
  @OptionMetadata(displayName = "Height shift range",
    description = "Fraction of height if <1; [-x, x) if x >= 1",
    commandLineParamName = "height-shift-range",
    commandLineParamSynopsis = "-height-shift-range <number>",
    displayOrder = 27, category = "Image processing")
  public void setHeightShiftRange(String heightShiftRange) {
    m_heightShiftRange = heightShiftRange;
  }

  /**
   * Get the amount by which to randomly shift height. Is a fraction of height
   * if < 1; otherwise a range [-x, x) if x greater than, or equal to, 1.
   *
   * @return the height shift
   */
  public String getHeightShiftRange() {
    return m_heightShiftRange;
  }

  /**
   * Set image rescaling factor
   *
   * @param rescale the rescaling factor, or 0/None for no rescaling
   */
  @OptionMetadata(displayName = "Rescaling factor",
    description = "Rescaling factor (0/None for no rescaling)",
    commandLineParamName = "rescale",
    commandLineParamSynopsis = "-rescale [number | expression]",
    displayOrder = 28, category = "Image processing")
  public void setRescale(String rescale) {
    m_rescale = rescale;
  }

  /**
   * Get image rescaling factor
   *
   * @return the rescaling factor, or 0/None for no rescaling
   */
  public String getRescale() {
    return m_rescale;
  }

  /**
   * Set the shear range - Shear Intensity (Shear angle in counter-clockwise
   * direction in degrees)
   *
   * @param shearRange the shear range
   */
  @OptionMetadata(
    displayName = "Shear range",
    description = "Shear Intensity (Shear angle in counter-clockwise direction "
      + "in degrees)", commandLineParamName = "shear-range",
    commandLineParamSynopsis = "-shear-range <number>", displayOrder = 29,
    category = "Image processing")
  public
    void setShearRange(String shearRange) {
    m_shearRange = shearRange;
  }

  /**
   * Set the shear range - Shear Intensity (Shear angle in counter-clockwise
   * direction in degrees)
   *
   * @return the shear range
   */
  public String getShearRange() {
    return m_shearRange;
  }

  /**
   * Set zoom range for random zoom. If a float, [lower, upper] = [1-zoom_range,
   * 1+zoom_range].
   *
   * @param zoomRange the zoom range
   */
  @OptionMetadata(displayName = "Zoom range",
    description = "Range for random zoom. If a float, [lower, upper] = "
      + "[1-zoom_range, 1+zoom_range]", commandLineParamName = "zoom-range",
    commandLineParamSynopsis = "-zoom-range <number>", displayOrder = 30,
    category = "Image processing")
  public void setZoomRange(String zoomRange) {
    m_zoomRange = zoomRange;
  }

  /**
   * Set zoom range for random zoom. If a float, [lower, upper] = [1-zoom_range,
   * 1+zoom_range].
   *
   * @return the zoom range
   */
  public String getZoomRange() {
    return m_zoomRange;
  }

  /**
   * Set the range for random channel shifts
   *
   * @param channelShiftRange the channel shift
   */
  @OptionMetadata(displayName = "Channel shift range",
    description = "Range for random channel shifts",
    commandLineParamName = "channel-shift",
    commandLineParamSynopsis = "-channel-shift <number>", displayOrder = 31,
    category = "Image processing")
  public void setChannelShiftRange(String channelShiftRange) {
    m_channelShiftRange = channelShiftRange;
  }

  /**
   * Get the range for random channel shifts
   *
   * @return the channel shift
   */
  public String getChannelShiftRange() {
    return m_channelShiftRange;
  }

  /**
   * Set whether to randomly flip images horizontally
   *
   * @param horizontalFlip true to randomly flip images horizontally
   */
  @OptionMetadata(displayName = "Horizontal flip",
    description = "Randomly flip inputs horizontally",
    commandLineParamName = "horizontal-flip",
    commandLineParamSynopsis = "-horizontal-flip",
    commandLineParamIsFlag = true, displayOrder = 32,
    category = "Image processing")
  public void setHorizontalFlip(boolean horizontalFlip) {
    m_horizontalFlip = horizontalFlip;
  }

  /**
   * Get whether to randomly flip images horizontally
   *
   * @return true if randomly flipping images horizontally
   */
  public boolean getHorizontalFlip() {
    return m_horizontalFlip;
  }

  /**
   * Set whether to randomly flip images vertically
   *
   * @param verticalFlip true to flip images vertically
   */
  @OptionMetadata(displayName = "Vertical flip",
    description = "Randomly flip inputs vertically",
    commandLineParamName = "vertical-flip",
    commandLineParamSynopsis = "-vertical-flip", commandLineParamIsFlag = true,
    displayOrder = 33, category = "Image processing")
  public void setVerticalFlip(boolean verticalFlip) {
    m_verticalFlip = verticalFlip;
  }

  /**
   * Get whether to randomly flip images vertically
   *
   * @return true if flipping images vertically
   */
  public boolean getVerticalFlip() {
    return m_verticalFlip;
  }

  /**
   * Set the fill mode to use - one of constant, nearest, reflect or wrap
   *
   * @param fillMode the fill mode to use
   */
  @OptionMetadata(
    displayName = "Fill mode",
    description = "One of {'constant', 'nearest', 'reflect' or 'wrap'}",
    commandLineParamName = "fill-mode",
    commandLineParamSynopsis = "-fill-mode [constant | nearest | reflect | wrap]",
    displayOrder = 34, category = "Image processing")
  public
    void setFillMode(String fillMode) {
    m_fillMode = fillMode;
  }

  /**
   * Get the fill mode to use - one of constant, nearest, reflect or wrap
   *
   * @return the fill mode to use
   */
  public String getFillMode() {
    return m_fillMode;
  }

  /**
   * Set the Cval - value used for points outside the boundaries when fill mode
   * is set to 'constant'.
   *
   * @param cVal the Cval to use
   */
  @OptionMetadata(
    displayName = "Cval",
    description = " Float or Int. Value used for points outside the boundaries "
      + "when fill_mode = 'constant'", commandLineParamName = "cval",
    commandLineParamSynopsis = "-cval <number>", displayOrder = 35,
    category = "Image processing")
  public
    void setCval(String cVal) {
    m_cval = cVal;
  }

  /**
   * Get the Cval - value used for points outside the boundaries when fill mode
   * is set to 'constant'.
   *
   * @return the Cval to use
   */
  public String getCval() {
    return m_cval;
  }

  /**
   * Set whether to use a callback that reduces the learning rate when the loss
   * has stopped improving
   *
   * @param reduceLROnPlateau true to use a reduce learning rate callback
   */
  @OptionMetadata(displayName = "Reduce learning rate on plateau",
    description = "Reduce learning rate when loss has stopped improving",
    commandLineParamName = "reduce-lr",
    commandLineParamSynopsis = "-reduce-lr", commandLineParamIsFlag = true,
    displayOrder = 36, category = "Training callbacks")
  public void setReduceLROnPlateau(boolean reduceLROnPlateau) {
    m_reduceLRCallback = reduceLROnPlateau;
  }

  /**
   * Get whether to use a callback that reduces the learning rate when the loss
   * has stopped improving
   *
   * @return true when using a reduce learning rate callback
   */
  public boolean getReduceLROnPlateau() {
    return m_reduceLRCallback;
  }

  /**
   * Set the factor by which to reduce the learning rate when using a reduce
   * learning rate on plateau callback
   *
   * @param factor the factor by which to reduce the learning rate
   */
  @OptionMetadata(displayName = "Reduce LR factor",
    description = "Factor by which the learning rate will be reduced. "
      + "new_lr = lr * factor", commandLineParamName = "reduce-lr-factor",
    commandLineParamSynopsis = "-reduce-lr-factor <number between 0 and 1>",
    displayOrder = 37, category = "Training callbacks")
  public void setReduceLRFactor(String factor) {
    m_reduceLRfactor = factor;
  }

  /**
   * Get the factor by which to reduce the learning rate when using a reduce
   * learning rate on plateau callback
   *
   * @return the factor by which to reduce the learning rate
   */
  public String getReduceLRFactor() {
    return m_reduceLRfactor;
  }

  /**
   * Set the patience for the reduce learning rate on plateau callback. This is
   * the number of epochs with no improvement in loss after which the learning
   * rate will be reduced.
   *
   * @param patience the number of epochs with no improvement before reducing
   *          the learning rate
   */
  @OptionMetadata(displayName = "Reduce LR patience",
    description = "Number of epochs with no improvement after which learning "
      + "rate will be reduced.", commandLineParamName = "reduce-lr-patience",
    commandLineParamSynopsis = "-reduce-lr-patience <integer num epochs>",
    displayOrder = 38, category = "Training callbacks")
  public void setReduceLRPatience(String patience) {
    m_reduceLRpatience = patience;
  }

  /**
   * Get the patience for the reduce learning rate on plateau callback. This is
   * the number of epochs with no improvement in loss after which the learning
   * rate will be reduced.
   *
   * @return the number of epochs with no improvement before reducing the
   *         learning rate
   */
  public String getReduceLRPatience() {
    return m_reduceLRpatience;
  }

  /**
   * Set threshold for measuring the new optimum for the reduce learning rate on
   * plateau callback.
   *
   * @param minDelta threshold for new optimum
   */
  @OptionMetadata(displayName = "Reduce LR min delta",
    description = "Threshold for measuring the new optimum, to only focus on "
      + "significant changes.", commandLineParamName = "reduce-lr-min-delta",
    commandLineParamSynopsis = "-reduce-lr-min-delta <number>",
    displayOrder = 39, category = "Training callbacks")
  public void setReduceLRMinDelta(String minDelta) {
    m_reduceLRMinDelta = minDelta;
  }

  /**
   * Get threshold for measuring the new optimum for the reduce learning rate on
   * plateau callback.
   *
   * @return threshold for new optimum
   */
  public String getReduceLRMinDelta() {
    return m_reduceLRMinDelta;
  }

  /**
   * Set the cooldown period for the reduce learning rate on plateau callback.
   * This is the number of epochs to wait before resuming normal operation.
   *
   * @param cooldown the cooldown period in epochs
   */
  @OptionMetadata(displayName = "Reduce LR cooldown",
    description = "number of epochs to wait before resuming normal operation "
      + "after lr has been reduced.",
    commandLineParamName = "reduce-lr-cooldown",
    commandLineParamSynopsis = "-reduce-lr-cooldown <integer num epochs>",
    displayOrder = 40, category = "Training callbacks")
  public void setReduceLRCooldown(String cooldown) {
    m_reduceLRCooldown = cooldown;
  }

  /**
   * Get the cooldown period for the reduce learning rate on plateau callback.
   * This is the number of epochs to wait before resuming normal operation.
   *
   * @return the cooldown period in epochs
   */
  public String getReduceLRCooldown() {
    return m_reduceLRCooldown;
  }

  /**
   * Set the lower bound on the learning rate when using the reduce learning
   * rate on plateau callback.
   *
   * @param minLR the minimum learning rate
   */
  @OptionMetadata(displayName = "Reduce LR minimum learning rate",
    description = "Lower bound on the learning rate.",
    commandLineParamName = "reduce-lr-min-lr",
    commandLineParamSynopsis = "-reduce-lr-min-lr <number>", displayOrder = 41,
    category = "Training callbacks")
  public void setReduceLRMinLR(String minLR) {
    m_reduceLRMinLR = minLR;
  }

  /**
   * Get the lower bound on the learning rate when using the reduce learning
   * rate on plateau callback.
   *
   * @return the minimum learning rate
   */
  public String getReduceLRMinLR() {
    return m_reduceLRMinLR;
  }

  /**
   * Set whether to use a learning rate schedule.
   *
   * @param useLearningRateSchedule true to use a learning rate schedule
   */
  @OptionMetadata(displayName = "Use a learning rate schedule",
    description = "Use an epoch-drive if-then-else learning rate schedule",
    commandLineParamName = "lr-schedule",
    commandLineParamSynopsis = "-lr-schedule", commandLineParamIsFlag = true,
    displayOrder = 42, category = "Training callbacks")
  public void setUseLearningRateSchedule(boolean useLearningRateSchedule) {
    m_learningRateSchedule = useLearningRateSchedule;
  }

  /**
   * Get whether to use a learning rate schedule.
   *
   * @return true if using a learning rate schedule
   */
  public boolean getUseLearningRateSchedule() {
    return m_learningRateSchedule;
  }

  /**
   * Set the schedule to use with the learning rate schedule callback. Format is
   * 'epoch:lr, epoch:lr, ..., lr', interpreted as 'if epoch # < epoch then lr;
   * else if ...; else lr'.
   *
   * @param schedule a string defining an if-then-else schedule for the learning
   *          rate
   */
  @OptionMetadata(
    displayName = "Learning rate schedule definition",
    description = "Definition of if-then-else schedule - format epoch:lr rate, "
      + "epoch:lr, ..., lr, interpreted as if epoch # < epoch then lr; "
      + "else if ...; else lr", commandLineParamName = "lr-schedule-def",
    commandLineParamSynopsis = "-lr-schedule-def <condition:rate,"
      + "condition:rate,...,rate>", displayOrder = 43,
    category = "Training callbacks")
  public
    void setLearningRateSchedule(String schedule) {
    m_lrScheduleDefinition = schedule;
  }

  /**
   * Get the schedule to use with the learning rate schedule callback. Format is
   * 'epoch:lr, epoch:lr, ..., lr', interpreted as 'if epoch # < epoch then lr;
   * else if ...; else lr'.
   *
   * @return a string defining an if-then-else schedule for the learning rate
   */
  public String getLearningRateSchedule() {
    return m_lrScheduleDefinition;
  }

  /**
   * Set whether to use model checkpoints (i.e. periodically save the model
   * during training)
   *
   * @param checkpoints true to use model checkpoints
   */
  @OptionMetadata(displayName = "Use model checkpoints",
    description = "Save model after each epoch",
    commandLineParamName = "checkpoints",
    commandLineParamSynopsis = "-checkpoints", commandLineParamIsFlag = true,
    displayOrder = 44, category = "Training callbacks")
  public void setUseModelCheckpoints(boolean checkpoints) {
    m_useModelCheckpoints = checkpoints;
  }

  /**
   * Get whether to use model checkpoints (i.e. periodically save the model
   * during training)
   *
   * @return true if using model checkpoints
   */
  public boolean getUseModelCheckpoints() {
    return m_useModelCheckpoints;
  }

  /**
   * Set the path to save checkpoint models to. Filename/path may include
   * variables that reference metrics (such as epoch or loss) - e.g.
   * '{epoch:02d}-{loss:.2f}'
   *
   * @param modelCheckpointsPath the path to save checkpoint models to.
   */
  @OptionMetadata(displayName = "Model checkpoints path",
    description = "Path to save checkpoint models to",
    commandLineParamName = "checkpoint-path",
    commandLineParamSynopsis = "-checkpoint-path <path string>",
    displayOrder = 45, category = "Training callbacks")
  public void setModelCheckpointsPath(String modelCheckpointsPath) {
    m_modelCheckpointPath = modelCheckpointsPath;
  }

  /**
   * Get the path to save checkpoint models to. Filename/path may include
   * variables that reference metrics (such as epoch or loss) - e.g.
   * '{epoch:02d}-{loss:.2f}'
   *
   * @return the path to save checkpoint models to.
   */
  public String getModelCheckpointsPath() {
    return m_modelCheckpointPath;
  }

  /**
   * Set the model checkpoint period, i.e. how often (in epochs) to save a model
   *
   * @param period how often to save a checkpoint model
   */
  @OptionMetadata(displayName = "Model checkpoints period",
    description = "How often (in epochs) to save a checkpoint model",
    commandLineParamName = "checkpoint-period",
    commandLineParamSynopsis = "-checkpoint-period <integer>",
    displayOrder = 46, category = "Training callbacks")
  public void setModelCheckpointsPeriod(String period) {
    m_modelCheckpointPeriod = period;
  }

  /**
   * Get the model checkpoint period, i.e. how often (in epochs) to save a model
   *
   * @return how often to save a checkpoint model
   */
  public String getModelCheckpointsPeriod() {
    return m_modelCheckpointPeriod;
  }

  /**
   * Set the name of the metric to monitor when saving checkpoint models.
   * Prevents the best saved model from being overwritten, unless outperformed.
   *
   * @param monitor the name of the metric to monitor
   */
  @OptionMetadata(displayName = "Model checkpoints metric to monitor",
    description = "Monitor this metric - prevents best saved model from being "
      + "overwritten, unless outperformed (loss, val_loss)",
    commandLineParamName = "checkpoint-monitor",
    commandLineParamSynopsis = "-checkpoint-monitor <[loss | val_loss]>",
    displayOrder = 47, category = "Training callbacks")
  public void setModelCheckpointsMonitor(String monitor) {
    m_modelCheckpointMonitor = monitor;
  }

  /**
   * Get the name of the metric to monitor when saving checkpoint models.
   * Prevents the best saved model from being overwritten, unless outperformed.
   *
   * @return the name of the metric to monitor
   */
  public String getModelCheckpointsMonitor() {
    return m_modelCheckpointMonitor;
  }

  /**
   * Set the number of GPUs to use (when GPUs are available). >1 will enable the
   * use of a Keras multi_gpu_model. Best results are when the number of GPUs
   * divide equally into the mini-batch size.
   *
   * @param numGPUs the number of GPUs to use
   */
  @OptionMetadata(displayName = "GPUs",
    description = "Number of GPUs to use (for best results, mini-batch size "
      + "should be divisible by number of GPUs)",
    commandLineParamName = "gpus",
    commandLineParamSynopsis = "-gpus <integer>", displayOrder = 48,
    category = "GPU")
  public void setNumGPUs(String numGPUs) {
    m_numGPUs = numGPUs;
  }

  /**
   * Get the number of GPUs to use (when GPUs are available). >1 will enable the
   * use of a Keras multi_gpu_model. Best results are when the number of * GPUs
   * divide equally into the mini-batch size.
   *
   * @return the number of GPUs to use
   */
  public String getNumGPUs() {
    return m_numGPUs;
  }

  /**
   * Set whether to turn off CPU-based weight merging when number of GPUs > 1.
   * Merging on the GPU can be useful when NVLink is available
   *
   * @param doNotCPUMerge true to turn off CPU merging of weights
   */
  @OptionMetadata(
    displayName = "Do not merge weights on CPU",
    description = "Multi-gpu: Do not force merging of model weights under the scope of "
      + "the CPU (useful when NVLink is available)",
    commandLineParamName = "gpu-turn-off-cpu-merge",
    commandLineParamSynopsis = "-gpu-turn-off-cpu-merge",
    commandLineParamIsFlag = true, displayOrder = 49, category = "GPU")
  public
    void setGPUDoNotCPUMerge(boolean doNotCPUMerge) {
    m_gpusDoNotMergeOnCPU = doNotCPUMerge;
  }

  /**
   * Get whether to turn off CPU-based weight merging when number of GPUs > 1.
   * Merging on the GPU can be useful when NVLink is available
   *
   * @return true if turning off CPU merging of weights
   */
  public boolean getGPUDoNotCPUMerge() {
    return m_gpusDoNotMergeOnCPU;
  }

  private void samplewiseCenterSetting(StringBuilder b) {
    b.append("samplewise_center=" + (m_samplewiseCenter ? "True" : "False")
      + ",");
  }

  private void samplewiseStdNormalizationSetting(StringBuilder b) {
    b.append("samplewise_std_normalization="
      + (m_samplewiseStdNormalization ? "True" : "False") + ",");
  }

  private void rotationRangeSetting(StringBuilder b) {
    b.append("rotation_range=" + environmentSubstitute(m_rotationRange))
      .append(",");
  }

  private void widthShiftRangeSetting(StringBuilder b) {
    b.append("width_shift_range=" + environmentSubstitute(m_widthShiftRange))
      .append(",");
  }

  private void heightShiftRangeSetting(StringBuilder b) {
    b.append("height_shift_range=" + environmentSubstitute(m_heightShiftRange))
      .append(",");
  }

  private void shearRangeSetting(StringBuilder b) {
    b.append("shear_range=" + environmentSubstitute(m_shearRange)).append(",");
  }

  private void zoomRangeSetting(StringBuilder b) {
    b.append("zoom_range=" + environmentSubstitute(m_zoomRange)).append(",");
  }

  private void channelShiftRangeSetting(StringBuilder b) {
    b.append(
      "channel_shift_range=" + environmentSubstitute(m_channelShiftRange))
      .append(",");
  }

  private void fillModeSetting(StringBuilder b) {
    String fillMode = environmentSubstitute(m_fillMode);
    if (!fillMode.startsWith("'")) {
      fillMode = "'" + fillMode;
    }
    if (!fillMode.endsWith("'")) {
      fillMode = fillMode + "'";
    }
    b.append("fill_mode=").append(fillMode).append(",");
  }

  private void cvalSetting(StringBuilder b) {
    b.append("cval=" + (m_cval != null ? environmentSubstitute(m_cval) : "0.0"))
      .append(",");
  }

  private void horizontalFlipSetting(StringBuilder b) {
    b.append("horizontal_flip=" + (m_horizontalFlip ? "True" : "False"))
      .append(",");
  }

  private void verticalFlipSetting(StringBuilder b) {
    b.append("vertical_flip=" + (m_verticalFlip ? "True" : "False"))
      .append(",");
  }

  private void rescaleSetting(StringBuilder b) {
    b.append("rescale=" + environmentSubstitute(m_rescale)).append(",");
  }

  private void validationSplitSetting(StringBuilder b) {
    String valSplit = environmentSubstitute(m_validationSplitPercent);
    try {
      double vS = Double.parseDouble(valSplit);
      if (vS > 0 && vS < 1) {
        b.append("validation_split=" + vS).append(",");
      }
    } catch (NumberFormatException e) {
      e.printStackTrace();
    }
  }

  /**
   * Does some validation of the input (training) data. Checks that the meta
   * training data contains a string attribute as the first attribute (values
   * contain paths, relative to the images directory, of images) and a nominal
   * class attribute as the second attribute.
   *
   * @param data the meta training data
   * @throws WekaException if there is a problem with the training data or
   *           images directory
   */
  protected void validateData(Instances data) throws WekaException {
    String resolved = getImagesLocation().toString();
    resolved = environmentSubstitute(resolved);
    File imagesLoc = new File(resolved);
    if (!imagesLoc.isDirectory()) {
      throw new WekaException("Directory not valid: " + resolved);
    }
    if (!(data.attribute(0).isString() && data.classIndex() == 1)) {
      throw new WekaException(
        "An dataset is required with a string attribute and a class attribute");
    }
  }

  /**
   * Generates the python code for imports needed.
   *
   * @param b the StringBuilder to add the imports to
   * @param clearSession whether to generate a keras.backend.clear_session()
   * @param zm the ZooModel being used
   */
  protected void generateImports(StringBuilder b, boolean clearSession,
    ZooModel zm) {
    String s = environmentSubstitute(m_seed);
    String modelPackage = getPackageForZooModel(zm);
    b.append("from numpy.random import seed\n")
      .append("seed(" + s + ")\n\n")
      .append(
        "import numpy as np\n"
          + "import pandas as pd\n"
          + "from math import ceil\n"
          + "from datetime import datetime\n"
          + "from keras.preprocessing.image import ImageDataGenerator\n"
          + "from keras.models import Sequential, Model\n"
          + "from keras.layers import Dropout, Flatten, GlobalAveragePooling2D, "
          + "AveragePooling2D, Dense\n"
          + "from keras import optimizers\n"
          + "from keras import applications\n"
          + "from "
          + modelPackage
          + " import preprocess_input\n"
          + "import keras.backend as K\n"
          + "from keras import utils\n"
          + "from keras.models import load_model\n"
          + "from keras.callbacks import Callback, CSVLogger, ReduceLROnPlateau, "
          + "LearningRateScheduler, ModelCheckpoint\n\n");

    if (clearSession) {
      b.append("K.clear_session()\n\n");
    }
  }

  /**
   * Generates the python code related to setting up ImageDataGenerator.
   *
   * @param b the StringBuilder to add the code to
   * @param testing true if this is for testing/prediction
   * @param validation true if this is for validation
   */
  protected void generateImageDataGenerator(StringBuilder b, boolean testing,
    boolean validation) {
    b.append(validation ? "datagen_validation" : "datagen");
    b.append("= ImageDataGenerator(");
    if (!m_dontUseModelSpecificImageProcFunc) {
      b.append("preprocessing_function=preprocess_input,");
    }

    rescaleSetting(b);
    samplewiseCenterSetting(b);
    samplewiseStdNormalizationSetting(b);

    if (!testing && !validation) {
      rotationRangeSetting(b);
      widthShiftRangeSetting(b);
      heightShiftRangeSetting(b);
      shearRangeSetting(b);
      zoomRangeSetting(b);
      channelShiftRangeSetting(b);
      fillModeSetting(b);
      cvalSetting(b);
      horizontalFlipSetting(b);
      verticalFlipSetting(b);
      validationSplitSetting(b);
    }

    b.setLength(b.length() - 1); // trim last ,
    b.append(")\n\n");
  }

  protected void checkAvailableGPUs() throws Exception{
    if (m_availableGPUs > -1) {
      return;
    }
    // now determine if multiple GPUs are available...
    PythonSession session = PythonSession.acquireSession(this);
    StringBuilder temp = new StringBuilder();
    temp.append("from keras import backend as K\n\n");
    temp
      .append("def _normalize_device_name(name):\n"
        + "    name = '/' + ':'.join(name.lower().replace('/', '').split(':')[-2:])\n"
        + "    return name\n\n");
    temp.append("z = [x.name for x in K.get_session().list_devices()]\n");
    temp
      .append("available_devices = [_normalize_device_name(name) for name in z]\n");
    temp
      .append("gpus = len([x for x in available_devices if '/gpu:' in x])\n");

    logMessage("Checking available gpus:\n\n" + temp.toString());
    List<String> outAndErr = session.executeScript(temp.toString(), getDebug());
    logOutAndErrFromPython(outAndErr);
    String actualG =
      session.getVariableValueFromPythonAsPlainString("gpus", getDebug());

    m_availableGPUs = Integer.parseInt(actualG);
    logMessage("Number of available GPUs: " + m_availableGPUs);
  }

  /**
   * Generates the python code related to setting up a mutli_gpu_model
   *
   * @param b the StringBuilder to add code to
   * @param modelName the name of the model in python
   * @throws Exception if a problem occurs
   */
  protected void generateMultiGPUModel(StringBuilder b, String modelName)
    throws Exception {
    if (m_numGPUs.length() > 0) {
      String nG = environmentSubstitute(m_numGPUs);
      int numG = 0;
      try {
        numG = Integer.parseInt(nG);
      } catch (NumberFormatException n) {
        throw new WekaException(n);
      }

      if (numG <= 1) {
        m_parallelWrapper = false;
        return;
      }

      if (numG % 2 != 0) {
        throw new WekaException("Must specify an even number of GPUs");
      }

      checkAvailableGPUs();

      if (m_availableGPUs <= 1) {
        m_parallelWrapper = false;
        return; // not enough GPUs available to use multi_gpu_model
      }
      if (m_availableGPUs < numG) {
        throw new WekaException("Requested number of GPUs (" + numG
          + ") is more " + "than are available (" + m_availableGPUs + ")");
      }

      b.append("p_" + modelName + " = ")
        .append("utils.multi_gpu_model(" + modelName)
        .append(", gpus=" + environmentSubstitute(m_numGPUs));

      if (m_gpusDoNotMergeOnCPU) {
        b.append(", cpu_merge=False)");
      } else {
        b.append(", cpu_merge=True)");
      }
      b.append("\n\n");
      m_parallelWrapper = true;
    }
  }

  /**
   * Generate the python code related to defining the zoo model
   *
   * @param b the StringBuilder to add code to
   * @param data the training instances
   */
  protected void generateModel(StringBuilder b, Instances data) {
    if (m_numGPUs.length() > 0 && !environmentSubstitute(m_numGPUs).equals("1")) {
      b.append("import tensorflow as tf\n");
      b.append("with tf.device('/cpu:0'):\n").append("    ");
    }
    b.append("keras_zoo_" + m_modelHash).append(" = applications.")
      .append(m_model.toString()).append("(");
    b.append("include_top=")
      .append(m_weightsType == WeightsType.None ? "True" : "False").append(",");
    b.append("weights=").append(
      m_weightsType == WeightsType.None ? "None" : "'"
        + m_weightsType.toString() + "'");
    if (m_weightsType == WeightsType.None) {
      b.append(",classes=").append(data.classAttribute().numValues());
    } else {
      String shape =
        environmentSubstitute(m_targetWidth) + ","
          + environmentSubstitute(m_targetHeight);
      b.append(",input_shape=(" + shape + ",3)");
    }
    b.append(")\n\n");
  }

  /**
   * Checks whether the supplied path for loading a .hdf5 serialized network
   * from is valid.
   *
   * @param loadFile the path to check
   * @return true if the supplied path is valid
   */
  protected boolean modelLoadPathValid(File loadFile) {
    // String loadPath = m_modelLoadPath.toString();
    String loadPath = loadFile.toString();
    if (loadPath != null && loadPath.length() > 0 && !loadPath.equals("-NONE-")) {
      loadPath = environmentSubstitute(loadPath);
      if (!loadPath.endsWith(".hdf5")) {
        loadPath += ".hdf5";
      }
      File file = new File(loadPath);
      return file.exists() && file.isFile() && file.length() > 0;
    }
    return false;
  }

  /**
   * Generate the python code related to loading a serialized network
   *
   * @param b the StringBuilder to add code to
   * @throws Exception
   */
  protected void generateLoadModel(StringBuilder b) throws Exception {
    String loadPath = m_modelLoadPath.toString();
    if (!modelLoadPathValid(m_modelLoadPath)) {
      boolean ok = true;
      if (m_modelLoadPath.toString().equals("-NONE-")) {
        // see if the save path is set
        if (!modelLoadPathValid(m_modelSavePath)) {
          ok = false;
        } else {
          logMessage("Model load path " + loadPath
            + " is not valid. Loading model " + "using the save path ("
            + m_modelSavePath.toString() + ") instead.");
          loadPath = m_modelSavePath.toString();
        }
      }
      if (!ok) {
        throw new WekaException("Model load path '"
          + environmentSubstitute(loadPath) + "' is not valid!");
      }
    }
    loadPath = environmentSubstitute(loadPath);
    if (!loadPath.endsWith(".hdf5")) {
      loadPath += ".hdf5";
    }

    loadPath = escapeWindowsBackslashes(loadPath);

    int nG = 0;
    if (m_numGPUs.length() > 0) {
      String numG = environmentSubstitute(m_numGPUs);
      try {
        nG = Integer.parseInt(numG);
      } catch (NumberFormatException n) {
        throw new WekaException(n);
      }

      if (nG >= 2) {
        if (nG % 2 != 0) {
          throw new WekaException("Must use an even number of GPUs");
        }
        if (m_availableGPUs >= nG) {
          b.append("import tensorflow as tf\n");
          b.append("with tf.device('/cpu:0'):\n").append("    ");
        }
      }
    }
    b.append("keras_zoo_" + m_modelHash + " = ").append("load_model(")
      .append("'" + loadPath + "')\n\n");

    m_pythonModelPrefix = "keras_zoo";
  }

  /**
   * Generate the python code related to freezing layers, adding top-level dense
   * and pooling layers etc.
   *
   * @param b the StringBuilder to add the code to
   * @param data the training instances
   */
  protected void generateTransferLayers(StringBuilder b, Instances data) {
    // first freeze base layers
    b.append("for layer in keras_zoo_" + m_modelHash).append(".layers:\n")
      .append("    layer.trainable=False\n\n");

    // then add dense layers etc.
    b.append("x = keras_zoo_" + m_modelHash).append(".output\n");
    if (m_globalAveragePoolingLayer) {
      b.append("x = GlobalAveragePooling2D()(x)\n\n");
    } else {
      if (m_averagePoolingLayer) {
        b.append("x = AveragePooling2D(pool_size=("
          + environmentSubstitute(m_poolingSize) + "))(x)\n");
      }
      b.append("x = Flatten()(x)\n\n");
    }
    String fcL = environmentSubstitute(m_fcl);
    if (m_fcl != null && m_fcl.length() > 0 && !fcL.equals("0")) {
      b.append("fc_layers = ").append("[" + environmentSubstitute(m_fcl) + "]")
        .append("\n");
      b.append("for fc in fc_layers:\n");
      b.append("    x = Dense(fc, activation='relu')(x)\n");
      if (m_dropOutsBetweenFCL) {
        b.append("    x = Dropout(")
          .append(environmentSubstitute(m_dropOutRate)).append(")(x)\n\n");
      }
    }

    // output layer
    b.append("preds = Dense(" + data.classAttribute().numValues())
      .append(", activation='softmax')(x)").append("\n\n");
    b.append("keras_zoo_transfer_" + m_modelHash)
      .append(" = Model(inputs=keras_zoo_" + m_modelHash)
      .append(".input, outputs=preds)\n\n");
  }

  /**
   * Generate the python code related to compiling the model
   *
   * @param b the StringBuilder to add the code to
   * @param modelName the variable name holding the model in python
   * @param loadModel true if the model has been loaded
   */
  protected void generateCompileModel(StringBuilder b, String modelName,
    boolean loadModel) {
    // compile model categorical_crossentropy loss optimizer, metrics
    Optimizer toUse =
      m_weightsType != WeightsType.None && m_finetuneNetwork && loadModel ? m_finetuneOptimizer
        : m_optimizer;
    String optimizerOpts =
      m_weightsType != WeightsType.None && m_finetuneNetwork && loadModel ? m_finetuneOptimizerOptions
        : m_optimizerOptions;
    String optimizer =
      toUse.toString()
        + "("
        + (optimizerOpts != null && optimizerOpts.length() > 0 ? environmentSubstitute(optimizerOpts)
          : "") + ")";
    String metrics =
      "['accuracy'" + (m_topKMetric ? ",'top_k_categorical_accuracy']" : "]");
    if (!m_parallelWrapper || loadModel) {
      b.append("optimizer=optimizers." + optimizer).append("\n");
    }
    b.append(modelName + "_" + m_modelHash)
      .append(".compile(")
      .append("optimizer")
      .append(", loss='categorical_crossentropy', metrics=" + metrics + ")\n\n");
  }

  private String makeClassList(Instances data) {
    Attribute classAtt = data.classAttribute();
    String classList = "[";
    for (int i = 0; i < classAtt.numValues(); i++) {
      classList += "'" + classAtt.value(i) + "',";
    }

    classList = classList.substring(0, classList.length() - 1);
    classList += "]";

    return classList;
  }

  /**
   * Generate the python code related to flowing images from a pandas data
   * frame.
   *
   * @param b the StringBuilder to add code to
   * @param data the training instances
   * @param test true if this is test/prediction code
   * @param validation true if this is flowing images for validation purposes
   * @throws Exception if a problem occurs
   */
  protected void generateFlowFromDataframe(StringBuilder b, Instances data,
    boolean test, boolean validation) throws Exception {
    String imageDirectory = m_imageDirectory.toString();
    imageDirectory = environmentSubstitute(imageDirectory);

    imageDirectory = escapeWindowsBackslashes(imageDirectory);

    String shape =
      m_weightsType == WeightsType.None ? m_model.getDefaultShape()
        : environmentSubstitute(m_targetWidth) + ","
          + environmentSubstitute(m_targetHeight);

    String dataPrefix =
      test ? "keras_zoo_test_" : (validation ? "keras_zoo_validation_"
        : "keras_zoo_train_");
    String classMode = test ? "None" : "'categorical'"; // only data, no labels
                                                        // for test

    b.append(dataPrefix + m_modelHash + "['" + data.attribute(0).name() + "']")
      .append(
        " = " + dataPrefix + m_modelHash + "['" + data.attribute(0).name()
          + "'].astype(str)\n");
    b.append(dataPrefix + m_modelHash + "['" + data.attribute(1).name() + "']")
      .append(
        " = " + dataPrefix + m_modelHash + "['" + data.attribute(1).name()
          + "'].astype(str)\n\n");

    String bs = environmentSubstitute(m_batchSize);
    if (validation) {
      try {
        int vbs = Integer.parseInt(bs);
        int numVal = getNumInstancesInValidation(data);
        // now adjust batch size if necessary in order to equally divide into
        // the number of validation set instances
        vbs = computeValBatchSize(vbs, numVal);
        bs = "" + vbs;
      } catch (NumberFormatException e) {
        throw new WekaException(e);
      }
    }

    b.append(
      (validation ? "valid_generator = datagen_validation"
        : "generator = datagen")
        + ".flow_from_dataframe("
        + dataPrefix
        + m_modelHash + ", directory='").append(imageDirectory)
      .append("', x_col='" + data.attribute(0).name()).append("'")
      .append(", y_col='" + data.attribute(1).name())
      .append("', target_size=(" + shape + ")").append(", classes=")
      .append(makeClassList(data))
      .append(", class_mode=" + classMode)
      .append((test ? ", shuffle=False" : ", shuffle=True"))
      // don't shuffle
      // test data!!
      .append(", batch_size=").append(bs)
      .append(", seed=" + environmentSubstitute(m_seed)).append(")\n\n");
  }

  /**
   * Generates the python code for a custom epoch time callback. Allows the
   * completion time for an epoch to be output to the CSV log.
   *
   * @return the code for a custom epoch time callback
   */
  protected String generateCustomEpochTimeCallback() {
    return "class ComputeDeltaTime(Callback):\n"
      + "    def on_epoch_end(self, epoch, logs):\n"
      + "        logs['time'] = datetime.now().time()\n\n"
      + "epoch_time = ComputeDeltaTime()\n\n";
  }

  /**
   * Generates the python code for a CSV callback.
   *
   * @param logAppend true if the callback should append to the CSV file rather
   *          than creating a new file
   * @return the code for a CSV callback
   */
  protected String generateCSVCallback(boolean logAppend) {
    String logF = m_trainingLogFile.toString();
    logF = environmentSubstitute(logF);
    File f = new File(logF);
    if (!logAppend && f.exists() && f.isFile()) {
      f.delete();
    }
    logF = escapeWindowsBackslashes(logF);
    return "csv_logger = CSVLogger('" + logF + "'" + ",append="
      + (logAppend ? "True" : "False") + ",separator=',')";
  }

  /**
   * Generates the python code for various user requested callbacks
   *
   * @param b the StringBuilder to add the code to
   * @param logAppend true if the CSV log is being appended to (i.e. training
   *          continues with a previously saved mode)
   * @param callbacksInUse an array for returning the code of callbacks in use
   */
  protected void generateCallbacks(StringBuilder b, boolean logAppend,
    String[] callbacksInUse) {
    String csvCallback = "";
    String epochTimeCallback = "";
    String logF = environmentSubstitute(m_trainingLogFile.toString());
    if (logF != null && logF.length() > 0) {
      epochTimeCallback = generateCustomEpochTimeCallback();
      csvCallback = generateCSVCallback(logAppend);
      b.append(epochTimeCallback).append(csvCallback).append("\n");
    }

    String reduceLR = "";
    if (m_reduceLRCallback) {
      String monitor = "'loss'";
      if (m_validationSplitPercent != null
        && m_validationSplitPercent.length() > 0) {
        monitor = "'val_loss'";
      }
      reduceLR =
        "reduce_lr = ReduceLROnPlateau(monitor=" + monitor
          + ",verbose=1,factor=" + environmentSubstitute(m_reduceLRfactor)
          + ",patience=" + environmentSubstitute(m_reduceLRpatience)
          + ",min_delta=" + environmentSubstitute(m_reduceLRMinDelta)
          + ",cooldown=" + environmentSubstitute(m_reduceLRCooldown)
          + ",min_lr=" + environmentSubstitute(m_reduceLRMinLR) + ")";
      b.append(reduceLR).append("\n");
    }

    String lrSchedule =
      m_lrScheduleDefinition == null ? ""
        : environmentSubstitute(m_lrScheduleDefinition);
    if (m_learningRateSchedule && lrSchedule.length() > 0) {
      String lrS = lrSchedule;
      double defaultLR = -1;
      String[] parts = lrS.split(",");
      if (parts.length < 2) {
        logMessage("Warning: learning rate schedule string has fewer than "
          + "two conditions - ignoring");
        lrSchedule = "";
      } else {
        lrSchedule = "def schedule(epoch):\n";
        boolean first = true;
        for (String p : parts) {
          String[] s = p.split(":");
          if (first && s.length != 2) {
            logMessage("Warning: first entry in learning rate schedule must be "
              + "conditional - ignoring");
            lrSchedule = "";
            break;
          } else if (first) {
            lrSchedule +=
              "    if epoch < " + s[0].trim() + ":\n" + "        return "
                + s[1].trim() + "\n";
            first = false;
          } else {
            if (s.length == 2) {
              lrSchedule +=
                "    elif epoch < " + s[0].trim() + ":\n" + "        return "
                  + s[1].trim() + "\n";
            } else {
              lrSchedule += "    else:\n        return " + s[0].trim() + "\n";
              break; // ignore any other parts after this
            }
          }
        }
        if (lrSchedule.length() > 0) {
          lrSchedule += "\n";
          lrSchedule +=
            "lr_scheduler = LearningRateScheduler(schedule, verbose=1)\n\n";
          b.append(lrSchedule);
        }
      }
    } else {
      lrSchedule = "";
    }

    String modelCheckpointsPath = "";
    if (m_useModelCheckpoints) {
      modelCheckpointsPath = environmentSubstitute(m_modelCheckpointPath);
      String modelCheckpointsPeriod =
        environmentSubstitute(m_modelCheckpointPeriod);
      String modelCheckpointsMonitor =
        environmentSubstitute(m_modelCheckpointMonitor);
      b.append("checkpointer = ModelCheckpoint(")
        .append("filepath='" + modelCheckpointsPath + "', ")
        .append("verbose=1, save_best_only=True, ")
        .append("period=" + modelCheckpointsPeriod)
        .append(", monitor='" + modelCheckpointsMonitor + "')\n\n");
    }

    callbacksInUse[0] = epochTimeCallback;
    callbacksInUse[1] = csvCallback;
    callbacksInUse[2] = reduceLR;
    callbacksInUse[3] = lrSchedule;
    callbacksInUse[4] = modelCheckpointsPath;
  }

  /**
   * Computes the batch size to use with the validation data (either percentage
   * split of the training data or a separate validation set). We attempt to set
   * the batch size to something that divides equally into the number of
   * instances in the validation data. This ensures that each validation
   * instance is evaluated exactly once.
   *
   * @param valBS the requested batch size
   * @param numVal the number of instances/images in the validation set
   * @return the batch size to use
   * @throws Exception if a problem occurs
   */
  protected int computeValBatchSize(int valBS, int numVal) throws Exception {
    int valStepSize = 1;

    if (valBS > numVal) {
      throw new WekaException("Batch size can't be larger than the number "
        + "of instances in the validation set/split: " + numVal);
    } else {
      double diff = ((double) numVal / (double) valBS) - (numVal / valBS);
      while (diff > 0) {
        valBS /= 2;
        if (valBS < 1) {
          valBS = 1;
          break;
        }
        diff = ((double) numVal / (double) valBS) - (numVal / valBS);
      }
    }

    return valBS;
  }

  /**
   * Get the number of instances in the validation data.
   *
   * @param trainData the training data (if percentage split validation is being
   *          used)
   * @return the number of instances in the validation data
   * @throws Exception
   */
  protected int getNumInstancesInValidation(Instances trainData)
    throws Exception {
    String valSplit = environmentSubstitute(m_validationSplitPercent);
    double vS = 0;
    try {
      vS = Double.parseDouble(valSplit);
    } catch (NumberFormatException e) {
      throw new WekaException(e);
    }
    if (vS > 0 || m_separateValidationDataSetInPython) {
      String batchSize = environmentSubstitute(m_batchSize);
      int bS = 0;
      try {
        bS = Integer.parseInt(batchSize);
      } catch (NumberFormatException e) {
        throw new WekaException(e);
      }
      return m_separateValidationDataSetInPython ? m_separateValidationDataSetNumInstances
        : (int) (trainData.numInstances() * vS);
    }
    return 0;
  }

  /**
   * Generate the python code for training the network.
   *
   * @param b the StringBuilder to add the code to
   * @param data the training instances
   * @param modelName the name of the variable holding the model in python
   * @param logAppend true if the CSV log is to be appended to (i.e. training is
   *          continuing with a saved model)
   * @param loadModel true if an existing model is being loaded
   * @throws Exception if a problem occurs
   */
  protected void generateNetworkTrain(StringBuilder b, Instances data,
    String modelName, boolean logAppend, boolean loadModel) throws Exception {

    if (m_finetuneNetwork && loadModel && m_weightsType != WeightsType.None) {
      generateUnfreezeLayers(b, modelName);
    }

    if (!loadModel) {
      generateCompileModel(b, modelName, loadModel);
    }

    generateMultiGPUModel(b, modelName + "_" + m_modelHash);
    if (m_parallelWrapper) {
      generateCompileModel(b, "p_" + modelName, loadModel);
    }

    String[] callbacksInUse = { "", "", "", "", "" };
    generateCallbacks(b, logAppend, callbacksInUse);

    String trainingSteps =
      "ceil(" + data.numInstances() + ".0 / "
        + environmentSubstitute(m_batchSize) + ")";
    String validationSteps = "";
    // check validation split...
    if ((m_validationSplitPercent != null && m_validationSplitPercent.length() > 0)
      || m_separateValidationDataSetInPython) {
      String valSplit = environmentSubstitute(m_validationSplitPercent);
      double vS = Double.parseDouble(valSplit);
      if (vS > 0 || m_separateValidationDataSetInPython) {
        String batchSize = environmentSubstitute(m_batchSize);
        int bS = Integer.parseInt(batchSize);
        int numVal =

        getNumInstancesInValidation(data);
        int numTrain =
          m_separateValidationDataSetInPython ? data.numInstances() : data
            .numInstances() - numVal;

        if (bS > numTrain) {
          throw new WekaException(
            "Batch size can't be larger than the number of "
              + "instances in the training split: " + numTrain);
        }

        // need to have a batch size for validation that divides equally into
        // the size of the validation set (so that instances are sampled exactly
        // once)
        int valBS = computeValBatchSize(bS, numVal);

        trainingSteps = "ceil(" + numTrain + ".0 / " + bS + ")";
        validationSteps = "" + (numVal / valBS);
      }
    }

    String startE = "";
    String numE = environmentSubstitute(m_numEpochs);
    int sE = 0;
    int nE = 0;
    if (loadModel) {
      startE = environmentSubstitute(m_initialEpoch);
      try {
        sE = Integer.parseInt(startE);
        nE = Integer.parseInt(numE);
        numE = "" + (sE + nE);
      } catch (NumberFormatException e) {
        startE = "";
      }
    }

    generateFitGenerator(b, m_parallelWrapper ? "p_" + modelName : modelName,
      loadModel, startE, numE, trainingSteps, validationSteps, callbacksInUse);

    // only generate the fine tuning code if we are not loading the model. I.e.
    // A loaded model will (if fine tuning was turned on originally) already be
    // configured for fine tuning
    if (m_finetuneNetwork && !loadModel && m_weightsType != WeightsType.None) {
      generateFinetuneNetwork(b, modelName);
      // update start and end epochs
      try {
        nE = Integer.parseInt(numE);
      } catch (NumberFormatException e) {
        // ignore
      }
      sE = sE + nE;
      startE = "" + sE;
      numE = "" + (sE + nE);
      if (!m_finetuneUseLRScheduleIfSet) {
        callbacksInUse[3] = "";
      }
      if (callbacksInUse[1].length() > 0) {
        callbacksInUse[1] = generateCSVCallback(true);
        b.append("\n").append(callbacksInUse[1]).append("\n\n");
      }

      generateFitGenerator(b, m_parallelWrapper ? "p_" + modelName : modelName,
        loadModel, startE, numE, trainingSteps, validationSteps, callbacksInUse);
    }

    if (m_printLayerIndexes) {
      b.append(
        "for i, layer in enumerate(" + modelName + "_" + m_modelHash
          + ".layers):")
        .append(
          "\n    print(i, layer.name, type(layer), 'trainable =', layer.trainable)")
        .append("\n\n");
    } else {
      b.append(modelName + "_" + m_modelHash).append(".summary()\n\n");
    }
  }

  /**
   * Generate the python code to unfreeze the layers involved in fine tuning
   * when performing transfer learning.
   *
   * @param b the StringBuilder to add the code to
   * @param modelName the name of the variable that holds the model in python
   */
  public void generateUnfreezeLayers(StringBuilder b, String modelName) {
    String finetuneIndex = environmentSubstitute(m_finetuneTopLayersIndex);
    b.append("for layer in " + modelName + "_" + m_modelHash + ".layers[:"
      + finetuneIndex + "]:\n");
    b.append("    layer.trainable=False\n");
    b.append("for layer in " + modelName + "_" + m_modelHash + ".layers["
      + finetuneIndex + ":]:\n");
    b.append("    layer.trainable=True\n\n");
  }

  /**
   * Generate the python code to perform network fine tuning when transfer
   * learning
   *
   * @param b the StringBuilder to add the code to
   * @param modelName the of the variable that holds the model in python
   * @throws Exception if a problem occurs
   */
  public void generateFinetuneNetwork(StringBuilder b, String modelName)
    throws Exception {
    generateUnfreezeLayers(b, modelName);

    String optimizer =
      m_finetuneOptimizer.toString()
        + "("
        + (m_finetuneOptimizerOptions != null
          && m_finetuneOptimizerOptions.length() > 0 ? environmentSubstitute(m_finetuneOptimizerOptions)
          : "") + ")";
    String metrics =
      "['accuracy'" + (m_topKMetric ? ",'top_k_categorical_accuracy']" : "]");

    generateMultiGPUModel(b, modelName + m_modelHash);
    modelName = m_parallelWrapper ? "p_" + modelName : modelName;

    b.append(modelName + "_" + m_modelHash)
      .append(".compile(optimizer=optimizers." + optimizer)
      .append(", loss='categorical_crossentropy', metrics=" + metrics + ")\n\n");

  }

  /**
   * Generate the python code for fitting the model from a generator
   *
   * @param b the StringBuilder to add the code to
   * @param modelName the name of the variable that holds the model in python
   * @param loadModel true if the model has been loaded
   * @param startE the initial (start) epoch to continue training from
   * @param numE the number of epochs to perform
   * @param trainingSteps the number of training steps to perform per epoch
   * @param validationSteps the number of validation steps to perform per epoch
   *          (if validation data is being used)
   * @param callbacksInUse the callbacks in use
   */
  public void generateFitGenerator(StringBuilder b, String modelName,
    boolean loadModel, String startE, String numE, String trainingSteps,
    String validationSteps, String[] callbacksInUse) {

    String epochTimeCallback = callbacksInUse[0];
    String csvCallback = callbacksInUse[1];
    String reduceLR = callbacksInUse[2];
    String lrSchedule = callbacksInUse[3];
    String modelCheckpoint = callbacksInUse[4];

    b.append(modelName + "_" + m_modelHash).append(
      ".fit_generator(generator=generator, steps_per_epoch=" + trainingSteps);

    if (validationSteps.length() > 0) {
      String gen =
        m_separateValidationDataSetInPython ? "valid_generator" : "generator";
      b.append(", validation_data=" + gen).append(
        ", validation_steps=" + validationSteps);
    }
    if (startE.length() > 0) {
      b.append(", initial_epoch=" + startE);
    }
    b.append(", epochs=" + numE).append(
      ", max_queue_size=" + environmentSubstitute(m_maxQueueSize));

    if (getWorkers().length() > 0
      && !environmentSubstitute(getWorkers()).equals("1")) {
      b.append(", workers=" + environmentSubstitute(getWorkers()));
    }
    if (csvCallback.length() > 0 || reduceLR.length() > 0
      || lrSchedule.length() > 0 || modelCheckpoint.length() > 0) {
      b.append(", callbacks=[");
      if (csvCallback.length() > 0) {
        b.append("epoch_time,");
        b.append("csv_logger").append(
          reduceLR.length() > 0 || lrSchedule.length() > 0
            || modelCheckpoint.length() > 0 ? "," : "");
      }
      if (reduceLR.length() > 0) {
        b.append("reduce_lr").append(
          lrSchedule.length() > 0 || modelCheckpoint.length() > 0 ? "," : "");
      }
      if (lrSchedule.length() > 0) {
        b.append("lr_scheduler")
          .append(modelCheckpoint.length() > 0 ? "," : "");
      }
      if (modelCheckpoint.length() > 0) {
        b.append("checkpointer");
      }
      b.append("]");
    }
    b.append(")\n\n");
  }

  /**
   * Generates the python code for saving the network to a .hdf5 file
   *
   * @param b the StringBuilder to add the code to
   * @param modelName the name of the variable that holds the model in python
   */
  protected void generateNetworkSave(StringBuilder b, String modelName) {
    if (m_modelSavePath != null && m_modelSavePath.toString().length() > 0
      && !m_modelSavePath.toString().equals("-NONE-")) {
      String savePath = environmentSubstitute(m_modelSavePath.toString());
      if (!savePath.endsWith(".hdf5")) {
        savePath += ".hdf5";
      }

      savePath = escapeWindowsBackslashes(savePath);

      b.append(modelName + "_" + m_modelHash).append(
        ".save('" + savePath + "')\n\n");
    }
  }

  /**
   * Generates the python code for training the network
   *
   * @param data the training data
   * @return the complete code for training the network
   * @throws Exception if a problem occurs
   */
  public String generateTrainingCode(Instances data) throws Exception {
    if (m_modelHash == null) {
      m_modelHash = "" + hashCode();
    }
    checkAvailableGPUs();

    boolean loadModel =
      getContinueTraining() && modelLoadPathValid(m_modelLoadPath);
    boolean logAppend = loadModel;

    StringBuilder toExecute = new StringBuilder();
    generateImports(toExecute, true, m_model);
    generateImageDataGenerator(toExecute, false, false);

    if (m_separateValidationDataSetInPython) {
      generateImageDataGenerator(toExecute, false, true);
    }

    if (loadModel) {
      generateLoadModel(toExecute);
    } else {
      generateModel(toExecute, data);
    }
    generateFlowFromDataframe(toExecute, data, false, false);
    if (m_separateValidationDataSetInPython) {
      generateFlowFromDataframe(toExecute, data, false, true);
    }

    if (m_weightsType != WeightsType.None && !loadModel) {
      generateTransferLayers(toExecute, data);

      // training code...
      generateNetworkTrain(toExecute, data, "keras_zoo_transfer", logAppend,
        false);
      generateNetworkSave(toExecute, "keras_zoo_transfer");
      m_pythonModelPrefix = "keras_zoo_transfer";
    } else {
      // generateCompileModel(toExecute, "keras_zoo", loadModel);
      generateNetworkTrain(toExecute, data, "keras_zoo", logAppend, loadModel);
      generateNetworkSave(toExecute, "keras_zoo");
      m_pythonModelPrefix = "keras_zoo";
    }

    // clean up
    toExecute.append("del " + m_pythonModelPrefix + "_" + m_modelHash).append("\n");
    toExecute.append("K.clear_session()\n\n");

    return toExecute.toString();
  }

  /**
   * Generates the python code for testing/prediction
   *
   * @param data a batch of test data
   * @param loadModel true to generate code to load the model first
   * @return the code for testing/prediction
   * @throws Exception if a problem occurs
   */
  public String generateTestingCode(Instances data, boolean loadModel)
    throws Exception {

    checkAvailableGPUs();

    StringBuilder b = new StringBuilder();
    generateImports(b, true, m_model);
    generateImageDataGenerator(b, true, false);
    if (loadModel) {
      if (m_modelHash == null) {
        m_modelHash = "" + hashCode();
      }
      generateLoadModel(b);
    }

    generateMultiGPUModel(b, m_pythonModelPrefix + "_" + m_modelHash);

    generateFlowFromDataframe(b, data, true, false);

    String parallel = m_parallelWrapper ? "p_" : "";
    b.append("predictions = ")
      .append(parallel + m_pythonModelPrefix + "_" + m_modelHash)
      .append(
        ".predict_generator(generator=generator, steps=ceil("
          + data.numInstances() + ".0 / " + environmentSubstitute(m_batchSize)
          + ")").append(")\n").append("predictions = predictions.tolist()\n\n");
    // b.append("import json\nr = json.dumps(preds)\nl = len(r)\n");

    // clean up
    b.append("del " + parallel + m_pythonModelPrefix + "_" + m_modelHash).append("\n");
    b.append("K.clear_session()").append("\n\n");

    return b.toString();
  }

  /**
   * Perform some initialization.
   *
   * @param data the training data
   * @return a PythonSession to use
   * @throws Exception if a problem occurs
   */
  protected PythonSession initialize(Instances data) throws Exception {
    getCapabilities().testWithFail(data);
    try {
      validateData(data);
    } catch (WekaException e) {
      throw new Exception(e);
    }
    m_zeroR = null;
    m_separateValidationDataSetInPython = false;

    if (m_modelHash == null) {
      m_modelHash = "" + hashCode();
    }

    data = new Instances(data);
    data.deleteWithMissingClass();

    m_zeroR = new ZeroR();
    m_zeroR.buildClassifier(data);

    m_classPriors =
      data.numInstances() > 0 ? m_zeroR.distributionForInstance(data
        .instance(0)) : new double[data.classAttribute().numValues()];
    if (data.numInstances() == 0 || data.numAttributes() == 1) {
      if (data.numInstances() == 0) {
        System.err
          .println("No instances with non-missing class - using ZeroR model");
      } else {
        System.err.println("Only the class attribute is present in "
          + "the data - using ZeroR model");
      }
      return null;
    } else {
      m_zeroR = null;
    }

    /* // check for empty classes
    AttributeStats stats = data.attributeStats(data.classIndex());
    m_nominalEmptyClassIndexes = new boolean[data.classAttribute().numValues()];
    for (int i = 0; i < stats.nominalWeights.length; i++) {
      if (stats.nominalWeights[i] == 0) {
        m_nominalEmptyClassIndexes[i] = true;
      }
    } */

    PythonSession session = PythonSession.acquireSession(this);
    session.instancesToPython(data, "keras_zoo_train_" + m_modelHash,
      getDebug());

    String separateVal = environmentSubstitute(m_validationFile.toString());
    if (separateVal.length() > 0 && new File(separateVal).exists()) {
      Instances valData = null;
      if (separateVal.toLowerCase().endsWith(".arff")) {
        valData =
          new Instances(new BufferedReader(new FileReader(separateVal)));
      } else {
        Attribute classA = data.classAttribute();
        String imageAttName =
          classA.index() == 0 ? data.attribute(1).name() : data.attribute(0)
            .name();
        String classAttName = classA.name();
        CSVLoader loader = new CSVLoader();
        loader.setFile(new File(separateVal));
        String opts = "-S " + Math.abs(classA.index() - 1) + 1;
        opts +=
          " -N " + (classA.index() + 1) + " L " + (classA.index() + 1) + ":";
        opts += concatNomVals(classA);
        loader.setOptions(Utils.splitOptions(opts));
        valData = loader.getDataSet();
      }
      if (valData.numInstances() > 0) {
        session.instancesToPython(valData, "keras_zoo_validation_"
          + m_modelHash, getDebug());
        m_separateValidationDataSetInPython = true;
        m_separateValidationDataSetNumInstances = valData.numInstances();
      }
    }

    return session;
  }

  private String concatNomVals(Attribute nom) {
    String result = "";

    for (int i = 0; i < nom.numValues(); i++) {
      result += nom.value(i) + ",";
    }
    result = result.substring(0, result.length() - 1);
    return result;
  }

  /**
   * Checks that python and keras are available
   */
  private void checkPythonEnvironment() {
    m_pythonAvailable = true;
    m_kerasInstalled = true;
    if (!PythonSession.pythonAvailable()) {
      // try initializing
      try {
        if (!PythonSession.initSession("python", getDebug())) {
          System.err.println("Python not available!!!!!!!!!!");
          m_pythonAvailable = false;
        }
      } catch (WekaException ex) {
        ex.printStackTrace();
        m_pythonAvailable = false;
      }
    }

    if (m_pythonAvailable) {
      // check for keras
      try {
        PythonSession session = PythonSession.acquireSession(this);
        String script = "import keras\n";
        List<String> outAndErr = session.executeScript(script, getDebug());
        if (outAndErr.get(1).length() > 0
          && !outAndErr.get(1).toLowerCase().contains("warning")
          && !outAndErr.get(1).toLowerCase().contains("using tensorflow")) {
          m_kerasInstalled = false;
          System.err
            .println("Keras does not seem to be available in the python "
              + "environment : \n" + outAndErr.get(1));
        }
      } catch (WekaException e) {
        m_kerasInstalled = false;
        e.printStackTrace();
      } finally {
        PythonSession.releaseSession(this);
      }
    } else {
      System.err.println("The python environment is either not available or "
        + "is not configured correctly:\n\n"
        + PythonSession.getPythonEnvCheckResults());
    }
  }

  /**
   * Get the capabilities of this classifier
   *
   * @return the capabilities of this classifier
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    result.enable(Capabilities.Capability.STRING_ATTRIBUTES);
    result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capabilities.Capability.NOMINAL_CLASS);
    result.enable(Capabilities.Capability.BINARY_CLASS);

    return result;
  }

  /**
   * Builds the classifier
   *
   * @param data set of instances serving as training data
   * @throws Exception if a problem occurs
   */
  @Override
  public void buildClassifier(Instances data) throws Exception {
    if (m_env == null) {
      m_env = Environment.getSystemWide();
    }

    checkPythonEnvironment();
    PythonSession session = initialize(data);

    if (!m_pythonAvailable) {
      String message =
        "The python environment is either not available or is "
          + "not configured correctly:\n\n"
          + PythonSession.getPythonEnvCheckResults();
      logMessage(message);
      throw new WekaException(message);
    }

    if (!m_kerasInstalled) {
      String message =
        "Keras does not seem to be available in your python " + "environment.";
      logMessage(message);
      throw new WekaException(message);
    }
    m_parallelWrapper = false;
    m_availableGPUs = -1;

    // Zero epochs is OK if model load path is valid. I.e user just wan't to
    // evaluate existing model (h5) without training (or having a pre-existing
    // serialized KerasZooClassifier)
    String epochs = environmentSubstitute(getNumEpochs());
    int eP = 0;
    try {
      eP = Integer.parseInt(epochs);
    } catch (NumberFormatException e) {
      throw new WekaException(e);
    }
    if (eP <= 0) {
      if (modelLoadPathValid(m_modelLoadPath)) {
        logMessage("Skipping training");
        return;
      } else {
        throw new WekaException("Number of epochs needs to be >= 1");
      }
    }

    try {
      if (session != null) {
        String code = generateTrainingCode(data);
        logMessage("KerasZooClassifier - generated training code:\n\n" + code);

        if (m_trainingLogFile != null
          && m_trainingLogFile.toString().length() > 0 && m_log != null
          && m_logMonitor == null) {
          String logF = environmentSubstitute(m_trainingLogFile.toString());
          m_logMonitor = new LogMonitor(logF, m_log);
          m_logMonitor.start();
        }

        List<String> outAndErr = session.executeScript(code, getDebug());
        logOutAndErrFromPython(outAndErr);

        // shutdown log monitor (if necessary)
        if (m_logMonitor != null) {
          m_logMonitor.stop();
          m_logMonitor = null;
        }
      }
    } finally {
      if (session != null) {
        PythonSession.releaseSession(this);
      }
    }
  }

  /**
   * Perform batch scoring with the default ZeroR model
   *
   * @param insts the training data
   * @return a batch of predictions
   * @throws Exception if a problem occurs
   */
  private double[][] batchScoreWithZeroR(Instances insts) throws Exception {
    double[][] result = new double[insts.numInstances()][];

    for (int i = 0; i < insts.numInstances(); i++) {
      Instance current = insts.instance(i);
      result[i] = m_zeroR.distributionForInstance(current);
    }

    return result;
  }

  /**
   * Predict a test instance
   *
   * @param instance the instance to be classified
   * @return a probability distribution over the classes
   * @throws Exception if a problem occurs
   */
  @Override
  public double[] distributionForInstance(Instance instance) throws Exception {
    Instances temp = new Instances(instance.dataset(), 0);
    temp.add(instance);

    return distributionsForInstances(temp)[0];
  }

  /**
   * Predict a batch of test instances
   *
   * @param insts the instances to get predictions for
   * @return an array of probability distributions
   * @throws Exception if a problem occurs
   */
  @SuppressWarnings("unchecked")
  @Override
  public double[][] distributionsForInstances(Instances insts) throws Exception {
    if (m_env == null) {
      m_env = Environment.getSystemWide();
    }

    if (m_zeroR != null) {
      return batchScoreWithZeroR(insts);
    }

    checkPythonEnvironment();

    if (!m_pythonAvailable) {
      String message =
        "The python environment is either not available or is "
          + "not configured correctly:\n\n"
          + PythonSession.getPythonEnvCheckResults();
      logMessage(message);
      throw new WekaException(message);
    }

    if (!m_kerasInstalled) {
      String message =
        "Keras does not seem to be available in your python " + "environment.";
      logMessage(message);
      throw new WekaException(message);
    }

    double[][] results = null;
    try {
      PythonSession session = PythonSession.acquireSession(this);
      // check to see if model is set in python
      boolean loadModel =
        !session.checkIfPythonVariableIsSet(m_pythonModelPrefix + "_"
          + m_modelHash, getDebug());

      if (loadModel) {
        m_availableGPUs = -1;
      }

      String toExecute = generateTestingCode(insts, loadModel);
      String message =
        "KerasZooClassifier - generated testing code\n\n" + toExecute;
      logMessage(message);

      session.instancesToPython(insts, "keras_zoo_test_" + m_modelHash,
        getDebug());

      List<String> outAndErr = session.executeScript(toExecute, getDebug());
      logOutAndErrFromPython(outAndErr);

      List<Object> preds =
        (List<Object>) session.getVariableValueFromPythonAsJson("predictions",
          getDebug());

      if (preds == null) {
        throw new Exception("Was unable to retrieve predictions from python");
      }

      if (preds.size() != insts.numInstances()) {
        throw new Exception(
          "Network did not return as many predictions as there "
            + "are test instances. preds = " + preds.size() + " insts = "
            + insts.numInstances());
      }

      results = new double[insts.numInstances()][];
      int j = 0;
      for (Object o : preds) {
        List<Number> dist = (List<Number>) o;
        double[] newDist = new double[insts.classAttribute().numValues()];
        int k = 0;
        for (int i = 0; i < newDist.length; i++) {
          /* if (m_nominalEmptyClassIndexes[i]) {
            continue;
          } */
          newDist[i] = dist.get(k++).doubleValue();
        }
        try {
          Utils.normalize(newDist);
        } catch (IllegalArgumentException e) {
          newDist = m_classPriors;
          logMessage("WARNING: " + e.getMessage() + ". Predicting using "
            + "class priors");
        }
        results[j++] = newDist;
      }
    } finally {
      PythonSession.releaseSession(this);
    }

    return results;
  }

  /**
   * Write a log message (to the log if one is set, otherwise to std err).
   *
   * @param message the message to write
   */
  protected void logMessage(String message) {
    if (m_log != null) {
      m_log.logMessage(message);
    } else {
      System.err.println(message);
    }
  }

  /**
   * Writes the output and any error messages from python to the log
   *
   * @param outAndErr the std out and err from python
   * @throws Exception if a problem occurs
   */
  protected void logOutAndErrFromPython(List<String> outAndErr)
    throws Exception {
    if (outAndErr.size() > 0) {
      String out = "Output from python:\n\n" + outAndErr.get(0);
      logMessage(out);

      if (outAndErr.size() > 1 && outAndErr.get(1).length() > 0) {
        String err = "Error from python:\n\n" + outAndErr.get(1);
        logMessage(err);
        if (err.contains("Traceback")) {
          throw new WekaException(err);
        }
      }
    }
  }

  /**
   * Set the environment variables to use
   *
   * @param env the environment variables to use
   */
  @Override
  public void setEnvironment(Environment env) {
    m_env = env;
  }

  /**
   * Set the log to use
   *
   * @param log the log to use
   */
  @Override
  public void setLog(Logger log) {
    m_log = log;
  }

  /**
   * Get the log in use
   *
   * @return the log in use
   */
  @Override
  public Logger getLog() {
    return null;
  }

  private String environmentSubstitute(String orig) {
    String result = orig;
    if (result != null && result.length() > 0) {
      try {
        result = m_env.substitute(orig);
      } catch (Exception ex) {
        // ignore
      }
    } else {
      result = "";
    }
    return result;
  }

  /**
   * Returns a textual description of this classifier
   *
   * @return a textual description of this classifier
   */
  public String toString() {
    if (m_zeroR != null) {
      return m_zeroR.toString();
    }

    return "KerasZooClassifier";
  }

  /**
   * Returns true, as we send entire test sets over to python for prediction
   *
   * @return true
   */
  @Override
  public boolean implementsMoreEfficientBatchPrediction() {
    return true;
  }

  public String globalInfo() {
    return "Wrapper classifier for Keras zoo models.";
  }

  /**
   * Returns the appropriate keras package for the supplied zoo model
   *
   * @param zm the zoo model to get the keras package for
   * @return the package that contains the zoo model
   */
  protected String getPackageForZooModel(ZooModel zm) {
    String result = "";
    switch (zm) {
    case Xception:
      result = "keras.applications.xception";
      break;
    case VGG16:
      result = "keras.applications.vgg16";
      break;
    case VGG19:
      result = "keras.applications.vgg19";
      break;
    case ResNet50:
      result = "keras.applications.resnet50";
      break;
    case ResNet101:
    case ResNet152:
      result = "keras_applications.resnet";
      break;
    case InceptionV3:
      result = "keras.applications.inception_v3";
      break;
    case InceptionResNetV2:
      result = "keras.applications.inception_resnet_v2";
      break;
    case MobileNet:
      result = "keras.applications.mobilenet";
      break;
    case MobileNetV2:
      result = "keras.applications.mobilenet_v2";
      break;
    case DenseNet121:
    case DenseNet169:
    case DenseNet201:
      result = "keras.applications.densenet";
      break;
    case NASNetLarge:
    case NASNetMobile:
      result = "keras.applications.nasnet";
      break;
    }
    return result;
  }

  protected static String escapeWindowsBackslashes(String path) {
    return path.replace("\\", "\\\\");
  }

  /**
   * Enum for zoo models
   */
  // keras_applications.resnet for ResNet101, ResNet152 (until keras 2.2.5 comes
  // out)
  // keras.applications should have these and the ResNeXt and ResNetV2 models in
  // 2.2.5
  public static enum ZooModel {
    Xception("299,299"), VGG16("224,224"), VGG19("224,224"),
    ResNet50("224,224"), ResNet101("224,224"), ResNet152("224,224"),
    InceptionV3("299,299"), InceptionResNetV2("299,299"), MobileNet("224,224"),
    MobileNetV2("224,224"), DenseNet121("224,224"), DenseNet169("224,224"),
    DenseNet201("224,224"), NASNetLarge("331,331"), NASNetMobile("224,224");

    /**
     * Constructor
     *
     * @param defaultShape default image width, height
     */
    ZooModel(String defaultShape) {
      m_defaultShape = defaultShape;
    }

    /**
     * Get the default image shape expected by this zoo model
     *
     * @return the default image shape expected by this zoo model
     */
    String getDefaultShape() {
      return m_defaultShape;
    }

    private final String m_defaultShape;
  }

  /**
   * Enum for weights type (imagenet or None).
   */
  public static enum WeightsType {
    None, imagenet;
  }

  /**
   * Enum for optimizer, along with some default settings
   */
  public static enum Optimizer {
    SGD("lr=0.01, momentum=0.0, decay=0.0, nesterov=False"),
    RMSprop("lr=0.001, rho=0.9, epsilon=None, decay=0.0"),
    Adam(
      "lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False"),
    Adamax("lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0"),
    Nadam(
      "lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004"),
    Adagrad("lr=0.01, epsilon=None, decay=0.0"), Adadelta(
      "lr=1.0, rho=0.95, epsilon=None, decay=0.0");

    /**
     * Constructor
     *
     * @param defaultOpts default options to use with this Optimizer
     */
    Optimizer(String defaultOpts) {
      m_defaultOpts = defaultOpts;
    }

    /**
     * Get the default options to use with this Optimizer
     *
     * @return the default options to use with this Optimizer
     */
    String getDefaultOpts() {
      return m_defaultOpts;
    }

    private final String m_defaultOpts;
  }

  /**
   * Inner class that launches a thread for monitoring the CSV log file written
   * by the CSVCallback in python and then relaying new lines to the Weka log
   */
  protected static class LogMonitor {

    protected transient File m_logFile;
    protected transient Logger m_logger;
    protected transient Thread m_monitor;
    protected transient BufferedReader m_reader;
    protected boolean m_stop;

    public LogMonitor(String logFile, Logger logger) {
      m_logFile = new File(logFile);
      m_logger = logger;
    }

    public void start() {
      m_monitor = new Thread() {
        public void run() {
          while (true) {
            if (m_logFile.exists()) {
              try {
                if (m_reader == null) {
                  m_reader = new BufferedReader(new FileReader(m_logFile));
                }

                String line = m_reader.readLine();
                if (line != null) {
                  m_logger.logMessage(line);
                }
              } catch (IOException e) {
                e.printStackTrace();
                m_stop = true;
              }
            }
            if (m_stop) {
              if (m_reader != null) {
                try {
                  m_reader.close();
                } catch (IOException e) {
                  e.printStackTrace();
                }
              }
              break;
            }
            try {
              Thread.sleep(500);
            } catch (InterruptedException e) {
              e.printStackTrace();
            }
          }
        }
      };
      m_monitor.setPriority(Thread.MIN_PRIORITY);
      m_monitor.start();
    }

    public void stop() {
      m_stop = true;
    }
  }

  public static void main(String[] args) {
    try {
      Instances meta = new Instances(new FileReader(args[0]));
      KerasZooClassifier classifier = new KerasZooClassifier();
      args[0] = "";
      meta.setClassIndex(meta.numAttributes() - 1);
      classifier.setOptions(args);
      // classifier.buildClassifier(meta);
      System.out.println(classifier.generateTrainingCode(meta));
      // System.out.println(classifier.generateTestingCode(meta, true));
    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }
}

