package org.allenai.ml.sequences.crf.conll;

import org.allenai.ml.linalg.Vector;
import org.allenai.ml.objective.BatchObjectiveFn;
import org.allenai.ml.optimize.*;
import org.allenai.ml.sequences.StateSpace;
import org.allenai.ml.sequences.TokenAccuracy;
import com.gs.collections.api.tuple.Pair;
import com.gs.collections.impl.tuple.Tuples;
import lombok.SneakyThrows;
import lombok.val;
import org.allenai.ml.sequences.crf.*;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.util.List;

import static org.allenai.ml.util.IOUtils.linesFromPath;
import static java.util.stream.Collectors.toList;

public class Trainer {

    private final static Logger logger = LoggerFactory.getLogger(Trainer.class);

    public static class Opts {
        @Option(name = "-featureTemplates", usage = "FeatureTemplate template pattern file", required = true)
        public String templateFile;

        @Option(name = "-trainData", usage = "Path to training data", required = true)
        public String trainPath;

        @Option(name = "-sigmaSq", usage = "L2 regularization to use")
        public double sigmaSquared = 1.0;

        @Option(name = "-numThreads", usage = "number of threads to train with")
        public int numThreads = 1;

        @Option(name = "-modelSave", usage = "where to write model", required = true)
        public String modelPath;

        @Option(name = "-featureKeepProb", usage = "probability of keeping a feature predicate")
        public double featureKeepProb = 1.0;

        @Option(name = "-maxTrainIters", usage = "max number of train iterations")
        public int maxIterations = Integer.MAX_VALUE;

        @Option(name = "-lbfgsHistorySize", usage = "history size for LBFGS")
        public int lbfgsHistorySize = 3;

    }

    @SneakyThrows
    public static void trainAndSaveModel(Opts opts) {
        List<String> templateLines = linesFromPath(opts.templateFile).collect(toList());
        val predExtractor = ConllFormat.predicatesFromTemplate(templateLines.stream());
        List<List<ConllFormat.Row>> labeledData = ConllFormat.readData(linesFromPath(opts.trainPath), true);
        logger.info("CRF training with {} threads and {} labeled examples", opts.numThreads, labeledData.size());
        List<List<String>> justLabels = labeledData.stream()
            .map(example -> example.stream().map(e -> e.getLabel().get()).collect(toList()))
            .collect(toList());
        val stateSpace = StateSpace.buildFromSequences(justLabels, ConllFormat.startState, ConllFormat.stopState);
        logger.info("StateSpace: num states {}, num transitions {}",
            stateSpace.states().size(), stateSpace.transitions().size());
        val featOpts = CRFFeatureEncoder.BuildOpts.builder()
            .numThreads(opts.numThreads)
            .probabilityToAccept(opts.featureKeepProb)
            .build();
        val featureEncoder = CRFFeatureEncoder.build(labeledData, predExtractor, stateSpace, featOpts);
        logger.info("Number of node predicates: {}, edge predicates: {}",
            featureEncoder.nodeFeatures.size(), featureEncoder.edgeFeatures.size());
        val weightEncoder = new CRFWeightsEncoder<String>(stateSpace,
            featureEncoder.nodeFeatures.size(), featureEncoder.edgeFeatures.size());
        Vector weights = trainWeights(opts, labeledData, weightEncoder, featureEncoder);
        val dos = new DataOutputStream(new FileOutputStream(opts.modelPath));
        logger.info("Writing model to {}", opts.modelPath);
        ConllFormat.saveModel(dos, templateLines, featureEncoder, weights);
    }

    private static Vector trainWeights(Opts opts,
                                       List<List<ConllFormat.Row>> labeledData,
                                       CRFWeightsEncoder<String> weightEncoder,
                                       CRFFeatureEncoder featureEncoder) {
        val objective = new CRFLogLikelihoodObjective<>(weightEncoder);
        List<CRFIndexedExample> indexedData = labeledData.stream()
            .map(rows -> {
                List<Pair<ConllFormat.Row, String>> pairs = rows.stream()
                    .map(r -> Tuples.pair(r, r.getLabel().get()))
                    .collect(toList());
                return featureEncoder.indexLabeledExample(pairs);
            })
            .collect(toList());
        val objFn = new BatchObjectiveFn<>(indexedData, objective, weightEncoder.numParameters(), opts.numThreads);
        GradientFn regularizer = Regularizer.l2(objFn.dimension(), opts.sigmaSquared);
        val cachedObjFn = new CachingGradientFn(opts.lbfgsHistorySize, objFn.add(regularizer));
        val quasiNewton = QuasiNewton.lbfgs(opts.lbfgsHistorySize);
        val optimizerOpts = new NewtonMethod.Opts();
        optimizerOpts.maxIters = opts.maxIterations;
        optimizerOpts.iterCallback = weights -> {
            CRFModel<String, ConllFormat.Row, String> crfModel = new CRFModel<>(featureEncoder,weightEncoder,weights);
            long start = System.currentTimeMillis();
            List<List<Pair<String, ConllFormat.Row>>> evalData = labeledData.stream()
                .map(x -> x.stream().map(ConllFormat.Row::asLabeledPair).collect(toList()))
                .collect(toList());
            double acc = TokenAccuracy.compute(crfModel, evalData, opts.numThreads);
            long stop = System.currentTimeMillis();
            logger.info("Accuracy: {} (took {} ms)", acc, stop-start);
        };
        val optimzier = new NewtonMethod(__ -> quasiNewton, optimizerOpts);
        return optimzier.minimize(cachedObjFn).xmin;
    }

    @SneakyThrows
    public static void main(String[] args)  {
        val opts = new Opts();
        val cmdLineParser = new CmdLineParser(opts);
        try {
            cmdLineParser.parseArgument(args);
        } catch (CmdLineException e) {
            cmdLineParser.printUsage(System.err);
            System.exit(2);
        }
        trainAndSaveModel(opts);
    }
}
