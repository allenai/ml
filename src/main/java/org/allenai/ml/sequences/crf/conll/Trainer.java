package org.allenai.ml.sequences.crf.conll;

import org.allenai.ml.eval.FMeasure;
import org.allenai.ml.eval.TrainCriterionEval;
import org.allenai.ml.linalg.Vector;
import org.allenai.ml.objective.BatchObjectiveFn;
import org.allenai.ml.optimize.*;
import org.allenai.ml.sequences.StateSpace;
import org.allenai.ml.sequences.Evaluation;
import com.gs.collections.api.tuple.Pair;
import com.gs.collections.impl.tuple.Tuples;
import lombok.SneakyThrows;
import lombok.val;
import org.allenai.ml.sequences.crf.*;
import org.allenai.ml.util.Parallel;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;

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

        @Option(name = "-testSplitRatio", usage = "Data to hold for eval ever iter")
        public double testSplitRatio = 0.2;

        @Option(name= "-maxNumDipIters", usage = "How many iterations after test eval drop to continue training")
        public int maxNumDipIters = 3;
    }

    private static <T> Pair<List<T>, List<T>> splitData(List<T> original, double splitForSecond) {
        List<T> first = new ArrayList<>();
        List<T> second = new ArrayList<>();
        if (splitForSecond > 0.0) {
            Collections.shuffle(original, new Random(0L));
            int numFirst = (int) ((1.0-splitForSecond) * original.size());
            first.addAll(original.subList(0, numFirst));
            second.addAll(original.subList(numFirst, original.size()));
        } else {
            first.addAll(original);
            // second stays empty
        }
        return Tuples.pair(first, second);
    }

    @SneakyThrows
    public static void trainAndSaveModel(Opts opts) {
        // Load labeled data
        List<String> templateLines = linesFromPath(opts.templateFile).collect(toList());
        logger.info("Loading train data from {}", opts.trainPath);
        val predExtractor = ConllFormat.predicatesFromTemplate(templateLines.stream());
        List<List<Pair<ConllFormat.Row, String>>> labeledData = ConllFormat
            .readData(linesFromPath(opts.trainPath), true)
            .stream()
            .map(x -> x.stream().map(y -> y.asLabeledPair().swap()).collect(Collectors.toList()))
            .collect(Collectors.toList());

        // Split train/test data
        logger.info("CRF training with {} threads and {} labeled examples", opts.numThreads, labeledData.size());
        val trainTestPair = splitData(labeledData, opts.testSplitRatio);
        List<List<Pair<ConllFormat.Row, String>>> trainLabeledData = trainTestPair.getOne();
        List<List<Pair<ConllFormat.Row, String>>> testLabeledData = trainTestPair.getTwo();

        // Set up Train options
        CRFTrainer.Opts<String, ConllFormat.Row, String> trainOpts =
            new CRFTrainer.Opts<>();
        trainOpts.sigmaSq = opts.sigmaSquared;
        trainOpts.lbfgsHistorySize = opts.lbfgsHistorySize;
        trainOpts.optimizerOpts.maxIters = opts.maxIterations;
        trainOpts.minExpectedFeatureCount = (int) (1.0/opts.featureKeepProb);
        trainOpts.numThreads = opts.numThreads;

        // Trainer
        CRFTrainer<String, ConllFormat.Row, String> trainer =
            new CRFTrainer<>(trainLabeledData, predExtractor, trainOpts);

        // Setup iteration callback, weird trick here where you require
        // the trainer to make a model for each iteration but then need
        // to modify the iteration-callback to use it
        Parallel.MROpts evalMrOpts = Parallel.MROpts.withIdAndThreads("mr-crf-train-eval", opts.numThreads);
        List<List<Pair<String, ConllFormat.Row>>> trainEvalData = trainLabeledData.stream()
            .map(x -> x.stream().map(Pair::swap).collect(toList()))
            .collect(toList());
        List<List<Pair<String, ConllFormat.Row>>> testEvalData = testLabeledData.stream()
            .map(x -> x.stream().map(Pair::swap).collect(toList()))
            .collect(toList());
        ToDoubleFunction<CRFModel<String, ConllFormat.Row, String>> trainEvalFn = (model) -> {
            Evaluation<String> eval = Evaluation.compute(model, trainEvalData, evalMrOpts);
            return eval.tokenAccuracy.accuracy();
        };
        ToDoubleFunction<CRFModel<String, ConllFormat.Row, String>> testEvalFn = (model) -> {
            Evaluation<String> eval = Evaluation.compute(model, testEvalData, evalMrOpts);
            return eval.tokenAccuracy.accuracy();
        };
        TrainCriterionEval<CRFModel<String, ConllFormat.Row, String>> criterion = new TrainCriterionEval<>(testEvalFn);
        criterion.maxNumDipIters = opts.maxNumDipIters;
        trainOpts.iterCallback = (CRFModel<String, ConllFormat.Row, String> crfModel) -> {
            logger.info("Train Accuracy: {}", trainEvalFn.applyAsDouble(crfModel));
            return criterion.test(crfModel);
        };
        trainer.train(trainLabeledData);
        // Criterion may have better model than last iteration
        CRFModel<String, ConllFormat.Row, String> crfModel = criterion.getBestModel();
            Parallel.shutdownExecutor(evalMrOpts.executorService, Long.MAX_VALUE);
        Vector weights = crfModel.weights();
        val dos = new DataOutputStream(new FileOutputStream(opts.modelPath));
        logger.info("Writing model to {}", opts.modelPath);
        ConllFormat.saveModel(dos, templateLines, crfModel.featureEncoder, weights);
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
