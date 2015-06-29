package org.allenai.ml.sequences.crf.conll;

import org.allenai.ml.sequences.Evaluation;
import org.allenai.ml.util.IOUtils;
import com.gs.collections.api.tuple.Pair;
import com.gs.collections.impl.tuple.Tuples;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.allenai.ml.util.Parallel;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.util.List;

import static java.util.stream.Collectors.toList;

@Slf4j
public class Evaluator {
    public static class Opts {
        @Option(name = "-model", usage = "where to read model", required = true)
        public String modelPath;

        @Option(name = "-data", usage = "where to read data", required = true)
        public String dataPath;
    }

    @SneakyThrows
    /**
     * Returns pair of (token-accuracy, inference-time-per-instance-as-millis)
     */
    public static Pair<Double, Double> evaluateModel(Opts opts) {
        val dis = new DataInputStream(new FileInputStream(opts.modelPath));
        val crf = ConllFormat.loadModel(dis);
        val data = ConllFormat.readData(IOUtils.linesFromPath(opts.dataPath), true);
        long start = System.currentTimeMillis();
        List<List<Pair<String, ConllFormat.Row>>> evalData = data.stream()
            .map(x -> x.stream().map(ConllFormat.Row::asLabeledPair).collect(toList()))
            .collect(toList());
        double acc = Evaluation.compute(crf, evalData, Parallel.MROpts.withThreads(1)).tokenAccuracy.accuracy();
        long stop = System.currentTimeMillis();
        return Tuples.pair(acc, (double)(stop-start)/data.size());
    }

    public static void main(String[] args) {
        val opts = new Opts();
        val cmdLineParser = new CmdLineParser(opts);
        try {
            cmdLineParser.parseArgument(args);
        } catch (CmdLineException e) {
            cmdLineParser.printUsage(System.err);
            System.exit(2);
        }
        val accTimePair = evaluateModel(opts);
        log.info("Accuracy: {}", accTimePair.getOne());
        log.info("Inference avg ms per example: {}", accTimePair.getTwo());

    }
}
