package org.allenai.ml.sequences.crf;

import com.gs.collections.api.tuple.Pair;
import lombok.val;
import org.allenai.ml.linalg.Vector;
import org.allenai.ml.objective.BatchObjectiveFn;
import org.allenai.ml.objective.ExampleObjectiveFn;
import org.allenai.ml.optimize.*;
import org.allenai.ml.sequences.StateSpace;
import org.allenai.ml.util.Parallel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.toList;

public class CRFTrainer<S, O, F> {

    private final static Logger logger = LoggerFactory.getLogger(CRFTrainer.class);

    public static class Opts {
        public int numThreads = 1;
        public double sigmaSq = 1.0;
        // Flop a coin with the 1.0/value to decide
        // to keep a given feature. Larger value prunes more
        public int minExpectedFeatureCount = 1;
        // Separate from optimzierOpts since it's specific to the LBFGS choice
        // and optimizerOpts is only for general NewtonMethod options
        public int lbfgsHistorySize = 3;
        public NewtonMethod.Opts optimizerOpts = new NewtonMethod.Opts();
    }

    public final CRFFeatureEncoder<S, O, F> featureEncoder;
    public final CRFPredicateExtractor<O, F> predicateExtractor;
    public final CRFWeightsEncoder<S> weightEncoder;
    private final Opts opts;

    /**
     *
     * @param labeledData Data for features and state-space, can be distinct
     *                    from training data, but probably shouldn't be. Assumes start/stop padded
     */
    public CRFTrainer(
        List<List<Pair<O, S>>> labeledData,
        CRFPredicateExtractor<O, F> predicateExtractor,
        Opts opts)
    {
        this.opts = opts;
        this.predicateExtractor = predicateExtractor;
        logger.info("CRF training with {} threads and {} labeled examples", opts.numThreads, labeledData.size());
        List<List<S>> justLabels = labeledData.stream()
            .map(example -> example.stream().map(e -> e.getTwo()).collect(toList()))
            .collect(toList());
        val firstSent = justLabels.get(0);
        S startState = firstSent.get(0);
        S stopState = firstSent.get(firstSent.size() - 1);
        // Ensure all sequences passed in padded with same start/stop
        ensureStartStopPadded(justLabels, startState, stopState);
        val stateSpace = StateSpace.buildFromSequences(justLabels, startState, stopState);
        logger.info("StateSpace: num states {}, num transitions {}",
            stateSpace.states().size(), stateSpace.transitions().size());
        double featAcceptProb = opts.minExpectedFeatureCount >= 1 ? 1.0/opts.minExpectedFeatureCount : 1.0;
        val featOpts = CRFFeatureEncoder.BuildOpts.builder()
            .numThreads(opts.numThreads)
            .probabilityToAccept(featAcceptProb)
            .build();
        List<List<O>> unlabeledData = labeledData.stream()
            .map(labeledDatum -> labeledDatum.stream().map(Pair::getOne).collect(Collectors.toList()))
            .collect(Collectors.toList());
        this.featureEncoder = CRFFeatureEncoder.build(unlabeledData, predicateExtractor, stateSpace, featOpts);
        logger.info("Number of node predicates: {}, edge predicates: {}",
            featureEncoder.nodeFeatures.size(), featureEncoder.edgeFeatures.size());
        this.weightEncoder = new CRFWeightsEncoder<>(stateSpace,
            featureEncoder.nodeFeatures.size(),
            featureEncoder.edgeFeatures.size());
    }

    private void ensureStartStopPadded(List<List<S>> justLabels, S startState, S stopState) {
        if (!justLabels.stream().allMatch(lst ->
            lst.get(0).equals(startState) &&
                lst.get(lst.size()-1).equals(stopState))) {
            throw new IllegalArgumentException("Not all states padded with start/stop");
        }
    }

    public CRFModel<S, O, F> modelForWeights(Vector weights) {
        return new CRFModel<>(featureEncoder, weightEncoder, weights);
    }

    /**
     * Train a CRFModel from labeled data and a predicateExtractor
     * @param labeledData Assume that each sequence is padded with start/stop states
     * @return Trained CRFModel
     */
    public CRFModel<S, O, F> train(List<List<Pair<O, S>>> labeledData)
    {
        ensureStartStopPadded(labeledData.stream()
                .map(x -> x.stream().map(Pair::getTwo).collect(Collectors.toList()))
                .collect(Collectors.toList()),
            featureEncoder.stateSpace.startState(),
            featureEncoder.stateSpace.stopState());
        ExampleObjectiveFn<CRFIndexedExample> objective = new CRFLogLikelihoodObjective<>(weightEncoder);
        List<CRFIndexedExample> indexedData = labeledData.stream()
            .map(featureEncoder::indexLabeledExample)
            .collect(toList());
        val mrOpts = Parallel.MROpts.withThreads(opts.numThreads);
        BatchObjectiveFn<CRFIndexedExample> objFn =
            new BatchObjectiveFn<>(indexedData, objective, weightEncoder.numParameters(), mrOpts);
        GradientFn regularizer = Regularizer.l2(objFn.dimension(), opts.sigmaSq);
        val cachedObjFn = new CachingGradientFn(opts.lbfgsHistorySize, objFn.add(regularizer));
        val quasiNewton = QuasiNewton.lbfgs(opts.lbfgsHistorySize);
        val optimzier = new NewtonMethod(__ -> quasiNewton, opts.optimizerOpts);
        Vector weights = optimzier.minimize(cachedObjFn).xmin;
        objFn.shutdown();
        return modelForWeights(weights);
    }
}
