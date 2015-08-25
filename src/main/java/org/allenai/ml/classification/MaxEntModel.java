package org.allenai.ml.classification;

import com.gs.collections.api.map.primitive.ObjectDoubleMap;
import com.gs.collections.api.tuple.Pair;
import com.gs.collections.api.tuple.primitive.IntObjectPair;
import com.gs.collections.impl.tuple.Tuples;
import com.gs.collections.impl.tuple.primitive.PrimitiveTuples;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.val;
import org.allenai.ml.linalg.DenseVector;
import org.allenai.ml.linalg.SparseVector;
import org.allenai.ml.linalg.Vector;
import org.allenai.ml.objective.BatchObjectiveFn;
import org.allenai.ml.optimize.*;
import org.allenai.ml.util.IOUtils;
import org.allenai.ml.util.Indexer;
import org.allenai.ml.util.Parallel;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@RequiredArgsConstructor
public class MaxEntModel<L, D, F> implements ProbabilisticClassifier<D, L> {
    private final Indexer<F> featureIndexer;
    private final Indexer<L> classIndexer;
    private final Vector weights;
    private final FeatureExtractor<D, F> featureExtractor;

    @Override
    public ObjectDoubleMap<L> probabilities(D datum) {
        ObjectDoubleMap<F> featureMap = featureExtractor.features(datum);
        val featVec = SparseVector.indexed(featureMap, featureIndexer);
        double[] classProbs = MaxEntObjective.classProbs(featVec, weights, classIndexer.size());
        return classIndexer.toMap(DenseVector.of(classProbs));
    }

    @SneakyThrows
    public static <D> MaxEntModel<String, D, String> load(DataInputStream dis,
                                                          FeatureExtractor<D, String> featureExtractor) {
        val featureIndexer = Indexer.load(dis);
        val classIndexer = Indexer.load(dis);
        Vector weights = DenseVector.of(IOUtils.loadDoubles(dis));
        return new MaxEntModel<>(featureIndexer, classIndexer, weights, featureExtractor);
    }

    @SneakyThrows
    public void save(DataOutputStream dos) {
        featureIndexer.save(dos);
        classIndexer.save(dos);
        IOUtils.saveDoubles(dos, weights.toDoubles());
    }

    public static class TrainOpts {
        public int minExpectedFeatureCount= 0;
        public int numThreads = 1;
        public double sigmaSq;
        public long randSeed = 0L;
        public NewtonMethod.Opts optimizerOpts = null;
    }


    public static <D> MaxEntModel<String, D, String> train(List<Pair<D, String>> labeledData,
                                                       FeatureExtractor<D, String> featureExtractor,
                                                       TrainOpts opts) {
        Random rand = new Random(opts.randSeed);
        double probAccept = opts.minExpectedFeatureCount > 0 ? 1.0/opts.minExpectedFeatureCount : 1.0;
        Stream<String> allFeats = labeledData.stream()
            .flatMap(pair ->  featureExtractor.features(pair.getOne()).keySet().stream())
            .filter(f -> rand.nextDouble() < probAccept);
        Indexer<String> featIndexer = Indexer.fromStream(allFeats);
        Indexer<String> classIndexer = Indexer.fromStream(labeledData.stream().map(Pair::getTwo));
        MaxEntObjective maxent = new MaxEntObjective(classIndexer.size());
        long dimension = featIndexer.size() * classIndexer.size();
        List<IntObjectPair<Vector>> indexedLabeledData = labeledData.stream()
            .map(pair -> {
                ObjectDoubleMap<String> featureMap = featureExtractor.features(pair.getOne());
                Vector featVec = SparseVector.indexed(featureMap, featIndexer);
                int classIdx = classIndexer.indexOf(pair.getTwo());
                return PrimitiveTuples.pair(classIdx, featVec);
            })
            .collect(Collectors.toList());
        Parallel.MROpts mrOpts = Parallel.MROpts.withIdAndThreads("mr-max-ent-train", opts.numThreads);
        GradientFn objFn = new BatchObjectiveFn(indexedLabeledData, maxent, dimension, mrOpts);
        GradientFn regularizer = Regularizer.l2(objFn.dimension(), opts.sigmaSq);
        val cachedObjFn = new CachingGradientFn(3, objFn.add(regularizer));
        val quasiNewton = QuasiNewton.lbfgs(3);
        val optimizerOpts = opts.optimizerOpts != null ? opts.optimizerOpts : new NewtonMethod.Opts();
        val optimzier = new NewtonMethod(__ -> quasiNewton, optimizerOpts);
        Vector weights = optimzier.minimize(cachedObjFn).xmin;
        Parallel.shutdownExecutor(mrOpts.executorService, Long.MAX_VALUE);
        return new MaxEntModel<>(featIndexer, classIndexer, weights, featureExtractor);
    }
}
