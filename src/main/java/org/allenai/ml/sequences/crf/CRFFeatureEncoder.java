package org.allenai.ml.sequences.crf;

import org.allenai.ml.linalg.SparseVector;
import org.allenai.ml.linalg.Vector;
import org.allenai.ml.sequences.StateSpace;
import org.allenai.ml.util.Indexer;
import org.allenai.ml.util.Parallel;
import com.gs.collections.api.map.primitive.ObjectDoubleMap;
import com.gs.collections.api.tuple.Pair;
import lombok.Builder;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.stream.Collectors;

@Slf4j
public class CRFFeatureEncoder<S, O, F extends Comparable<F>> {

    private final CRFPredicateExtractor<O, F> predicateExtractor;
    public final StateSpace<S> stateSpace;
    public final Indexer<F> nodeFeatures;
    public final Indexer<F> edgeFeatures;

    public CRFFeatureEncoder(CRFPredicateExtractor<O, F> predicateExtractor,
                             StateSpace<S> stateSpace,
                             Indexer<F> nodeFeatures,
                             Indexer<F> edgeFeatures) {
        this.predicateExtractor = predicateExtractor;
        this.stateSpace = stateSpace;
        this.nodeFeatures = nodeFeatures;
        this.edgeFeatures = edgeFeatures;
    }

    public CRFIndexedExample indexedExample(List<O> example) {
        List<Vector> nodePreds = indexFeatures(predicateExtractor.nodePredicates(example), nodeFeatures);
        List<Vector> edgePreds = indexFeatures(predicateExtractor.edgePredicates(example), edgeFeatures);
        return new CRFIndexedExample(nodePreds, edgePreds);
    }

    private static <F extends Comparable<F>> List<Vector> indexFeatures(List<ObjectDoubleMap<F>> featVecs, Indexer<F> index) {
        List<Vector> result = new ArrayList<>(featVecs.size());
        for (ObjectDoubleMap<F> featVec : featVecs) {
            result.add(SparseVector.indexed(featVec, index));
        }
        return result;
    }

    public CRFIndexedExample indexLabeledExample(List<Pair<O, S>> labeledExample) {
        List<O> observations = labeledExample.stream()
            .map(Pair::getOne)
            .collect(Collectors.toList());
        List<Vector> nodePreds = indexFeatures(predicateExtractor.nodePredicates(observations), nodeFeatures);
        List<Vector> edgePreds = indexFeatures(predicateExtractor.edgePredicates(observations), edgeFeatures);
        int[] goldLabels = labeledExample.stream()
            .map(Pair::getTwo)
            .mapToInt(stateSpace::stateIndex)
            .toArray();
        if (goldLabels[0] != stateSpace.startStateIndex()) {
            throw new IllegalArgumentException("Must use StateSpace startState to start sequence, instead got " +
                labeledExample.get(0).getTwo());
        }
        if (goldLabels[labeledExample.size()-1] != stateSpace.stopStateIndex()) {
            throw new IllegalArgumentException("Must use StateSpace stopState to end sequence, instead got " +
                labeledExample.get(labeledExample.size()-1).getTwo());
        }
        return new CRFIndexedExample(nodePreds, edgePreds, goldLabels);
    }

    @Builder
    public static class BuildOpts {
        private long randSeed = 0L;
        private int numThreads = 1;
        private double probabilityToAccept = 1.0;
    }

    public static <S, O, F extends Comparable<F>> CRFFeatureEncoder build(
            List<List<O>> examples,
            CRFPredicateExtractor<O, F> predicateExtractor,
            StateSpace<S> stateSpace,
            BuildOpts opts) {

        class IndexData {
            private final Set<F> nodeFeatures = new HashSet<>();
            private final Set<F> edgeFeatures = new HashSet<>();
            private final Random rand = new Random(opts.randSeed);
        }

        @RequiredArgsConstructor
        class IndexWorker implements Parallel.MapReduceDriver<List<O>, IndexData> {

            @Override
            public IndexData newData() {
                return new IndexData();
            }

            @Override
            public void update(IndexData data, List<O> lst) {
                stochasticAddAll(data.rand, data.nodeFeatures, predicateExtractor.nodePredicates(lst));
                stochasticAddAll(data.rand, data.edgeFeatures, predicateExtractor.edgePredicates(lst));
            }

            @Override
            public void merge(IndexData a, IndexData b) {
                a.nodeFeatures.addAll(b.nodeFeatures);
                a.edgeFeatures.addAll(b.edgeFeatures);
            }

            private void stochasticAddAll(Random rand, Set<F> set, List<ObjectDoubleMap<F>> featVecs) {
                for (ObjectDoubleMap<F> featVec : featVecs) {
                    Set<F> fSet = featVec.keySet();
                    List<F> fList = new ArrayList<>(fSet);
                    Collections.sort(fList);
                    for (F f : featVec.keysView()) {
                        if (rand.nextDouble() < opts.probabilityToAccept) {
                            set.add(f);
                        }
                    }
                }
            }
        }
        log.info("Indexing features with {} prob to keep and {} threads", opts.probabilityToAccept, opts.numThreads);
        Parallel.MROpts mrOpts = Parallel.MROpts.withIdAndThreads("mr-feature-index", opts.numThreads);
        IndexData indexData = Parallel.mapReduce(examples, new IndexWorker(), mrOpts);
        Parallel.shutdownExecutor(mrOpts.executorService, Long.MAX_VALUE);
        return new CRFFeatureEncoder(predicateExtractor,
            stateSpace,
            Indexer.fromStream(indexData.nodeFeatures.stream()),
            Indexer.fromStream(indexData.edgeFeatures.stream()));
    }
}
