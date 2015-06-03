package com.allenai.ml.sequences.crf;

import com.allenai.ml.linalg.SparseVector;
import com.allenai.ml.linalg.Vector;
import com.allenai.ml.sequences.StateSpace;
import com.allenai.ml.util.Indexer;
import com.allenai.ml.util.Parallel;
import com.gs.collections.api.map.primitive.ObjectDoubleMap;
import com.gs.collections.api.tuple.Pair;
import lombok.Builder;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

@RequiredArgsConstructor
@Slf4j
public class CRFFeatureEncoder<S, O, F> {

    private final CRFPredicateExtractor<O, F> predicateExtractor;
    public final StateSpace<S> stateSpace;
    public final Indexer<F> nodeFeatures;
    public final Indexer<F> edgeFeatures;

    public CRFIndexedExample indexedExample(List<O> example) {
        List<Vector> nodePreds = indexFeatures(predicateExtractor.nodePredicates(example), nodeFeatures);
        List<Vector> edgePreds = indexFeatures(predicateExtractor.edgePredicates(example), edgeFeatures);
        return new CRFIndexedExample(nodePreds, edgePreds);
    }

    private static <F> List<Vector> indexFeatures(List<ObjectDoubleMap<F>> featVecs, Indexer<F> index) {
        return featVecs
            .stream()
            .map(featVec -> SparseVector.indexed(featVec, index))
            .collect(Collectors.toList());
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

    public static <S, O, F> CRFFeatureEncoder build(
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
                    for (F f : featVec.keysView()) {
                        if (rand.nextDouble() < opts.probabilityToAccept) {
                            set.add(f);
                        }
                    }
                }
            }
        }
        log.info("Indexing features with {} prob to keep and {} threads", opts.probabilityToAccept, opts.numThreads);
        IndexData indexData = Parallel.mapReduce(examples, new IndexWorker(), opts.numThreads);
        return new CRFFeatureEncoder(predicateExtractor,
            stateSpace,
            Indexer.fromStream(indexData.nodeFeatures.stream()),
            Indexer.fromStream(indexData.edgeFeatures.stream()));
    }
}
