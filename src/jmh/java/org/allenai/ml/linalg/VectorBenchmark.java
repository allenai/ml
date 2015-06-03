package org.allenai.ml.linalg;

import com.gs.collections.api.map.primitive.LongDoubleMap;
import com.gs.collections.api.map.primitive.MutableLongDoubleMap;
import com.gs.collections.api.tuple.primitive.LongDoublePair;
import com.gs.collections.impl.map.mutable.primitive.LongDoubleHashMap;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toList;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Threads(1)
@Warmup(iterations = 3, time = 100, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
public class VectorBenchmark {

    private final static int NUM_DIMENSIONS = 10000;
    private final static int NUM_ACTIVE_VECTORS = 100;
    private final static int NUM_FEATURE_VECTORS = 10;
    private final double[] weights = new double[NUM_DIMENSIONS];
    private Vector weightVector;
    private List<LongDoubleMap> gsFeatureVectors = new ArrayList<>();
    private List<Vector> featureVectors = new ArrayList<>();

    @Benchmark
    public void sparseDenseDotProductThroughVector() throws Exception {
        double sum = featureVectors.stream().mapToDouble(weightVector::dotProduct).sum();
    }

    @Benchmark
    public void sparseDenseDotProductDirect() throws Exception {
        double sum = 0.0;
        for (LongDoubleMap featureVector : gsFeatureVectors) {
            double dotProduct = 0.0;
            for (Iterator<LongDoublePair> it = featureVector.keyValuesView().iterator(); it.hasNext(); ) {
                LongDoublePair pair = it.next();
                long idx = pair.getOne();
                double val = pair.getTwo();
                dotProduct += val * weights[(int)idx];
            }
            sum += dotProduct;
        }
    }

    @Setup
    public void up() {
        Random rand = new Random(0);
        for (int i = 0; i < NUM_DIMENSIONS; i++) {
            // in range [-1,1] uniformly
            weights[i] = 2.0 * (rand.nextDouble() - 0.5);
        }
        weightVector = DenseVector.of(weights);
        gsFeatureVectors = IntStream.range(0, NUM_FEATURE_VECTORS)
            .mapToObj(ignored -> {
                MutableLongDoubleMap vec = new LongDoubleHashMap(NUM_FEATURE_VECTORS);
                for (int j = 0; j < NUM_ACTIVE_VECTORS; j++) {
                    int dimensionIdx = rand.nextInt(NUM_DIMENSIONS);
                    vec.put(dimensionIdx, rand.nextDouble());
                }
                return vec;
            })
            .collect(toList());
        featureVectors = gsFeatureVectors.stream()
            .map(fv -> SparseVector.make(fv, NUM_DIMENSIONS))
            .collect(toList());
    }

    public static void main(String[] args) throws Exception {
        Options opts = new OptionsBuilder()
            .include(".*" + VectorBenchmark.class.getSimpleName() + ".*")
            .build();
        new Runner(opts).run();
    }
}
