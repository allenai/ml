package com.allenai.ml.objective;

import com.allenai.ml.linalg.DenseVector;
import com.allenai.ml.linalg.Vector;
import com.allenai.ml.optimize.GradientFn;
import com.allenai.ml.util.Functional;
import com.allenai.ml.util.Parallel;
import lombok.SneakyThrows;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

/**
 * Objective function calculation over an entire dataset. Natively supports multiple threads of execution. You can
 * wrap this function with a regularization `GradientFn` if desired.
 */
public class BatchObjectiveFn<T> implements GradientFn {
    private final List<T> data;
    private final long dimension;
    private final ExampleObjectiveFn<T> exampleObjectiveFn;
    private final int numThreads;

    public BatchObjectiveFn(List<T> data, ExampleObjectiveFn<T> exampleObjectiveFn, long dimension, int numThreads) {
        // copy to a GS collection
        this.data = new ArrayList<>(data);
        this.exampleObjectiveFn = exampleObjectiveFn;
        this.dimension = dimension;
        this.numThreads = numThreads;
    }


    @Override
    public Result apply(Vector weightsOriginal) {
        // defensive copy so all workers can read this instance
        final Vector weights = weightsOriginal.copy();
        class ObjectiveStats {
            double value;
            Vector gradient = DenseVector.of(weights.dimension());
        }
        Parallel.MapReduceDriver<T, ObjectiveStats> driver = new Parallel.MapReduceDriver<T, ObjectiveStats>() {
            @Override
            public ObjectiveStats newData() {
                return new ObjectiveStats();
            }

            @Override
            public void update(ObjectiveStats data, T elem) {
                data.value += exampleObjectiveFn.evaluate(elem, weights, data.gradient);
            }

            @Override
            public void merge(ObjectiveStats a, ObjectiveStats b) {
                a.value += b.value;
                a.gradient.addInPlace(1.0, b.gradient);
            }
        };
        // The optimization code is in terms of 'minimizing' so we want to
        // return the negative objective value and gradient
        ObjectiveStats stats = Parallel.mapReduce(data, driver, numThreads);
        return Result.of(-stats.value, stats.gradient.scale(-1.0));
    }

    @Override
    public long dimension() {
        return dimension;
    }
}
