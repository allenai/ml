package com.allenai.ml.objective;

import com.allenai.ml.linalg.DenseVector;
import com.allenai.ml.linalg.Vector;
import com.allenai.ml.optimize.GradientFn;
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
    private final int numThreads = Runtime.getRuntime().availableProcessors();
    private final ExecutorService executorService;

    private class Worker implements Runnable {
        List<T> subList;
        double objectiveValue = 0.0;
        Vector objectiveGradient = DenseVector.of(dimension);
        Vector weights ;

        Worker(List<T> subList, Vector weights) {
            this.subList = subList;
            this.weights = weights;
        }

        @Override
        public void run() {
            for (T t : subList) {
                objectiveValue += exampleObjectiveFn.evaluate(t, weights, objectiveGradient);
            }
        }
    }

    public BatchObjectiveFn(List<T> data, ExampleObjectiveFn<T> exampleObjectiveFn, long dimension, int numThreads) {
        // copy to a GS collection
        this.data = new ArrayList<>(data);
        this.exampleObjectiveFn = exampleObjectiveFn;
        this.dimension = dimension;
        this.executorService = Executors.newFixedThreadPool(numThreads);
    }

    @Override
    @SneakyThrows({InterruptedException.class, ExecutionException.class})
    public Result apply(Vector weights) {
        // defensive copy so all workers can read this instance
        weights = weights.copy();
        List<Worker> workers = new ArrayList<>();
        int chunkSize = data.size() / numThreads;
        for (int idx = 0; idx < numThreads; idx++) {
            int start = idx * chunkSize;
            int stop = idx + 1 < numThreads ? (idx+1) * chunkSize : data.size();
            workers.add(new Worker(data.subList(start, stop), weights));
        }
        List<Future<?>> futureStream = workers.stream()
            .map(executorService::submit)
            .collect(Collectors.toList());
        for (Future<?> f : futureStream) {
            f.get();
        }
        double objectiveValue = workers.stream().mapToDouble(w -> w.objectiveValue).sum();
        Vector objectiveGradient = DenseVector.of(dimension);
        workers.stream()
            .map(w -> w.objectiveGradient)
            .forEach(g -> objectiveGradient.addInPlace(1.0, g));
        return Result.of(objectiveValue, objectiveGradient);
    }

    @Override
    public long dimension() {
        return dimension;
    }
}
