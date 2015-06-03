package org.allenai.ml.util;

import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public class Parallel {

    private Parallel() {
        // intentional no-op
    }

    public interface MapReduceDriver<T, D>  {
        D newData();
        void update(D data, T elem);
        void merge(D a, D b);
    }

    public static <T, D> D mapReduce(List<T> data, MapReduceDriver<T, D> driver, int numThreads) {
        return mapReduce(data, driver, numThreads, Double.MAX_VALUE);
    }

    @SneakyThrows({InterruptedException.class})
    /**
     * Perform an in-memory version of MapReduce targeted at accumulating sufficient statistics
     * from a data-set.
     */
    public static <T, D> D mapReduce(List<T> data, MapReduceDriver<T, D> driver, int numThreads, double maxSecs) {
        ExecutorService executorService = Executors.newFixedThreadPool(numThreads);
        @RequiredArgsConstructor
        class Worker implements Runnable {
            private final List<T> dataSlice;
            private final D data = driver.newData();
            public void run() {
                for (T t : dataSlice) {
                    driver.update(data, t);
                }
            }
        }
        List<Worker> workers = Functional.partition(data, numThreads).stream()
            .map(Worker::new)
            .collect(Collectors.toList());
        workers.forEach(executorService::submit);
        executorService.shutdown();
        executorService.awaitTermination((long)(maxSecs * 1000), TimeUnit.MILLISECONDS);
        D finalData = driver.newData();
        for (Worker worker : workers) {
            driver.merge(finalData, worker.data);
        }
        return finalData;
    }
}
