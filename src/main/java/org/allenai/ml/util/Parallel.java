package org.allenai.ml.util;

import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.val;

import java.util.List;
import java.util.concurrent.*;

import static java.util.stream.Collectors.toList;

public class Parallel {

    private Parallel() {
        // intentional no-op
    }

    public interface MapReduceDriver<T, D>  {
        D newData();
        void update(D data, T elem);
        void merge(D a, D b);
    }

    public static <T, D> D mapReduce(List<T> data, MapReduceDriver<T, D> driver) {
        return mapReduce(data, driver, new MROpts());
    }

    @RequiredArgsConstructor
    public static class MROpts {
        public int numWorkers = Runtime.getRuntime().availableProcessors();
        public ExecutorService executorService;
        public double maxSecs = 1000000.0;

        public static MROpts withThreads(int numThreads) {
            val opts = new MROpts();
            opts.numWorkers = numThreads;
            return opts;
        }
    }

    @SneakyThrows()
    /**
     * Perform an in-memory version of MapReduce targeted at accumulating sufficient statistics
     * from a data-set.
     */
    public static <T, D> D mapReduce(List<T> data, MapReduceDriver<T, D> driver, MROpts mrOpts) {
        ExecutorService executorService = mrOpts.executorService != null ?
            mrOpts.executorService :
            Executors.newFixedThreadPool(mrOpts.numWorkers);
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
        List<Worker> workers = Functional.partition(data, mrOpts.numWorkers).stream()
            .map(Worker::new)
            .collect(toList());
        List<Future<?>> futures =  workers.stream().map(executorService::submit).collect(toList());
        for (Future<?> future : futures) {
            future.get((long) mrOpts.maxSecs * 1000, TimeUnit.MILLISECONDS);
        }
        D finalData = driver.newData();
        for (Worker worker : workers) {
            driver.merge(finalData, worker.data);
        }
        return finalData;
    }
}
