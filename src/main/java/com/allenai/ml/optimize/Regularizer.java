package com.allenai.ml.optimize;

import com.allenai.ml.linalg.Vector;

public class Regularizer {

    public static GradientFn l2(long dimension, double sigmaSq) {
        return new GradientFn() {
            @Override
            public Result apply(Vector vec) {
                Vector grad = vec.scale(2.0/sigmaSq);
                double val = vec.nonZeroEntries().mapToDouble(e -> e.value * e.value / sigmaSq).sum();
                return Result.of(val, grad);
            }

            @Override
            public long dimension() {
                return dimension;
            }
        };
    }
}
