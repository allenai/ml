package com.allenai.ml.optimize;

import com.allenai.ml.linalg.Vector;
import lombok.RequiredArgsConstructor;

import java.util.function.Function;

/**
 * Abstraction for a function from a real-valued Vector to a pair of `[f(x), grad f(x)]`. This is the type
 * of function which can be easily minimized via numerical optimization
 */
public interface GradientFn extends Function<Vector, GradientFn.Result> {

    @RequiredArgsConstructor(staticName = "of")
    class Result {
        public final double fx;
        public final Vector grad;
    }

    @Override
    Result apply(Vector vec);

    long dimension();

    static GradientFn from(long dimension, Function<Vector, Result> fn) {
        return new GradientFn() {
            @Override
            public Result apply(Vector vec) {
                if (vec.dimension() != this.dimension()) {
                    throw new IllegalArgumentException("Argument doesn't match dimesnion(): " +
                        vec.dimension() + " != " + this.dimension());
                }
                return fn.apply(vec);
            }

            @Override
            public long dimension() {
                return dimension;
            }
        };
    }
}
