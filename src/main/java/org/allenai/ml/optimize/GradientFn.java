package org.allenai.ml.optimize;

import org.allenai.ml.linalg.Vector;
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
        public Result merge(Result other) {
            return new Result(this.fx + other.fx, this.grad.add(other.grad));
        }
    }

    @Override
    Result apply(Vector vec);

    long dimension();

    default GradientFn add(GradientFn other) {
        if (this.dimension() != other.dimension()) {
            throw new IllegalArgumentException("Dimensions don't agree");
        }
        GradientFn parent = this;
        return new GradientFn() {
            @Override
            public Result apply(Vector vec) {
                return parent.apply(vec).merge(other.apply(vec));
            }

            @Override
            public long dimension() {
                return other.dimension();
            }
        };
    }

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
