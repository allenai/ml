package com.allenai.ml.optimize;

import com.allenai.ml.linalg.Vector;
import lombok.RequiredArgsConstructor;

/**
 * Interface for solving problems of the form `min_alpha f(x + stepLength * dir)` for a direction `dir`
 */
@FunctionalInterface
public interface LineMinimizer {
    @RequiredArgsConstructor(staticName = "of")
    class Result {
        /**
         * The distance along `dir` which yields the minimal value
         */
        public final double stepLength;
        /**
         * The function value at `gradFn( x.add(stepLength, dir) )`
         */
        public final double fxmin;
    }

    Result minimize(GradientFn gradFn, Vector x, Vector dir);
}