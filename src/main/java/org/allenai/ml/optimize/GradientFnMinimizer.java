package org.allenai.ml.optimize;

import org.allenai.ml.linalg.DenseVector;
import org.allenai.ml.linalg.Vector;
import lombok.RequiredArgsConstructor;

@FunctionalInterface
/**
 * Core interface to be used by users of this package.
 */
public interface GradientFnMinimizer {

    /**
     * Result struct for the output of minimization
     */
    @RequiredArgsConstructor(staticName = "of")
    class Result {
        /**
         * The minimal value of f(xmin)
         */
        public final double fxmin;
        /**
         * The argument vector yielding the minimum value
         */
        public final Vector xmin;
    }

    default Result minimize(GradientFn gradFn) {
        return this.minimize(gradFn, DenseVector.of(gradFn.dimension()));
    }

    /**
     * Key interface function
     * @param gradFn Function to be minimized
     * @param initGuess Initial guess. For a non-approximate minimizer, this only affects convergence speed.
     * @return
     */
    Result minimize(GradientFn gradFn, Vector initGuess);
}
