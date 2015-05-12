package com.allenai.ml.optimize;

import com.allenai.ml.linalg.Vector;
import com.gs.collections.api.tuple.Pair;
import com.gs.collections.impl.tuple.Tuples;

import java.util.ArrayList;
import java.util.List;

/**
 * Interface for how to approximate `H^{-1} dir`, where `H^{-1}` is an approximation for the inverse hessian and
 * dir is the current search direction. If we use `H^{-1}` as the identity matrix, you get standard gradient descent.
 * If you remember some history, you get LBFGS.
 */
@FunctionalInterface
public interface QuasiNewton {

    /**
     * @return Approximation of inverse hessian matrix times `dir`
     */
    Vector implictMultiply(Vector dir);

    /**
     * Update the approximation
     * @param xDelta The delta from the last guess
     * @param gradDelta The delta from the function gradient at last guess
     */
    default void update(Vector xDelta, Vector gradDelta) {
        // intentional no-op
    }

    /**
     * Approximate inverse hessian with the identity
     */
    static QuasiNewton gradientDescent() {
        return dir -> dir;
    }

    /**
     * LBFGS inverse hessian approximation
     */
    static QuasiNewton lbfgs(final int maxHistorySize) {
        return new QuasiNewton() {
            // Store (xDelta, gradDelta) pairs
            private final List<Pair<Vector, Vector>> history = new ArrayList<>();

            private double initialScale() {
                if (history.isEmpty()) {
                    return 1.0;
                }
                Vector lastInputDiff = history.get(0).getOne();
                Vector lastGradDiff = history.get(0).getTwo();
                double numer = lastGradDiff.dotProduct(lastInputDiff);
                double denom = lastGradDiff.l2NormSquared();
                assert denom > 0.0 : "Shouldn't have gotten a 0 diff between successive gradients";
                return numer / denom;
            }

            @Override
            public Vector implictMultiply(Vector dir) {
                double[] rho = new double[history.size()];
                double[] alpha = new double[history.size()];
                Vector right = dir.copy();
                for (int i = history.size() - 1; i >= 0; i--) {
                    Vector inputDifference = history.get(i).getOne();
                    Vector derivativeDifference = history.get(i).getTwo();
                    rho[i] = inputDifference.dotProduct(derivativeDifference);
                    assert rho[i]!= 0.0 : "Input diff and derivative diff can't be orthogonal by construction";
                    alpha[i] = inputDifference.dotProduct(right) / rho[i];
                    right = right.add(-alpha[i], derivativeDifference);
                }
                right.scaleInPlace(initialScale());
                Vector left = right;
                for (int i = 0; i < history.size(); i++) {
                    Vector inputDifference = history.get(i).getOne();
                    Vector derivativeDifference = history.get(i).getTwo();
                    double beta = derivativeDifference.dotProduct(left) / rho[i];
                    left = left.add(alpha[i] - beta, inputDifference);
                }
                return left;
            }


            private final static double EPS = 1.0e-200;

            @Override
            public void update(Vector xDelta, Vector gradDelta) {
                if (xDelta.l2NormSquared() < EPS || gradDelta.l2NormSquared() < EPS) {
                    throw new IllegalArgumentException("Too small a diff between successive input or gradient." +
                        "Should have already converged already");
                }
                if (gradDelta.l2NormSquared() < EPS) {
                    throw new IllegalArgumentException("Can't have this small input delta: " + gradDelta.l2NormSquared());
                }
                this.history.add(0, Tuples.pair(xDelta, gradDelta));
                while (this.history.size() > maxHistorySize) {
                    this.history.remove(this.history.size()-1);
                }
              }
        };
    }
}
