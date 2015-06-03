package org.allenai.ml.optimize;

import org.allenai.ml.linalg.Vector;
import lombok.extern.slf4j.Slf4j;
import lombok.val;

import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Family of gradient minimizer based on Newton's method. Each iteration of the algorithm takes a current
 * minimum guess and then based on that guess, a next step direction is chosen using the `QuasiNewton` approximation.
 * We perform a line-minimization along that direction for our new guess. Depending on the `QuasiNewton`, you get different
 * algorithms.
 */
@Slf4j(topic = "NewtonMethodOptimize")
public class NewtonMethod implements GradientFnMinimizer {

    private final Function<GradientFn, QuasiNewton> quasiNewtonFn;
    private final Opts opts;

    public static class Opts {
        public int maxIters = 150;
        public double tolerance = 1.0e-10;
        public double alpha = 0.5;
        public double beta = 0.01;
        public double stepLenTolerance = 1.0e-10;
        public Consumer<Vector> iterCallback;
        public LineMinimizer lineMinimizer() {
            return BacktrackingLineMinimizer.of(alpha, beta, stepLenTolerance);
        }
    }

    public NewtonMethod(Function<GradientFn, QuasiNewton> quasiNewtonFn) {
        this(quasiNewtonFn, new Opts());
    }

    public NewtonMethod(Function<GradientFn, QuasiNewton> quasiNewtonFn, Opts opts) {
        this.quasiNewtonFn = quasiNewtonFn;
        this.opts = opts;
    }

    private Vector step(GradientFn gradFn, Vector x, LineMinimizer ls, QuasiNewton qn) {
        Vector grad = gradFn.apply(x).grad;
        Vector dir = qn.implictMultiply(grad);
        dir.scaleInPlace(-1.0);
        val lsRes = ls.minimize(gradFn, x, dir);
        return x.add(lsRes.stepLength, dir);
    }

    private final static double EPS = 1.0e-200;

    @Override
    public Result minimize(GradientFn gradFn, Vector initGuess) {
        QuasiNewton qn = this.quasiNewtonFn.apply(gradFn);
        val lm = this.opts.lineMinimizer();
        Vector x = initGuess;
        log.info("Optimization started with {} parameters", initGuess.dimension());
        for (int i=0; i < opts.maxIters; ++i) {
            // iteration
            long start = System.currentTimeMillis();
            val curRes = gradFn.apply(x);
            Vector xnew = step(gradFn, x, lm, qn);
            val newRes = gradFn.apply(xnew);
            if (newRes.fx > curRes.fx) {
                throw new IllegalStateException(
                    String.format("Step increased function value: old %.3f new: %.3f", curRes.fx, newRes.fx));
            }
            double larger = Math.min(Math.abs(curRes.fx), Math.abs(newRes.fx));
            double relDiff = Math.abs(newRes.fx-curRes.fx)/ Math.max(larger, EPS);
            long stop = System.currentTimeMillis();
            log.info("[Iteration {}][{} ms] Ended with value {} and relDiff {}", i, (stop-start), newRes.fx, relDiff);
            if (relDiff < opts.tolerance) {
                break;
            }
            // update
            qn.update(xnew.add(-1.0,x), newRes.grad.add(-1.0, curRes.grad));
            x = xnew;
            if (opts.iterCallback != null) {
                opts.iterCallback.accept(x);
            }
        }
        return Result.of(gradFn.apply(x).fx, x);
    }
}