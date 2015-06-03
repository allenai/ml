package com.allenai.ml.optimize;

import com.allenai.ml.linalg.Vector;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RequiredArgsConstructor(staticName = "of")
public class BacktrackingLineMinimizer implements LineMinimizer {
    final double alpha, beta, minStepLen;

    private final static Logger logger = LoggerFactory.getLogger(BacktrackingLineMinimizer.class);

    @Override
    public Result minimize(GradientFn gradFn, Vector x, Vector dir) {
        val valGradPair = gradFn.apply(x);
        double f0 = valGradPair.fx;
        val grad = valGradPair.grad;
        logger.trace("Starting with grad l2NormSquared: {}", grad.l2NormSquared());
        if (grad.l2NormSquared() < minStepLen) {
            return Result.of(0.0, f0);
        }
        final double delta = beta * grad.dotProduct(dir);
        double stepLen = 1.0;
        while (stepLen >= minStepLen) {
            Vector stepX = x.add(stepLen, dir);
            final double fx = gradFn.apply(stepX).fx;
            logger.trace("Step size: alpha {}, new {}, old {}", stepLen, fx, f0);
            if (fx <= f0 + stepLen * delta) {
                return Result.of(stepLen, fx);
            }
            stepLen *= alpha;
        }
        throw new RuntimeException("Step-size underflow: can't make the value smaller along gradient");
    }
}
