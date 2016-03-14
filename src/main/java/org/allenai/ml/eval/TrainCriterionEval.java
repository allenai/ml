package org.allenai.ml.eval;

import lombok.extern.log4j.Log4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.function.DoubleFunction;
import java.util.function.Predicate;
import java.util.function.ToDoubleBiFunction;
import java.util.function.ToDoubleFunction;

public class TrainCriterionEval<M> implements Predicate<M> {
  private final ToDoubleFunction<M> baseEvalFn;
  private int numIters = 0;
  private int dipIter = -1;
  private double lastVal = Double.NEGATIVE_INFINITY;
  public M bestModel = null;

  private final static Logger log = LoggerFactory.getLogger(TrainCriterionEval.class);

  /**
   * How much of a dip in eval constitutes a down-step
   */
  public double dipTolerance = 1.0e-4;

  /**
   * Maximum number of iters to wait after
   * dip to quit training. If this
   * is set to 0, will quit when
   * you hit the dip.
   */
  public int maxNumDipIters = 0;

  public TrainCriterionEval(ToDoubleFunction<M> baseEvalFn) {
    this.baseEvalFn = baseEvalFn;
  }

  public M getBestModel() {
    return bestModel;
  }

  @Override
  public boolean test(M m) {
    double testEval = baseEvalFn.applyAsDouble(m);

    log.info(String.format("[Iteration %d] Eval metric: %.3f", this.numIters, testEval));

    double delta = lastVal - testEval;

    this.numIters++;

    if (delta > this.dipTolerance) {
      int numDipIters = this.numIters - this.dipIter;
      if (numDipIters > this.maxNumDipIters) {
        log.info("Exceeded max dip iters, bailing");
        return false;
      } else {
        int numLeft = this.maxNumDipIters - numDipIters;
        log.info("Another down iteration, waiting " + numLeft + " more iters");
      }
    } else {
      dipIter = -1;
      lastVal = testEval;
      bestModel = m;
    }

    return true;
  }
}
