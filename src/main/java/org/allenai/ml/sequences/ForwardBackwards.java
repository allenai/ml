package org.allenai.ml.sequences;

import com.gs.collections.api.list.primitive.DoubleList;
import com.gs.collections.api.list.primitive.MutableDoubleList;
import com.gs.collections.impl.list.mutable.primitive.DoubleArrayList;
import org.allenai.ml.math.SloppyMath;
import lombok.AccessLevel;
import lombok.Getter;

import java.util.*;
import java.util.function.ToDoubleFunction;
import java.util.stream.DoubleStream;

/**
 * Efficient implementation of the ForwardBackwards algorithm
 */
public class ForwardBackwards<S> {
    private final StateSpace<S> stateSpace;
    private final int numStates;
    private final int numTransitions;

    public ForwardBackwards(StateSpace<S> stateSpace) {
        this.stateSpace = stateSpace;
        this.numStates = stateSpace.states().size();
        this.numTransitions = stateSpace.transitions().size();
    }

    /**
     *
     * @param logPotentials 2d array of the scores associated with the state-space transition for each sequence position.
     *                      Specifically there are [seqLen-1][numStateSpaceTransitions] values, where
     *                      logPotentials[i][t] represents the log-space score for using the `t`th indexed transition
     *                      in the underlying `StateSpace` and `i` represents the `i`th transition.
     * @return Result object that lazily yields ForwardBackwards quantities. 
     */
    public Result compute(double[][] logPotentials) {
        return new Result(logPotentials);
    }

    interface RingOp {
        void clear();
        void add(double x);
        double compute();
    }

    class MaxRing implements RingOp {
        private double max = Double.NEGATIVE_INFINITY;

        @Override
        public void clear() {
            max = Double.NEGATIVE_INFINITY;
        }

        @Override
        public void add(double x) {
            if (x > max) {
                max = x;
            }
        }

        @Override
        public double compute() {
            return max;
        }
    }

    class LogAddRing implements RingOp {
        MutableDoubleList xs = new DoubleArrayList(stateSpace.transitions().size());
        double max = Double.NEGATIVE_INFINITY;

        public void clear() {
            xs.clear();
            max = Double.NEGATIVE_INFINITY;
        }

        @Override
        public void add(double x) {
            if (x > max) {
                max = x;
            }
            if (x - max >= SloppyMath.EXP_THRESH) {
                xs.add(x);
            }
        }

        @Override
        public double compute() {
            double sumExpNegDiffs = 0.0;
            for (int idx = 0; idx < xs.size(); idx++) {
                double x = xs.get(idx);
                sumExpNegDiffs += SloppyMath.sloppyExp(x - max);
            }
            return sumExpNegDiffs > 0.0
                ? max + Math.log(sumExpNegDiffs)
                : max;
        }
    }

    /**
     * A class that lazily computes various quantities associated with the ForwardBackwards algorithms. You only
     * pay for the things you actually consume (e.g. if you only need the viterbi decoding, you don't do the backward
     * pass).
     */
    public class Result {
        private final double[][] potentials;
        private final int seqLen;

        private Result(double[][] potentials) {
            this.potentials = potentials;
            this.seqLen = potentials.length+1;
        }

        @Getter(lazy = true)
        private final List<S> viterbi = computeViterbi();

        @Getter(value = AccessLevel.PRIVATE, lazy = true)
        private final double[][] alphas = computeAlphas(new LogAddRing());

        @Getter(value = AccessLevel.PRIVATE, lazy = true)
        private final double[][] betas = computeBetas();

        @Getter(value = AccessLevel.PUBLIC, lazy = true)
        private final double[][] nodeMarginals = computeNodeMarginals();

        @Getter(value = AccessLevel.PUBLIC, lazy = true)
        private final double[][] edgeMarginals = computeEdgeMarginals();

        public double getLogZ() {
            return getAlphas()[seqLen-1][stateSpace.stopStateIndex()];
        }

        private double[][] computeAlphas(RingOp ringOp) {
            double[][] alphas = new double[seqLen][numStates];
            for (double[] row: alphas) {
                Arrays.fill(row, Double.NEGATIVE_INFINITY);
            }
            // initialize
            alphas[0][stateSpace.startStateIndex()] = 0.0;
            // go forwards
            for (int i=1; i < seqLen; ++i) {
                int prevPos = i-1;
                for (int s=0; s < numStates; ++s) {
                    // potential bottleneck
                    ringOp.clear();
                    for (Transition t : stateSpace.transitionsTo(s)) {
                        double pathVal = alphas[prevPos][t.fromState] + potentials[prevPos][t.selfIndex];
                        ringOp.add(pathVal);
                    }
                    alphas[i][s] = ringOp.compute();
                }
            }
            return alphas;
        }

        private double[][] computeBetas() {
            double[][] betas = new double[seqLen][numStates];
            for (double[] row: betas) {
                Arrays.fill(row, Double.NEGATIVE_INFINITY);
            }
            // initialize
            betas[seqLen-1][stateSpace.stopStateIndex()] = 0.0;
            RingOp ringOp = new LogAddRing();
            // go forwards
            for (int i=seqLen-2; i >= 0; --i) {
                // create new effectively final variable for lambda use
                int curPos = i;
                int nextPos = i+1;
                for (int s=0; s < numStates; ++s) {
                    // potential bottleneck
                    ringOp.clear();
                    for (Transition t : stateSpace.transitionsFrom(s)) {
                        double val = betas[nextPos][t.toState] + potentials[curPos][t.selfIndex];
                        ringOp.add(val);
                    }
                    betas[i][s] = ringOp.compute();
                }
            }
            return betas;
        }


        private List<S> computeViterbi() {
            // Use the MAX operation to compute alphas
            double[][] maxAlphas = computeAlphas(new MaxRing());
            // Compute the best path iteratively by figuring out which operation lead to it rather
            // than compute and store back-pointers, this is more efficient since a memory read is much
            // cheaper than a write
            double targetValue =  maxAlphas[seqLen-1][stateSpace.stopStateIndex()];
            int targetState = stateSpace.stopStateIndex();
            List<S> result = new ArrayList<>();
            for (int pos=seqLen-2; pos >= 0; --pos) {
                // Need these next two variables to be effectively final
                int curPos = pos;
                double curTarget = targetValue;
                // Find the transition that leads to the target value
                Transition correctTransition = null;
                for (Transition trans : stateSpace.transitionsTo(targetState)) {
                    double value = potentials[curPos][trans.selfIndex] + maxAlphas[curPos][trans.fromState];
                    if (Math.abs(value - curTarget) < 1.0e-8) {
                        correctTransition = trans;
                        break;
                    }
                }
                if (correctTransition == null) {
                    throw new RuntimeException("viterbi can't find path found by computeAlphas(MAX)");
                }
                targetState = correctTransition.fromState;
                targetValue = maxAlphas[pos][targetState];
                // Add state to result as long as not start state
                if (pos > 0) {
                    result.add(stateSpace.states().get(targetState));
                }
            }
            // We've built answer backwards
            Collections.reverse(result);
            return result;
        }

        private double[][] computeNodeMarginals() {
            double[][] alphas = getAlphas();
            double[][] nodeMarginals = new double[seqLen][numStates];
            double[][] edgeMarginals = getEdgeMarginals();
            // Fist: Must Have All Mass on Start State
            nodeMarginals[0][stateSpace.startStateIndex()] = 1.0;
            // Last: Must Have All Mass on Stop State
            nodeMarginals[seqLen-1][stateSpace.stopStateIndex()] = 1.0;
            // Middle States: leverage edge marginals to compute node marginals
            for (int i=1; i < seqLen-1; ++i) {
                for (int s=0; s < numStates; ++s) {
                    if (alphas[i][s] == Double.NEGATIVE_INFINITY) {
                        continue;
                    }
                    // potential bottleneck
                    for (Transition t : stateSpace.transitionsFrom(s)) {
                        nodeMarginals[i][s] += edgeMarginals[i][t.selfIndex];
                    }
                }
            }
            return nodeMarginals;
        }

        private double[][] computeEdgeMarginals() {
            // will trigger alphas, betas computation if not already computed
            double[][] alphas = getAlphas();
            double[][] betas = getBetas();
            double[][] edgeMarginals = new double[seqLen-1][numTransitions];
            double logZ = getLogZ();
            for (int i=0; i < seqLen-1; ++i) {
                for (int s=0; s < numStates; ++s) {
                    // skip edges which land on an impossible state
                    if (alphas[i][s] == Double.NEGATIVE_INFINITY) {
                        continue;
                    }
                    for (Transition t: stateSpace.transitionsFrom(s)) {
                        // score for all paths that use `t` transition. Three pieces
                        // (1) score to paths that lead to start of transition (alphas[i][s])
                        // (2) score of transition itself
                        // (3) score of all paths starting at t.toState
                        if (potentials[i][t.selfIndex] == Double.NEGATIVE_INFINITY ||
                            betas[i+1][t.toState] == Double.NEGATIVE_INFINITY) {
                            continue;
                        }
                        double logNumer = alphas[i][s] + potentials[i][t.selfIndex] + betas[i+1][t.toState];
                        edgeMarginals[i][t.selfIndex] = SloppyMath.sloppyExp(logNumer - logZ);
                    }
                }
            }
            return edgeMarginals;
        }
    }
}
