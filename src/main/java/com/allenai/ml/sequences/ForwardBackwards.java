package com.allenai.ml.sequences;

import com.allenai.ml.math.SloppyMath;
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
        private final double[][] alpahs = computeAlphas(SloppyMath::logSumExp);

        public double getLogZ() {
            return getAlpahs()[seqLen-1][stateSpace.stopStateIndex()];
        }

        private double[][] computeAlphas(ToDoubleFunction<double[]> combiner) {
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
                    double[] forwardPaths = stateSpace.transitionsTo(s)
                        .stream()
                        .mapToDouble(t -> alphas[prevPos][t.fromState] + potentials[prevPos][t.selfIndex])
                        .toArray();
                    alphas[i][s] = combiner.applyAsDouble(forwardPaths);
                }
            }
            return alphas;
        }

        private List<S> computeViterbi() {
            // Use the MAX operation to compute alphas
            double[][] maxAlphas = computeAlphas(xs -> DoubleStream.of(xs).max().orElse(Double.NEGATIVE_INFINITY));
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
                Optional<Transition> correctTransition = stateSpace.transitionsTo(targetState)
                        .stream()
                        .filter(trans -> {
                            double value = potentials[curPos][trans.selfIndex] + maxAlphas[curPos][trans.fromState];
                            return Math.abs(value - curTarget) < 1.0e-8;
                        })
                        .findFirst();
                if (!correctTransition.isPresent()) {
                    throw new RuntimeException("viterbi can't find path found by computeAlphas(MAX)");
                }
                targetState = correctTransition.get().fromState;
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
    }
}
