package com.allenai.ml.sequences.crf;

import com.allenai.ml.linalg.Vector;
import com.allenai.ml.sequences.StateSpace;
import com.allenai.ml.sequences.Transition;
import lombok.RequiredArgsConstructor;

import java.util.List;

@RequiredArgsConstructor
public class CRFWeightsEncoder<S> {
    public final StateSpace<S> stateSpace;
    public final int numNodePredicates;
    public final int numEdgePredicates;


    /**
     * Parameters for a CRF problem are the number of node predicates needed for each label and for each
     * state space transition.
     * @return total number of parameters
     */
    public int numParameters() {
        int numNodeParameters = numNodePredicates * stateSpace.states().size();
        int numEdgeParameters = numEdgePredicates * stateSpace.transitions().size();
        return numNodeParameters + numEdgeParameters;
    }

    // only visible for testing
    static double[] fillRowPotentials(Vector weights, Vector.Iterator predValueIt, int numValues, int weightOffset) {
        double[] rowPotentials = new double[numValues];
        for (int i = 0; i < numValues; i++) {
            double score = 0.0;
            while (!predValueIt.isExhausted()) {
                long predIdx = predValueIt.index();
                double predVal = predValueIt.value();
                long weightIdx = predIdx * numValues + i + weightOffset;
                score +=  weights.at(weightIdx) * predVal;
                predValueIt.advance();
            }
            rowPotentials[i] = score;
            predValueIt.reset();
        }
        return rowPotentials;
    }

    /**
     * Produce an array of the scores for the states given the input predicate indices/values. Note this
     * code tends to be a bottleneck, so written more producedurally than otherwise
     * @param weights CRF weights
     * @param predValueIt Iterator over the pred index/values we want to score
     * @return array in size of `stateSpace.states().size()` of scores
     */
    private double[] nodePotentials(Vector weights, Vector.Iterator predValueIt) {
        int numStates = stateSpace.states().size();
        return fillRowPotentials(weights, predValueIt, numStates, 0);
    }

    private double[] edgePotentials(Vector weights, Vector.Iterator predValueIt) {
        List<Transition> transitions = stateSpace.transitions();
        int transitionOffset = numNodePredicates;
        return fillRowPotentials(weights, predValueIt, transitions.size(), transitionOffset);
    }

    double[][] fillPotentials(Vector weights,  CRFIndexedExample example) {
        int numTransitions = example.getSequenceLength() - 1;
        List<Transition> transitions = stateSpace.transitions();
        double[][] potentials = new double[numTransitions][transitions.size()];
        for (int i=0; i < numTransitions; ++i) {
            // The index is in terms of state
            double[] nodePotentials = nodePotentials(weights, example.getNodePredicateValues(i));
            double[] edgePotentials = edgePotentials(weights, example.getEdgePredicateValues(i));
            for (int t = 0; t < transitions.size(); t++) {
                int s = transitions.get(t).fromState;
                potentials[i][t] = edgePotentials[t] + nodePotentials[s];
            }
        }
        return potentials;
    }

    public int nodeWeightIndex(int predIdx, int fromState) {
        return predIdx * stateSpace.states().size() + fromState;
    }

    public int edgeWeightIndex(int predIdx, int transitionIdx) {
        return numNodePredicates + predIdx * stateSpace.transitions().size() + transitionIdx;
    }
}
