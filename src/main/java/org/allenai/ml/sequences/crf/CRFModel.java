package org.allenai.ml.sequences.crf;

import org.allenai.ml.linalg.Vector;
import org.allenai.ml.sequences.ForwardBackwards;
import org.allenai.ml.sequences.SequenceTagger;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.val;

import java.util.ArrayList;
import java.util.List;

@RequiredArgsConstructor
public class CRFModel<S, O, F> implements SequenceTagger<S, O> {
    public final CRFFeatureEncoder<S, O, F> featureEncoder;
    public final CRFWeightsEncoder<S> weightsEncoder;
    // This is private because it's mutable. The weights() method
    // will return a copy
    private  final Vector weights;
    @Setter
    private InferenceMode inferenceMode = InferenceMode.VITERBI;

    public static enum InferenceMode {
        VITERBI,
        MAX_TOKEN
    }

    /**
     * Return copy of weights
     */
    public Vector weights() {
        return weights.copy();
    }



    @Override
    public List<S> bestGuess(List<O> input) {
        input = new ArrayList<>(input);
        val indexedExample = featureEncoder.indexedExample(input);
        double[][] potentials = weightsEncoder.fillPotentials(weights, indexedExample);
        val forwardBackwards = new ForwardBackwards<>(featureEncoder.stateSpace);
        ForwardBackwards.Result fbResult = forwardBackwards.compute(potentials);
        if (inferenceMode == InferenceMode.VITERBI) {
            return fbResult.getViterbi();
        }
        double[][] edgeMarginals = fbResult.getEdgeMarginals();
        return forwardBackwards.compute(edgeMarginals).getViterbi();
    }
}
