package org.allenai.ml.sequences.crf;

import org.allenai.ml.linalg.Vector;
import org.allenai.ml.sequences.ForwardBackwards;
import org.allenai.ml.sequences.SequenceTagger;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.val;
import org.allenai.ml.util.Indexer;

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


    private static <F> List<String> diff(String prefix, Indexer<F> x, Indexer<F> y) {
        List<String> res = new ArrayList<>();
        if (x.size() != y.size()) {
            res.add(prefix + " - size diff " + x.size() + " != " + y.size());
        }
        int n = Math.min(x.size(), y.size());
        for (int idx=0; idx < n; ++idx) {
            if (!x.get(idx).equals(y.get(idx))) {
                res.add(prefix + " - elem diff " + idx + " "  + x.get(idx)  + " / "  + y.get(idx));
            }
        }
        return res;
    }

    private static List<String> diff(String prefix, Vector x, Vector y) {
        List<String> res = new ArrayList<>();
        if (x.dimension() != y.dimension()) {
            res.add(prefix + " - dim diff " + x.dimension() + " != " + y.dimension());
        }
        long n = Math.min(x.dimension(), y.dimension());
        for (long idx=0; idx < n; ++idx) {
            double xx = x.at(idx);
            double yy = y.at(idx);
            if (Math.abs(xx-yy) > 0.001) {
                res.add(prefix + " - elem diff " + idx + " "  + x.at(idx)  + " / "  + y.at(idx));
            }
        }
        return res;
    }

    public List<String> diff(CRFModel<S, O, F> otherModel) {
        List<String> res = new ArrayList<>();
        res.addAll(diff("nodeFeatures", this.featureEncoder.nodeFeatures, otherModel.featureEncoder.nodeFeatures));
        res.addAll(diff("edgeFeatures", this.featureEncoder.nodeFeatures, otherModel.featureEncoder.nodeFeatures));
        res.addAll(diff("weightDiffs", this.weights, otherModel.weights));
        return res;
    }

    @Override
    public List<S> bestGuess(List<O> input) {
        if (input.size() < 2) {
            throw new IllegalArgumentException("Need to have at least two elements");
        }
        if (input.size() == 2) {
            // only have start stop, so return empty (unpadded)
            return new ArrayList<>();
        }
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
