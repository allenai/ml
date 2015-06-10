package org.allenai.ml.classification;

public interface Classifier<D, L> {
    L bestGuess(D datum);
}
