package com.allenai.ml.sequences;

import java.util.List;

@FunctionalInterface
public interface SequenceTagger<S, O> {
    List<S> bestGuess(List<O> input);
}
