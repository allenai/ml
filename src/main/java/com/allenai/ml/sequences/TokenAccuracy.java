package com.allenai.ml.sequences;

import com.gs.collections.api.tuple.Pair;

import java.util.List;

import static java.util.stream.Collectors.toList;

public class TokenAccuracy {
    public static <S, O> double compute(SequenceTagger<S, O> tagger, List<List<Pair<S, O>>> data) {
        int numer = 0;
        int denom = 0;
        for (List<Pair<S, O>> elem : data) {
            List<O> input = elem.stream().map(Pair::getTwo).collect(toList());
            List<S> gold = elem.stream().map(Pair::getOne).collect(toList())
                .subList(1, elem.size()-1);
            List<S> guess = tagger.bestGuess(input);
            for (int idx = 0; idx < gold.size(); idx++) {
                if (gold.get(idx).equals(guess.get(idx))) {
                    numer ++;
                }
                denom++;
            }
        }
        return (double)numer/(double)denom;
    }
}
