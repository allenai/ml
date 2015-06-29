package org.allenai.ml.sequences;

import com.gs.collections.api.tuple.Pair;
import lombok.RequiredArgsConstructor;
import lombok.val;
import org.allenai.ml.eval.Accuracy;
import org.allenai.ml.eval.FMeasure;
import org.allenai.ml.util.Parallel;

import java.util.*;

import static java.util.stream.Collectors.toList;

@RequiredArgsConstructor
public class Evaluation<S> {

    public final Accuracy tokenAccuracy;
    public final Map<S, FMeasure> stateFMeasures;

    public static <S, O> Evaluation<S> compute(SequenceTagger<S, O> tagger, List<List<Pair<S, O>>> data, Parallel.MROpts mrOpts) {
        Parallel.MapReduceDriver<List<Pair<S, O>>, Evaluation<S>> driver =
                new Parallel.MapReduceDriver<List<Pair<S, O>>, Evaluation<S>>() {
            @Override
            public Evaluation<S> newData() {
                return new Evaluation<>(new Accuracy(), new HashMap<>());
            }

            @Override
            public void update(Evaluation<S> eval, List<Pair<S, O>> elem) {
                List<O> input = elem.stream().map(Pair::getTwo).collect(toList());
                List<S> gold = elem.stream().map(Pair::getOne).collect(toList())
                    .subList(1, elem.size()-1);
                List<S> guess = tagger.bestGuess(input);
                for (int idx = 0; idx < gold.size(); idx++) {
                    S goldLabel = gold.get(idx);
                    S guessLabel = guess.get(idx);
                    boolean correctGuess = goldLabel.equals(guessLabel);
                    eval.tokenAccuracy.update(correctGuess);
                    if (correctGuess) {
                        val fmeasure = eval.stateFMeasures.computeIfAbsent(goldLabel, (__) -> new FMeasure());
                        fmeasure.update(FMeasure.Event.CORRECT_HIT);
                    } else {
                        FMeasure wrongGuessMeasure =
                            eval.stateFMeasures.computeIfAbsent(guessLabel, (__) -> new FMeasure());
                        wrongGuessMeasure.update(FMeasure.Event.INCORRECT_GUESS);
                        eval.stateFMeasures.computeIfAbsent(goldLabel, (__) -> new FMeasure())
                            .update(FMeasure.Event.INCORRECT_MISS);
                    }
                }
            }

            @Override
            public void merge(Evaluation<S> a, Evaluation<S> b) {
                a.tokenAccuracy.combine(b.tokenAccuracy);
                Set<S> allStates = new HashSet<>();
                allStates.addAll(a.stateFMeasures.keySet());
                allStates.addAll(b.stateFMeasures.keySet());
                for (S state : allStates) {
                    FMeasure aStats = a.stateFMeasures.getOrDefault(state, new FMeasure());
                    FMeasure bStats = b.stateFMeasures.getOrDefault(state, new FMeasure());
                    aStats.merge(bStats);
                    a.stateFMeasures.put(state, aStats);
                }
            }
        };
        return Parallel.mapReduce(data, driver, mrOpts);
    }
}
