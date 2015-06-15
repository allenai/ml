package org.allenai.ml.sequences;

import com.gs.collections.api.tuple.Pair;
import org.allenai.ml.util.Parallel;

import java.util.List;

import static java.util.stream.Collectors.toList;

public class TokenAccuracy {
    public static <S, O> double compute(SequenceTagger<S, O> tagger, List<List<Pair<S, O>>> data, Parallel.MROpts mrOpts) {
        class AccStats {
            int numer = 0;
            int denom = 0;
        }
        Parallel.MapReduceDriver<List<Pair<S, O>>, AccStats> driver =
                new Parallel.MapReduceDriver<List<Pair<S, O>>, AccStats>() {
            @Override
            public AccStats newData() {
                return new AccStats();
            }

            @Override
            public void update(AccStats data, List<Pair<S, O>> elem) {
                List<O> input = elem.stream().map(Pair::getTwo).collect(toList());
                List<S> gold = elem.stream().map(Pair::getOne).collect(toList())
                    .subList(1, elem.size()-1);
                List<S> guess = tagger.bestGuess(input);
                for (int idx = 0; idx < gold.size(); idx++) {
                    if (gold.get(idx).equals(guess.get(idx))) {
                        data.numer ++;
                    }
                    data.denom++;
                }
            }

            @Override
            public void merge(AccStats a, AccStats b) {
                a.numer += b.numer;
                a.denom += b.denom;
            }
        };
        AccStats stats = Parallel.mapReduce(data, driver, mrOpts);
        return (double)stats.numer/(double)stats.denom;
    }
}
