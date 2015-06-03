package org.allenai.ml.optimize;

import org.allenai.ml.linalg.Vector;
import lombok.RequiredArgsConstructor;
import lombok.val;

import java.util.LinkedList;
import java.util.Optional;

@RequiredArgsConstructor
public class CachingGradientFn implements GradientFn {
    private final int maxHistory;
    private final GradientFn gradFn;
    private final LinkedList<HistoryEntry> history = new LinkedList<>();

    @RequiredArgsConstructor
    public class HistoryEntry {
        final Vector input;
        final double output;
        final Vector grad;
    }

    @Override
    public GradientFn.Result apply(Vector x) {
        Optional<HistoryEntry> foundEntry = this.history
            .stream()
            .filter(e -> e.input.closeTo(x))
            .findFirst();
        if (foundEntry.isPresent()) {
            return GradientFn.Result.of(foundEntry.get().output, foundEntry.get().grad);
        }
        val result = this.gradFn.apply(x);
        val entry = new HistoryEntry(x, result.fx, result.grad);
        this.history.addFirst(entry);
        if (this.history.size() > this.maxHistory) {
            this.history.removeLast();
        }
        return result;
    }
    @Override
    public long dimension() {
        return gradFn.dimension();
    }
}