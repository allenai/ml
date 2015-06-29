package org.allenai.ml.eval;

public class FMeasure {
    private int tp;
    private int fp;
    private int fn;

    public double precision() {
        return tp + fp > 0 ? (double) tp / (double)(tp + fp) : 0.0;
    }

    public double recall() {
        return tp + fn > 0 ? (double) tp / (double)(tp + fn) : 0.0;
    }

    public int numTrueEvents() {
        return tp + fn;
    }

    public double f1() {
        double p = precision();
        double r = recall();
        return p + r > 0.0 ? (2 * p * r) / (p + r) : 0.0;
    }

    public static enum Event {
        CORRECT_HIT,
        INCORRECT_GUESS,
        INCORRECT_MISS
    }

    public void update(Event event) {
        switch (event) {
            case CORRECT_HIT:
                tp++;
                break;
            case INCORRECT_GUESS:
                fp++;
                break;
            case INCORRECT_MISS:
                fn++;
                break;
        }
    }

    public void merge(FMeasure other) {
        tp += other.tp;
        fp += other.fp;
        fn += other.fn;
    }
}
