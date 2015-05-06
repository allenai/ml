package com.allenai.ml.sequences;

import com.gs.collections.api.list.MutableList;
import com.gs.collections.api.map.primitive.MutableIntObjectMap;
import com.gs.collections.api.tuple.Pair;
import com.gs.collections.impl.list.mutable.FastList;
import com.gs.collections.impl.map.mutable.primitive.IntObjectHashMap;
import com.gs.collections.impl.tuple.Tuples;
import lombok.val;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * A container object for various aspects of the state space. This object is only meant to be built via factory methods.
 */
public class StateSpace<S> {
    private final List<S> states;
    private final List<Transition> transitions;
    private final MutableIntObjectMap<MutableList<Transition>> fromTransitions;
    private final MutableIntObjectMap<MutableList<Transition>> toTransitions;

    /**
     * Package private constructor.
     * @param states It's assumed that states[0] and states[1] are the start/stop states.
     * @param transitionPairs All allowed transitions, including those starting/ending from start/stop respectively
     */
    StateSpace(List<S> states, Set<Pair<S, S>> transitionPairs) {
        Map<S, Integer> stateIndex = IntStream.range(0, states.size())
                .boxed()
                .collect(Collectors.toMap(states::get, Function.identity()));
        this.states = new ArrayList<>(states);
        this.transitions = new ArrayList<>();
        for (Pair<S, S> transitionPair : transitionPairs) {
            int startIndex = stateIndex.get(transitionPair.getOne());
            int stopIndex = stateIndex.get(transitionPair.getTwo());
            int transitionIndex = transitions.size();
            transitions.add(Transition.of(startIndex, stopIndex, transitionIndex));
        }
        this.fromTransitions = new IntObjectHashMap<>(this.states.size());
        this.toTransitions = new IntObjectHashMap<>(this.states.size());
        for (Transition trans : transitions) {
            fromTransitions
                .getIfAbsentPut(trans.fromState, FastList::new)
                .add(trans);
            toTransitions
                .getIfAbsentPut(trans.toState, FastList::new)
                .add(trans);

        }
    }

    public Optional<Transition> transitionFor(S from, S to) {
        int fromIndex = states.indexOf(from);
        int toIndex = states.indexOf(to);
        if (fromIndex < 0 || toIndex < 0) {
            return Optional.empty();
        }
        return transitionFrom(fromIndex).stream()
            .filter(t -> t.toState == toIndex)
            .findFirst();
    }

    public List<Transition> transitionFrom(int stateIndex) {
        return Collections.unmodifiableList(fromTransitions.getIfAbsent(stateIndex, FastList::new));
    }

    public List<Transition> transitionsTo(int stateIndex) {
        return Collections.unmodifiableList(toTransitions.getIfAbsent(stateIndex, FastList::new));
    }

    public Pair<S, S> transition(int transitionIndex) {
        val t = transitions.get(transitionIndex);
        return Tuples.pair(states.get(t.fromState), states.get(t.toState));
    }

    public S startState() {
        return states.get(0);
    }

    public S stopState() {
        return states.get(1);
    }

    /**
     * __NOTE__: This isn't a performant method and is intended only for debugging
     * @return Index of the underlying state or -1 if not there.
     */
    public int stateIndex(S state) {
        return states.indexOf(state);
    }

    public List<S> states() {
        return Collections.unmodifiableList(states);
    }

    public List<Transition> transitions() {
        return Collections.unmodifiableList(transitions);
    }

    public int startStateIndex() {
        return 0;
    }

    public int stopStateIndex() {
        return 1;
    }

    private static <S> List<S> ensureStartStopPadded(List<S> seq, S start, S stop) {
        // defensive copy
        seq = new ArrayList<>(seq);
        if (seq.size() == 0 || !seq.get(0).equals(start)) {
            seq.add(0, start);
        }
        int lastIndex = seq.size()-1;
        if (seq.size() < 2 || !seq.get(lastIndex).equals(stop)) {
            seq.add(stop);
        }
        return seq;
    }

    private static <S> List<Pair<S,S>> transitions(List<S> seq) {
        List<Pair<S, S>> pairs = new ArrayList<>(seq.size()-1);
        for (int idx=0; idx < seq.size()-1; ++idx) {
            pairs.add(Tuples.pair(seq.get(idx), seq.get(idx+1)));
        }
        return pairs;
    }

    public static <S> StateSpace<S> buildFromSequences(Collection<List<S>> sequences, S startState, S stopState) {
        List<S> states = new ArrayList<>();
        states.addAll(Arrays.asList(startState, stopState));
        Set<S> nonStartStopStates = sequences.stream()
                .flatMap(Collection::stream)
                .filter(s -> !s.equals(startState) && !s.equals(stopState))
                .collect(Collectors.toSet());
        states.addAll(nonStartStopStates);
        Set<Pair<S, S>> transitionPairs = sequences.stream()
                .map(seq -> ensureStartStopPadded(seq, startState, stopState))
                .flatMap(seq -> transitions(seq).stream())
                .collect(Collectors.toSet());
        return new StateSpace<>(states, transitionPairs);
    }
}
