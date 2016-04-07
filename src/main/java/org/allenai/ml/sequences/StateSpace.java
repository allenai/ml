package org.allenai.ml.sequences;

import org.allenai.ml.util.IOUtils;

import com.gs.collections.api.list.MutableList;
import com.gs.collections.api.map.primitive.MutableIntObjectMap;
import com.gs.collections.api.tuple.Pair;
import com.gs.collections.impl.list.mutable.FastList;
import com.gs.collections.impl.map.mutable.primitive.IntObjectHashMap;
import com.gs.collections.impl.tuple.Tuples;
import lombok.SneakyThrows;
import lombok.val;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
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
     * @param states It's assumed that states[0] and states[1] are the start/stop states and no duplicates.
     * @param transitionPairs All allowed transitions, including those starting/ending from start/stop respectively.
     *                        Assumes no duplicates.
     */
    public StateSpace(List<S> states, List<Pair<S, S>> transitionPairs) {
        if (states.size() != new HashSet<>(states).size()) {
            throw new IllegalArgumentException("Passed in duplicate states: " + states);
        }
        if (transitionPairs.size() != new HashSet<>(transitionPairs).size()) {
            throw new IllegalArgumentException("Passed in transition pairs");
        }
        Map<S, Integer> stateIndex = IntStream.range(0, states.size())
                .boxed()
                .collect(Collectors.toMap(states::get, Function.identity()));
        this.states = new ArrayList<>(states);
        this.transitions = new ArrayList<>();
        for (Pair<S, S> transitionPair : transitionPairs) {
            int startIndex = stateIndex.get(transitionPair.getOne());
            int stopIndex = stateIndex.get(transitionPair.getTwo());
            int transitionIndex = transitions.size();
            transitions.add(new Transition(startIndex, stopIndex, transitionIndex));
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
        return transitionFor(fromIndex, toIndex);
    }

    public Optional<Transition> transitionFor(int fromIndex, int toIndex) {
        return transitionsFrom(fromIndex).stream()
            .filter(t -> t.toState == toIndex)
            .findFirst();
    }

    public List<Transition> transitionsFrom(int stateIndex) {
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

    public static <S> StateSpace<S> buildFullStateSpace(Set<S> stateSet, S startState, S stopState) {
        List<S> states = new ArrayList<>();
        states.add(startState);
        states.add(stopState);
        stateSet = new HashSet<>(stateSet);
        stateSet.remove(startState);
        stateSet.remove(stopState);
        states.addAll(stateSet);

        List<Pair<S, S>> pairs = new ArrayList<>(states.size() * states.size());
        for (S s: states) {
            for (S t: states) {
                if (s != stopState && t != startState) {
                    pairs.add(Tuples.pair(s, t));
                }
            }
        }
        return new StateSpace<>(states, pairs);
    }

    public static <S> StateSpace<S> buildFromSequences(Collection<List<S>> sequences, S startState, S stopState) {
        List<S> states = new ArrayList<>();
        states.addAll(Arrays.asList(startState, stopState));
        Set<S> nonStartStopStates = sequences.stream()
                .flatMap(Collection::stream)
                .filter(s -> !s.equals(startState) && !s.equals(stopState))
                .collect(Collectors.toSet());
        states.addAll(nonStartStopStates);
        List<Pair<S, S>> transitionPairs = sequences.stream()
                .map(seq -> ensureStartStopPadded(seq, startState, stopState))
                .flatMap(seq -> transitions(seq).stream())
                .distinct()
                .collect(Collectors.toList());
        return new StateSpace<>(states, transitionPairs);
    }

    private final static String DATA_VERSION = "1.0";

    @SneakyThrows
    public static StateSpace<String> load(DataInputStream dis) {
        IOUtils.ensureVersionMatch(dis, DATA_VERSION);
        List<String> states = IOUtils.loadList(dis);
        int numTransitions = dis.readInt();
        List<Pair<String, String>> transitions = new ArrayList<>();
        for (int idx = 0; idx < numTransitions; idx++) {
            int from = dis.readInt();
            int to = dis.readInt();
            transitions.add(Tuples.pair(states.get(from), states.get(to)));
        }
        return new StateSpace<>(states, transitions);
    }

    public void save(DataOutputStream dos) throws IOException {
        dos.writeUTF(DATA_VERSION);
        IOUtils.saveList(dos, states.stream().map(Object::toString).collect(Collectors.toList()));
        dos.writeInt(transitions().size());
        for (Transition transition : transitions) {
            dos.writeInt(transition.fromState);
            dos.writeInt(transition.toState);
        }
    }
}
