package com.allenai.ml.sequences;

import com.gs.collections.impl.factory.Lists;
import com.gs.collections.impl.factory.Sets;
import org.testng.annotations.Test;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

import static org.testng.Assert.*;

@Test
public class StateSpaceTest {

    public void testBuildFromSequences() throws Exception {
        String START = "<s>";
        String STOP = "</s>";
        String S1 = "s1";
        String S2 = "s2";
        String S3 = "s3";
        // S1 -> S3
        // S1 -> S2 -> S3
        StateSpace<String> stateSpace = StateSpace.buildFromSequences(
                Lists.fixedSize.of(Lists.fixedSize.of(S1, S2), Lists.fixedSize.of(S1, S2, S3)),
                START, STOP);
        // Test States
        assertEquals(stateSpace.startState(), START);
        assertEquals(stateSpace.stopState(), STOP);
        assertEquals(new HashSet<>(stateSpace.states()), new HashSet<>(Lists.fixedSize.of(START, STOP, S1, S2, S3)));
        // Test Transitions
        Set<String> fromStartStates = stateSpace.transitionFrom(stateSpace.stateIndex(START)).stream()
                .map(t -> stateSpace.transition(t.selfIndex).getTwo())
                .collect(Collectors.toSet());
        assertEquals(fromStartStates, Sets.immutable.of(S1));
        Set<String> s1Transitions = stateSpace.transitionFrom(stateSpace.stateIndex(S1))
                .stream()
                .map(t -> stateSpace.transition(t.selfIndex).getTwo())
                .collect(Collectors.toSet());
        assertEquals(s1Transitions, Sets.mutable.of(S2));
        Set<String> toStopTransitions = stateSpace.transitionsTo(stateSpace.stateIndex(STOP))
                .stream()
                .map(t -> stateSpace.transition(t.selfIndex).getOne())
                .collect(Collectors.toSet());
        assertEquals(toStopTransitions, Sets.mutable.of(S2, S3));
    }
}