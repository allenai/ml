package org.allenai.ml.sequences.crf;

import org.allenai.ml.linalg.SparseVector;
import org.allenai.ml.linalg.Vector;
import lombok.val;
import org.testng.annotations.Test;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.testng.Assert.*;

@Test
public class CRFIndexedExampleTest {

    public void testGoldLabels() {
        List<Vector> toyNodePreds = CRFTestUtils.toyNodePreds();
        List<Vector> toyEdgePreds = CRFTestUtils.toyEdgePreds();
        int[] goldLabels = new int[]{0, 1, 2};
        val example = new CRFIndexedExample(toyNodePreds, toyEdgePreds, goldLabels);
        assertTrue(example.isLabeled());
        assertEquals(example.getGoldLabels(), goldLabels);
    }

    public void testPredicateCompaction() {
        List<Vector> toyNodePreds = CRFTestUtils.toyNodePreds();
        List<Vector> toyEdgePreds = CRFTestUtils.toyEdgePreds();
        val example = new CRFIndexedExample(toyNodePreds, toyEdgePreds);
        List<Vector> extractedNodePreds = IntStream.range(0, 3)
            .mapToObj(idx -> SparseVector.make(10).addInPlace(example.getNodePredicateValues(idx)))
            .collect(Collectors.toList());
        List<Vector> extractedEdgePreds = IntStream.range(0, 2)
            .mapToObj(idx -> SparseVector.make(10).addInPlace(example.getEdgePredicateValues(idx)))
            .collect(Collectors.toList());
        assertFalse(example.isLabeled());
        assertEquals(example.getSequenceLength(), 3);
        // Verify we can recover the vectors using the iterators for each positon in the node/edges
        for (int idx = 0; idx < extractedNodePreds.size(); idx++) {
            assertTrue(extractedNodePreds.get(idx).closeTo(toyNodePreds.get(idx)));
        }
        for (int idx = 0; idx < extractedEdgePreds.size(); idx++) {
            assertTrue(extractedEdgePreds.get(idx).closeTo(toyEdgePreds.get(idx)));
        }
    }
}