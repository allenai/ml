package org.allenai.ml.sequences.crf;

import org.allenai.ml.sequences.StateSpace;
import com.gs.collections.api.map.primitive.ObjectDoubleMap;
import com.gs.collections.impl.tuple.Tuples;
import lombok.val;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.testng.Assert.*;

@Test
public class CRFFeatureEncoderTest {



    @Test
    public void testFeatureEncoder() throws Exception {
        val predExtractor = new CRFPredicateExtractor<String, String>() {
            @Override
            public List<ObjectDoubleMap<String>> nodePredicates(List<String> elems) {
                return Arrays.asList(
                    CRFTestUtils.make("<s>", 1.0),
                    CRFTestUtils.make("a", 1.0),
                    CRFTestUtils.make("b", 1.0),
                    CRFTestUtils.make("c", 1.0),
                    CRFTestUtils.make("</s>", 1.0));
            }

            @Override
            public List<ObjectDoubleMap<String>> edgePredicates(List<String> elems) {
                return Arrays.asList(
                    CRFTestUtils.make("#bias", 1.0),
                    CRFTestUtils.make("#bias", 1.0),
                    CRFTestUtils.make("#bias", 1.0),
                    CRFTestUtils.make("#bias", 1.0));
            }
        };
        List<List<String>> observations = Arrays.asList(
            Arrays.asList("o1","o2","o3"),
            Arrays.asList("o2","o1","o3"));
        Set<String> states = Stream.of("s1", "s2", "s3")
            .collect(Collectors.toSet());
        StateSpace<String> stateSpace = StateSpace.buildFullStateSpace(states, "<s>","</s>");
        CRFFeatureEncoder.BuildOpts buildOpts = CRFFeatureEncoder.BuildOpts.builder()
            .numThreads(1)
            .probabilityToAccept(1.0)
            .randSeed(0L)
            .build();
        val featEncoder = CRFFeatureEncoder.build(observations, predExtractor, stateSpace, buildOpts);
        assertEquals(featEncoder.nodeFeatures.size(), 5);
        assertEquals(featEncoder.edgeFeatures.size(), 1);
        CRFIndexedExample indexedExample = featEncoder.indexLabeledExample(Arrays.asList(
                Tuples.pair("<s>", "<s>"),
                Tuples.pair("o1", "s1"),
                Tuples.pair("o2", "s2"),
                Tuples.pair("o3", "s3"),
                Tuples.pair("</s>", "</s>")));
        assertTrue( indexedExample.isLabeled() );
        int[] expectedLabelIndices = Stream.of("<s>", "s1", "s2", "s3", "</s>")
            .mapToInt(stateSpace::stateIndex)
            .toArray();
        assertEquals( indexedExample.getGoldLabels(), expectedLabelIndices );

        // Hack to cover the generated toString method
        assertNotNull( CRFFeatureEncoder.BuildOpts.builder().toString() );
    }
}