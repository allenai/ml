package org.allenai.ml.sequences.crf.conll;

import org.allenai.ml.sequences.crf.CRFPredicateExtractor;
import com.gs.collections.api.map.primitive.ObjectDoubleMap;
import com.gs.collections.impl.tuple.primitive.PrimitiveTuples;
import lombok.val;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.testng.Assert.*;

@Test
public class ConllFormatTest {

    public void testFeatureTemplateParsing() throws Exception {
        String inputTemplate = "U00:%x[-2,0]/%x[2,0]";
        val f = ConllFormat.FeatureTemplate.fromLineSpec(inputTemplate);
        assertEquals(f.type, ConllFormat.FeatureTemplate.Type.NODE);
        assertEquals(f.prefix, "U00");
        assertEquals(f.rowCols, Arrays.asList(PrimitiveTuples.pair(-2,0), PrimitiveTuples.pair(2,0)));

        val bigram = ConllFormat.FeatureTemplate.fromLineSpec("B");
        assertEquals(bigram.toString(), "B");
    }

    ConllFormat.FeatureTemplate nodeTemplate = new ConllFormat.FeatureTemplate("Uprefix", Arrays.asList(
        PrimitiveTuples.pair(0,0),
        PrimitiveTuples.pair(1,0),
        PrimitiveTuples.pair(1,1),
        PrimitiveTuples.pair(-1,0)));

    ConllFormat.FeatureTemplate biasEdgeTemplate = new ConllFormat.FeatureTemplate("B", Arrays.asList());

    public void testFeatureTemplateValue() throws Exception {
        List<ConllFormat.Row> rows = Arrays.asList(
            new ConllFormat.Row(Arrays.asList("f1","f2","f3")) ,
            new ConllFormat.Row(Arrays.asList("f4","f5","f6")),
            new ConllFormat.Row(Arrays.asList("f7","f8","f9")));
        String featValue = nodeTemplate.value(rows, 1);
        assertEquals(featValue, "Uprefix:f4/f7/f8/f1");
        nodeTemplate.toString();
    }

    public void testPredicateExtractor() throws Exception {
        CRFPredicateExtractor<ConllFormat.Row, String> predExtractor = ConllFormat.predicatesFromTemplate(
            Stream.of("U00:%x[0,0]", "U10:%x[1,0]", "UBoth:%x[0,0]/%x[1,0]", "B"));

        List<ConllFormat.Row> rows = Arrays.asList(
            new ConllFormat.Row(Arrays.asList("<s>")),
            new ConllFormat.Row(Arrays.asList("f1")),
            new ConllFormat.Row(Arrays.asList("f2")),
            new ConllFormat.Row(Arrays.asList("f3")),
            new ConllFormat.Row(Arrays.asList("</s>")));
        List<ObjectDoubleMap<String>> nodePredMaps = predExtractor.nodePredicates(rows);
        List<ObjectDoubleMap<String>> edgePredMaps = predExtractor.edgePredicates(rows);
        assertEquals(new HashSet<>(nodePredMaps.get(2).keySet()),
            Stream.of("U00:f2", "U10:f3", "UBoth:f2/f3").collect(Collectors.toSet()));
        assertTrue(edgePredMaps.get(0).keySet().contains("B"));
    }
}