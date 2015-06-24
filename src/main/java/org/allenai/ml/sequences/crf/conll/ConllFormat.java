package org.allenai.ml.sequences.crf.conll;

import lombok.extern.slf4j.Slf4j;
import org.allenai.ml.linalg.DenseVector;
import org.allenai.ml.linalg.Vector;
import org.allenai.ml.sequences.StateSpace;
import org.allenai.ml.sequences.crf.CRFFeatureEncoder;
import org.allenai.ml.sequences.crf.CRFModel;
import org.allenai.ml.sequences.crf.CRFPredicateExtractor;
import org.allenai.ml.sequences.crf.CRFWeightsEncoder;
import org.allenai.ml.util.IOUtils;
import org.allenai.ml.util.Indexer;
import com.gs.collections.api.list.ImmutableList;
import com.gs.collections.api.map.primitive.ObjectDoubleMap;
import com.gs.collections.api.tuple.Pair;
import com.gs.collections.api.tuple.primitive.IntIntPair;
import com.gs.collections.impl.factory.Lists;
import com.gs.collections.impl.map.mutable.primitive.ObjectDoubleHashMap;
import com.gs.collections.impl.tuple.Tuples;
import com.gs.collections.impl.tuple.primitive.PrimitiveTuples;
import lombok.RequiredArgsConstructor;
import lombok.val;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;

@Slf4j
public class ConllFormat {

    private static List<List<String>> chunkedLines(Stream<String> lines) {
        List<List<String>> chunks = new ArrayList<>();
        List<String> cur = new ArrayList<>();
        lines.forEach(line -> {
            if (line.trim().isEmpty()) {
                List<String> newTail = new ArrayList<String>(cur);
                chunks.add(newTail);
                cur.clear();
            } else {
                cur.add(line);
            }
        });
        return chunks;
    }

    public static List<Row> readDatum(List<String> lines, boolean labeled) {
        List<Row> feats = lines.stream().map(s -> {
            List<String> cols = Arrays.asList(s.split("\\t+"));
            if (labeled && cols.size() < 2) {
                throw new IllegalArgumentException("Labeled row doesn't appear to have at least two columns");
            }
            return labeled ?
                new Row(cols.subList(0, cols.size() - 1), cols.get(cols.size() - 1)) :
                new Row(cols);
        }).collect(toList());
        feats.add(0, new Row(Arrays.asList(startState), startState));
        feats.add(new Row(Arrays.asList(stopState), stopState));
        return feats;
    }

    public static List<List<Row>> readData(Stream<String> lines, boolean labeled) {
        List<List<Row>> result = new ArrayList<>();
        for (List<String> chunk : chunkedLines(lines)) {
           result.add(readDatum(chunk, labeled));
        }
        return result;
    }

    public static class Row {
        public final ImmutableList<String> features;
        private final String label;

        public Row(List<String> features) {
            this(features, null);
        }

        public Optional<String> getLabel() {
            return Optional.ofNullable(label);
        }

        public Row(List<String> features, String label) {
            this.features = Lists.immutable.ofAll(features);
            this.label = label;
        }

        public Pair<String, Row> asLabeledPair() {
            if (label == null) {
                throw new RuntimeException("Must be a labeled example");
            }
            return Tuples.pair(label, this);
        }
    }

    public final static String startState = "<s>";
    public final static String stopState = "</s>";

    public static class FeatureTemplate {
        public enum Type {
            NODE,
            EDGE
        }

        public final String prefix;
        public final ImmutableList<IntIntPair> rowCols;
        public final Type type;

        //private final List<String> parts = new ArrayList<>(100);
        //private final StringBuilder builder = new StringBuilder(3200);

        public FeatureTemplate(String prefix, List<IntIntPair> rowCols) {
            if (prefix.startsWith("U")) {
                type = Type.NODE;
            } else if (prefix.startsWith("B")) {
                type = Type.EDGE;
            } else {
                throw new IllegalArgumentException("FeatureTemplate prefix must begin with 'U' or 'B'");
            }
            this.prefix = prefix;
            this.rowCols = Lists.immutable.ofAll(rowCols);
        }

        // %x[ROW, COL]/ pattern
        private final static Pattern rowColPattern = Pattern.compile("\\%x\\[(-?\\d+),(\\d+)\\]");

        public static FeatureTemplate fromLineSpec(String line) {
            int colonIdx = line.indexOf(':');
            String prefix = colonIdx < 0 ? line : line.substring(0, colonIdx);
            String[] patterns = line.substring(colonIdx+1).split("/");
            List<IntIntPair> pairs = Stream.of(patterns)
                .map(rowColPattern::matcher)
                .filter(Matcher::matches)
                .map(m -> {
                    int row = Integer.parseInt(m.group(1));
                    int col = Integer.parseInt(m.group(2));
                    return PrimitiveTuples.pair(row, col);
                })
                .collect(toList());
            return new FeatureTemplate(prefix, pairs);
        }

        public String value(List<Row> input, int idx) {
            if (rowCols.isEmpty()) {
                return prefix;
            }
            int n = input.size();
            List<String> parts = new ArrayList<>(rowCols.size());
            for (IntIntPair rowCol : rowCols) {
                int rowIdx = idx + rowCol.getOne();
                if (rowIdx < 0 || rowIdx >= n) {
                    parts.add("@_X"  + rowIdx);
                    continue;
                }
                Row row = input.get(rowIdx);
                int colIdx = rowCol.getTwo();
                if (colIdx >= row.features.size()) {
                    parts.add("@_Y"  + rowIdx);
                    continue;
                }
                val feat = row.features.get(colIdx);
                parts.add(feat);
            }
            val sb = new StringBuilder();
            sb.append(prefix);
            sb.append(':');
            for (int i = 0; i < parts.size(); i++) {
                if (i > 0) {
                    sb.append('/');
                }
                sb.append(parts.get(i));
            }
            return sb.toString();
        }

        @Override
        public String toString() {
            if (rowCols.isEmpty()) {
                return prefix;
            }
            val b = new StringBuilder();
            b.append(prefix);
            b.append(':');
            String featurePatterns = rowCols.toList().stream()
                .map(rc -> String.format("%%x[%d,%d]", rc.getOne(), rc.getTwo()))
                .collect(Collectors.joining("/"));
            b.append(featurePatterns);
            return b.toString();
        }
    }

    @RequiredArgsConstructor
    static class ConllPredicateExtractor implements CRFPredicateExtractor<Row, String> {
        private final List<FeatureTemplate> nodeTemplates;
        private final List<FeatureTemplate> edgeTemplates;

        private static List<ObjectDoubleMap<String>> buildPredVals(List<FeatureTemplate> templates, List<Row> rows) {
            List<ObjectDoubleMap<String>> predVals = new ArrayList<>(rows.size());
            for (int idx = 0; idx  < rows.size(); idx++) {
                val m = new ObjectDoubleHashMap<String>(templates.size());
                assert rows.get(0).features.equals(Arrays.asList(startState));
                assert rows.get(rows.size()-1).features.equals(Arrays.asList(stopState));
                for (FeatureTemplate nodeTemplate : templates) {
                    String pred = nodeTemplate.value(rows, idx);
                    if (pred != null) {
                        m.put(pred, 1.0);
                    }
                }
                predVals.add(m);
            }
            return predVals;
        }

        @Override
        public List<ObjectDoubleMap<String>> nodePredicates(List<Row> elems) {
            val preds = buildPredVals(nodeTemplates, elems);
            // clear start/stop
            preds.set(0, ObjectDoubleHashMap.newMap());
            preds.set(preds.size() - 1, ObjectDoubleHashMap.newMap());
            return preds;
        }

        @Override
        public List<ObjectDoubleMap<String>> edgePredicates(List<Row> elems) {
            return buildPredVals(edgeTemplates, elems).subList(0, elems.size()-1);
        }
    }

    public static CRFPredicateExtractor<Row, String> predicatesFromTemplate(Stream<String> lines) {
        List<FeatureTemplate> templates = lines.filter(l -> l.startsWith("U") || l.startsWith("B"))
            .map(FeatureTemplate::fromLineSpec)
            .collect(toList());
        List<FeatureTemplate> nodeTemplates = templates.stream()
            .filter(t -> t.type == FeatureTemplate.Type.NODE)
            .collect(toList());
        List<FeatureTemplate> edgeTemplates = templates.stream()
            .filter(t -> t.type == FeatureTemplate.Type.EDGE)
            .collect(toList());
        return new ConllPredicateExtractor(nodeTemplates, edgeTemplates);
    }

    private final static String DATA_VERSION = "1.1";

    public static void saveModel(DataOutputStream dos,
                                 List<String> featureTemplateLines,
                                 CRFFeatureEncoder<String, Row, String> featureEncoder,
                                 Vector weights) throws IOException {
        // save feature templates, node feature indexer, edge feature indexer
        dos.writeUTF(DATA_VERSION);
        IOUtils.saveList(dos, featureTemplateLines);
        featureEncoder.stateSpace.save(dos);
        featureEncoder.nodeFeatures.save(dos);
        featureEncoder.edgeFeatures.save(dos);
        IOUtils.saveDoubles(dos, weights.toDoubles());
    }

    public static CRFModel<String, Row, String> loadModel(DataInputStream dis) throws IOException {
        IOUtils.ensureVersionMatch(dis, DATA_VERSION);
        val predExtractor = predicatesFromTemplate(IOUtils.loadList(dis).stream());
        val stateSpace = StateSpace.load(dis);
        Indexer<String> nodeFeatures = Indexer.load(dis);
        Indexer<String> edgeFeatures = Indexer.load(dis);
        val featureEncoder =
            new CRFFeatureEncoder<String, Row, String>(predExtractor, stateSpace, nodeFeatures, edgeFeatures);
        val weightEncoder = new CRFWeightsEncoder<String>(stateSpace, nodeFeatures.size(), edgeFeatures.size());
        Vector weights = DenseVector.of(IOUtils.loadDoubles(dis));
        Pair<Row, Row> startStopObservations = Tuples.pair(
            new Row(Arrays.asList(startState)),
            new Row(Arrays.asList(stopState)));
        return new CRFModel<String, Row, String>(featureEncoder, weightEncoder, weights);
    }

    public static void main(String[] args) {
        val f = FeatureTemplate.fromLineSpec("U00:%x[-2,0]/%x[2,0]");
        System.out.println(f);
    }
}
