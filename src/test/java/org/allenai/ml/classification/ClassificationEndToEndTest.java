package org.allenai.ml.classification;

import com.gs.collections.api.tuple.Pair;
import com.gs.collections.impl.map.mutable.primitive.ObjectDoubleHashMap;
import com.gs.collections.impl.tuple.Tuples;
import lombok.SneakyThrows;
import org.allenai.ml.eval.Accuracy;
import org.allenai.ml.util.IOUtils;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

import static org.testng.Assert.assertTrue;

@Test
public class ClassificationEndToEndTest {

    private static Pair<Set<String>, String> getFeatureLabels(String svmLightFormatLine) {
        List<String> fields = Arrays.asList(svmLightFormatLine.split("\\s+"));
        String label = fields.get(fields.size()-1);
        Set<String> feats = fields.subList(0, fields.size() - 1)
            .stream()
            .map(s -> s.substring(0, s.indexOf(':')))
            .collect(Collectors.toSet());
        return Tuples.pair(feats, label);
    }

    @SneakyThrows
    public void testBinarySentimentClassification() {
        String labeledDataPath = this.getClass().getResource("/classification/dvd_sentiment.data").getFile();
        List<Pair<Set<String>, String>> labeledData = IOUtils.linesFromPath(labeledDataPath)
            .map(ClassificationEndToEndTest::getFeatureLabels)
            .collect(Collectors.toList());
        // Shuffle data since all negative examples came first, then positive ones
        Collections.shuffle(labeledData, new Random(0L));
        int split = (int) (0.8 * labeledData.size());
        List<Pair<Set<String>, String>> trainData = labeledData.subList(0, split);
        List<Pair<Set<String>, String>> testData = labeledData.subList(split, labeledData.size());
        MaxEntModel.TrainOpts trainOpts = new MaxEntModel.TrainOpts();
        trainOpts.sigmaSq = 1.0;
        // Feature extract is bag of words
        FeatureExtractor<Set<String>, String> featureExtractor = bag -> {
            ObjectDoubleHashMap<String> fv = new ObjectDoubleHashMap<>(bag.size());
            bag.forEach(x -> fv.put(x, 1.0));
            return fv;
        };
        MaxEntModel<String, Set<String>, String> maxEntModel = MaxEntModel.train(trainData, featureExtractor, trainOpts);
        File modelFile = File.createTempFile("classification","model");
        modelFile.deleteOnExit();
        DataOutputStream dos = new DataOutputStream(new FileOutputStream(modelFile));
        maxEntModel.save(dos);
        DataInputStream dis = new DataInputStream(new FileInputStream(modelFile));
        maxEntModel = MaxEntModel.load(dis, featureExtractor);
        Accuracy acc = new Accuracy();
        for (Pair<Set<String>, String> pair : testData) {
            String guess = maxEntModel.bestGuess(pair.getOne());
            acc.update(guess.equals(pair.getTwo()));
        }
        assertTrue(acc.accuracy() > 0.8);
    }
}
