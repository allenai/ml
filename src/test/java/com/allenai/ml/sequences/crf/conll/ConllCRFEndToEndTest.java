package com.allenai.ml.sequences.crf.conll;

import lombok.val;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;

@Test
public class ConllCRFEndToEndTest {

    public String filePathOfResource(String path) {
        return this.getClass().getResource(path).getFile();
    }


    public void testEndToEnd() throws Exception {
        val trainOpts = new Trainer.Opts();
        trainOpts.featureKeepProb = 1.0;
        trainOpts.templateFile = filePathOfResource("/crf/template");
        trainOpts.trainPath = filePathOfResource("/crf/train.data");
        trainOpts.sigmaSquared = 1.0;
        val modelFile = File.createTempFile("crf","model");
        modelFile.deleteOnExit();
        trainOpts.modelPath = modelFile.getAbsolutePath();
        Trainer.trainAndSaveModel(trainOpts);
        val evalOpts = new Evaluator.Opts();
        evalOpts.modelPath = modelFile.getAbsolutePath();
        evalOpts.dataPath = filePathOfResource("/crf/test.data");
        val accPerfPair = Evaluator.evaluateModel(evalOpts);
        Assert.assertTrue(accPerfPair.getOne() > 0.90);
    }
}
