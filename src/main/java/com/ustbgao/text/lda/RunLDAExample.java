package com.ustbgao.text.lda;

import java.io.IOException;
import java.net.URL;
import java.util.Map;

/**
 * Created by ustbgao on 16-3-10.
 */
public class RunLDAExample {
    public static void main(String [] args) throws IOException{

        URL url = Thread.currentThread().getContextClassLoader().getResource("");
        String filePath = url.toString().substring(url.toString().indexOf("/")+1, url.toString().lastIndexOf("/"));
        System.out.println(filePath);
        Corpus corpus = null;

        corpus = Corpus.load("C:\\Users\\ustbgao\\Topic-Model\\target\\classes\\mini");


        GibbsSample ldaGibbsSampler = new GibbsSample(corpus.getDocument(), corpus.getVocabularySize());

        ldaGibbsSampler.gibbs(10);

        double[][] phi = ldaGibbsSampler.getPhi();
        Map<String, Double>[] topicMap = LDAUtil.translate(phi, corpus.getVocabulary(), 10);
        LDAUtil.explain(topicMap);
        int[] document = Corpus.loadDocument("data/mini/军事_510.txt", corpus.getVocabulary());
        double[] tp = GibbsSample.inference(phi, document);
        Map<String, Double> topic = LDAUtil.translate(tp, phi, corpus.getVocabulary(), 10);
        LDAUtil.explain(topic);
   }
}
