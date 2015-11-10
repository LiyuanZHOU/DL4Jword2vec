import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;

/**
 * @author Liyuan ZHOU (liyuan.zhou@nicta.com.au)
 */

public class DL4JWordEmbedLearner {
	static String corpusDir = "/media/data3tb2/socialWatchData/wordEmbeddings/trainData/reddit-crawled-drugs-subrredits_plainText.txt";
	static String embeddingFilePath = "/media/data3tb1/wordEmbedings/word2vec/NegSamp_n10/vectors_w5_v25";
	static String resultFile = "/media/data3tb2/socialWatchData/wordEmbeddings/model/reddit-crawled-drugs-subrredits_plainText.txt";

	static int wordDim = 25;
	static int contextSize = 5;
	static int numNegSamples = 5;
	static int batchSize = 1000;
	static int iterations = 30;
	static int minWordFrequency = 2;
	static float initLearningRate = 0.1f;
	static float minLearningRate = 1e-2f;

	static int seed = 1000;
	static LineSentenceIterator iter = new LineSentenceIterator(
			new File(
					"/media/data3tb2/socialWatchData/wordEmbeddings/trainData/test.txt"));
	static EndingPreProcessor preProcessor = new EndingPreProcessor();
	static DefaultTokenizerFactory tokenizer = new DefaultTokenizerFactory();
	static WordVectors wordVectors = new WordVectorsImpl();
	static Word2Vec vec = new Word2Vec();

	public static void loadData() throws FileNotFoundException {
		System.out.println("Load training data....");
		iter = new LineSentenceIterator(new File(corpusDir));
		System.out.println("Load pre-trained data....");
		wordVectors = WordVectorSerializer.loadTxtVectors(new File(
				embeddingFilePath));

		iter.setPreProcessor(new SentencePreProcessor() {
			private static final long serialVersionUID = 1L;

			@Override
			public String preProcess(String sentence) {
				return sentence.toLowerCase();
			}
		});
	}

	public static void tokenizor() {
		System.out.println("Tokenize data....");
		tokenizer.setTokenPreProcessor(new TokenPreProcess() {
			@Override
			public String preProcess(String token) {
				token = token.toLowerCase();
				String base = preProcessor.preProcess(token);
				base = base.replaceAll("\\d", "d");
				if (base.endsWith("ly") || base.endsWith("ing"))
					System.out.println();
				return base;
			}
		});
	}

	public static void trainModel() throws IOException {
		System.out.println("Build model....");
		vec = new Word2Vec.Builder()
				.batchSize(batchSize)
				// # words per minibatch.
				.sampling(1e-5)
				// negative sampling. drops words out
				.minWordFrequency(minWordFrequency)
				//
				.useAdaGrad(true)
				//
				.layerSize(wordDim)
				// word feature vector size
				.iterations(iterations)
				// # iterations to train
				.learningRate(initLearningRate)
				//
				.minLearningRate(minLearningRate)
				// learning rate decays wrt # words. floor learning
				.negativeSample(numNegSamples)
				// sample size 10 words
				.iterate(iter)
				//
				.tokenizerFactory(tokenizer)
				.lookupTable(wordVectors.lookupTable())
				.vocabCache(wordVectors.vocab()).saveVocab(true).seed(seed)
				.windowSize(contextSize).build();

		vec.fit();
	}

	public static void saveWord2Vec() throws IOException {
		System.out.println("Save vectors....");
		File resultf = new File(resultFile);
		if(!resultf.exists()) {
			resultf.createNewFile();
			resultFile = resultf.getAbsolutePath();
		}
		WordVectorSerializer.writeWordVectors(vec, resultFile);
	}

	public static void main(String[] args) throws IOException {
		
		if(args.length() >0){
			corpusDir = args[0];
			embeddingFilePath = args[1];
			resultFile = args[2];
			
		}
		
		loadData();
		tokenizor();
		trainModel();

		saveWord2Vec();
	}
}
