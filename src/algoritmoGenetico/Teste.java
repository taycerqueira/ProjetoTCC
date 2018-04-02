package algoritmoGenetico;

import java.util.Random;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Teste {

	public static void main(String[] args) throws Exception {
		
		String databaseName = "basefilmes_53atributos";
		
		DataSource source = new DataSource (databaseName + ".arff");
		Instances data = source.getDataSet();
		data.setClass(data.attribute("polarity"));
		
		int seed = 2;          // the seed for randomizing the data
		int folds = 5;
		
		Random rand = new Random(seed);   // create seeded number generator
		Instances randData = new Instances(data);   // create copy of original data
		randData.randomize(rand);         // randomize data with number generator
		
		//randData.stratify(folds);
		
		
		 /*for (int n = 0; n < folds; n++) {
			 
			Instances train = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);
			
			System.out.println("treinamento: " + train.size());
			System.out.println("teste: " + test.size());
			
		}*/
		
		Instances train = randData.trainCV(folds, 0);
		Instances test = randData.testCV(folds, 0);
		
		System.out.println("treinamento: " + train.size());
		System.out.println("teste: " + test.size());

	}

}
