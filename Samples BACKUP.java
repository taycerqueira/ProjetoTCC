package algoritmoGenetico;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Samples {
	
    // traSamples_ stores the training data
    // testSamples_ stores the test data    
    private Instances traSamples_, testSamples_;    
    private int numberOfVariables_, numberOfTestSamples_, numberOfTraSamples_;
    
    private int numberOfSamples_;
    
    // this vector indicates which samples are or not selected. true is equal selected,
    // and false is equal not selected
    private boolean[] selectedSamples_;
    
    private int numberOfSelectedSamples_;
    
    // typeDataSet_ is "classification" or "regression"
    private String typeDataSet_;
    
    // typeProcedure_ is "training" or "test"
    private String typeProcedure_;
    
    public Samples() {
        traSamples_              = null;
        testSamples_             = null;        
        selectedSamples_         = null;        
        numberOfTestSamples_     = 0;
        numberOfTraSamples_      = 0; 
        numberOfVariables_       = 0;
        numberOfSamples_         = 0; 
        typeDataSet_             = null;
        typeProcedure_           = null;
        numberOfSelectedSamples_ = 0;
    }
    
    public Instances getTestSamples() {
        return testSamples_;
    }

    public Instances getTraSamples() {
        return traSamples_;
    } 

    public int getNumberOfVariables() {
        return numberOfVariables_;
    }

    /**
     * @return The number of test samples
     */
    public int getNumberOfTestSamples() {
        return numberOfTestSamples_;
    }    

    /**
     * @return The number of training samples
     */
    public int getNumberOfTraSamples() {
        return numberOfTraSamples_;
    } 

    /**
     * @return The type of dataset. It can be "classification" problem or 
     * "regression" problem
     */
    public String getTypeDataSet() {
        return typeDataSet_;
    }

    /**
     * Set the type of dataset. It is "classification" problem or "regression"
     * problem
     * @param typeDataSet "classification" or "regression"
     */
    public void setTypeDataSet(String typeDataSet) {
        if (typeDataSet.equalsIgnoreCase("classification") || typeDataSet.equalsIgnoreCase("regression"))
            this.typeDataSet_ = typeDataSet;
        else {
            System.err.print("Samples class > setTypeDataSet_ method error: typeDataSet parameter: " + 
                typeDataSet + " invalid.");
            System.exit(-1);
        }
    }
    
    /**
     * @return The type of procedure. It can be "training" or "test"
     */
    public String getTypeProcedure() {
        return typeProcedure_;
    }
    
    /**
     * Set the type of current procedure. It is "training", "trainingKB" or "test"
     * "trainingKB" is used when only learningKB is executed
     * @param typeProcedure 
     */
    public void setTypeProcedure(String typeProcedure) {
        if (typeProcedure.equalsIgnoreCase("training") || typeProcedure.equalsIgnoreCase("test") ||
            typeProcedure.equalsIgnoreCase("trainingKB") || typeProcedure.equalsIgnoreCase("none"))
            this.typeProcedure_ = typeProcedure;
        else {
            System.err.print("Samples class > setTypeProcedure method error: typeProcedure parameter: " + 
                typeProcedure + " invalid.");
            System.exit(-1);            
        }
    }
       
    /**
     * This method set the index "idx" of boolean vector selectedSamples_ with
     * boolean value
     * @param idx Index of boolean vector selectedSamples_
     * @param value Boolean value
     */
    public void setSelectedSamples(int idx, boolean value) {
        selectedSamples_[idx] = value;
    }

    /**
     * This method returns a boolean vector that indicates which samples were or not selected. 
     * True is equal selected and false is equal not selected
     * @return The selectedSamples_ vector
     */
    public boolean[] getSelectedSamples() {
        return selectedSamples_;
    }    

    /**
     * @return The number of selected samples
     */
    public int getNumberOfSelectedSamples() {
        return numberOfSelectedSamples_;
    }

    /**
     * This method set the number of selected samples
     * @param numberOfSelectedSamples_ Stores the number of selected samples
     */    
    public void setNumberOfSelectedSamples(int numberOfSelectedSamples_) {
        this.numberOfSelectedSamples_ = numberOfSelectedSamples_;
    }
    
    public void loadSamples (String datasetName, double porcentagemTeste) throws Exception { 
    	
		DataSource source = new DataSource (datasetName + ".arff");
	    Instances data = source.getDataSet();
	    int numInstancias = data.numInstances();
	    
	    data.setClassIndex(data.numAttributes() - 1);
	    
	    int quantidadeTeste = (int) (numInstancias*porcentagemTeste);
	    
		//Separo as instâncias que irão ser utilizadas para teste e gravo em uma arquivo de texto
	    gerarInidicesTeste(quantidadeTeste, numInstancias);
	    
	    //Pego as instancias de teste a partir de um arquivo 
		int[] indicesInstanciasTeste = getIndicesTeste(quantidadeTeste);
		int[] indicesInstanciasTreinamento = getIndicesTreinamento(indicesInstanciasTeste, numInstancias);

        testSamples_ = getInstanciasTeste(new Instances(data), indicesInstanciasTreinamento);
        numberOfTestSamples_ = testSamples_.size();     
        
        traSamples_ = getInstanciasTeste(new Instances(data), indicesInstanciasTreinamento);
        numberOfTraSamples_ = traSamples_.size(); 
        
        numberOfVariables_ = data.numAttributes();
        numberOfSamples_ = numInstancias;
        
        /* this vector indicates which samples are or not selected. true is 
        equal selected, and false is equal not selected
        */
        selectedSamples_ = new boolean[numberOfTraSamples_];
        // initializing the vector with false
        for (int i = 0; i < numberOfTraSamples_; i++)
            selectedSamples_[i] = false;
        
    } //end loadSamples method
    
	private static int[] sortearInstanciasTeste(int quantidade, int numInstancias){
		
		int[] indices = new int[quantidade];
		
		List<Integer> numeros = new ArrayList<Integer>();
		
		for (int i = 0; i < numInstancias; i++) { 
			
		    numeros.add(i);
		    
		}
		
		//Embaralhamos os números:
		Collections.shuffle(numeros);
		
		//Adicionamos os números aleatórios no vetor
		for (int i = 0; i < quantidade; i++) {
			
			indices[i] = numeros.get(i);
			
		}
		
		return indices;
		
	}
    
	//Gera os indices de teste e coloca em um arquivo
	public static void gerarInidicesTeste(int quantidade, int numInstancias){
		
		int[] indicesTeste = sortearInstanciasTeste(quantidade, numInstancias);
		
		File arquivo = new File("indicesTeste.txt");
		
		try(FileWriter fw = new FileWriter(arquivo)){
			for (int indice : indicesTeste) {
				fw.write(indice + "\r\n");
			}
		    fw.flush();
		}catch(IOException ex){
		  ex.printStackTrace();
		}
		
	}
	
	private static boolean isParaTeste(int k, int[] indicesTeste){
		
		boolean resultado = false;
		
		for(int i = 0; i < indicesTeste.length; i++){
			
			if(indicesTeste[i] == k){
				
				resultado = true;
				break;
				
			}
			
		}
		
		return resultado;
	}
	
	//Lê o arquivo e retorna os indices de teste
	public static int[] getIndicesTeste(int quantidade){
		
		int[] indicesTeste = new int[quantidade];
		int i = 0;
		
		File arquivo = new File("indicesTeste.txt");
		try(InputStream in = new FileInputStream(arquivo) ){
		  Scanner scan = new Scanner(in);
		  while(scan.hasNext()){
		    String indice = scan.nextLine();
		    if(indice.length() > 0){
		    	indicesTeste[i] = Integer.parseInt(indice);
		    	i++;
		    }
		  }
		}catch(IOException ex){
		  ex.printStackTrace();
		}
		
		//System.out.println("Quantidade de instâncias de teste = " + indicesTeste.length);
		
		return indicesTeste;
		
	}
	
	public static int[] getIndicesTreinamento(int[] indicesInstanciasTeste, int numInstancias){
		
		int[] indicesTreinamento = new int[numInstancias - indicesInstanciasTeste.length];
		
		int cont = 0;
		
		for(int i = 0; i < numInstancias; i++){
			
			if(!isParaTeste(i, indicesInstanciasTeste)){
				
				indicesTreinamento[cont] = i;
				cont++;
				
			}
			
		}
		
		//System.out.println("Quantidade de instâncias de treinamento = " + indicesTreinamento.length);
		
		return indicesTreinamento;
		
	}
	
	public static Instances getInstanciasTreinamento(Instances instancias, int[] indicesTeste){
		
		ArrayList<Instance> instanciasTeste = new ArrayList<Instance>();
		
		for(int i = 0; i < indicesTeste.length; i++){
			instanciasTeste.add(instancias.get(i));			
		}
		
		instancias.removeAll(instanciasTeste);
			
		//System.out.println("Quantidade de instâncias de treinamento = " + instancias.numInstances());
		
		return instancias;
		
	}
	
	public static Instances getInstanciasTeste(Instances instancias, int[] indicesTreinamento){
		
		ArrayList<Instance> instanciasTreinamento = new ArrayList<Instance>();
		
		for(int i = 0; i < indicesTreinamento.length; i++){
			instanciasTreinamento.add(instancias.get(i));			
		}
		
		instancias.removeAll(instanciasTreinamento);
			
		//System.out.println("Quantidade de instâncias de teste = " + instancias.numInstances());
		
		return instancias;
		
	}

}
