package sistemaFuzzy;

import wangMendel.*;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Scanner;
import net.sourceforge.jFuzzyLogic.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.experiment.Stats;

public class Main {
	
	//static String nomeBase = "basefilmes";
	static String nomeBase = "basefilmes_53atributos";
	static DataSource source;
	static int quantConjuntosFuzzy = 3;
	static double porcentagemTeste = 0.2;
	
	public static void main(String[] args) throws Exception {
		
		source = new DataSource (nomeBase + ".arff");
	    Instances data = source.getDataSet();
	    //imprime informações associadas à base de dados
	    int numInstancias = data.numInstances();
	    System.out.println("* Quantidade de instâncias: " + numInstancias);  
	    System.out.println("* Quantidade de atributos: " + data.numAttributes());
	    System.out.println("* Quantidade de conjuntos fuzzy por atributo: " + quantConjuntosFuzzy);
		
	    int quantidade = (int) (numInstancias*porcentagemTeste);
	    System.out.println("Quantidade de instâncias de teste: " + quantidade);
	    
		//Separo as instâncias que irão ser utilizadas para teste
	    //gerarInidicesTeste(quantidade, numInstancias);
	    
	    //Pego as instancias de teste a partir de um arquivo 
		int[] indicesTeste = getIndicesTeste(quantidade);
		
		FIS fis = new FIS();
		
		//Gera as regras usando wang-mendel e gera o arquivo .fcl
		wangMendel(indicesTeste, fis, source, 3, "polarity");
		
		//Cria o sistema fuzzy a partir do arquivo fcl
		/*FIS fis = FIS.load((nomeBase+".fcl"), true);

		if (fis == null) {
			System.err.println("Can't load file: '" + nomeBase + "'");
			System.exit(1);
		}*/
		
		//Testa o sistema
		//testarSistema(indicesTeste, fis);
		
		//JFuzzyChart.get().chart(fb);

		// Set inputs
		//fb.setVariable("temperatura", 965);
		//fb.setVariable("volume", 11);

		// Evaluate
		//fb.evaluate();

		// Show output variable's chart
		//fb.getVariable("polarity").defuzzify();
		

        // Print ruleSet
        //System.out.println(fis);
		
		// Print ruleSet
		//System.out.println(fb);
		//System.out.println("Pressão: " + fb.getVariable("pressao").getValue());
		
		//fis.getVariable("polarity").chartDefuzzifier(true);
		
		 // Show each rule (and degree of support)
	    //for(Rule r : fis.getFunctionBlock("caldeira").getFuzzyRuleBlock("No1").getRules())
	      //System.out.println(r);
		
	}
	
	public static void wangMendel(int[] indicesTeste, FIS fis, DataSource dados, int quant_regioes, String output_name){
		
		long inicio = System.currentTimeMillis(); 
	    
		try {
		    
		    System.out.println("=> Executando algoritmo de Wang-Mendel. Aguarde...");
		    WangMendel wm = new WangMendel(source, indicesTeste);
		    //ArrayList<Regra> regras = wm.gerarRegras();
		    
		    FunctionBlock fb = wm.generateFunctionBlock(fis, "TESTE", dados, quant_regioes, output_name);
		    
		    System.out.println("Quantidade de regras geradas: " + fb.getFuzzyRuleBlock("ruleblock1").getRules().size());
		    
		    long fim  = System.currentTimeMillis();  
		    System.out.println("* Tempo de execução (min/seg): " + new SimpleDateFormat("mm:ss").format(new Date(fim - inicio)));

		    
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
	private static void testarSistema(int indicesTeste[], FIS fis) throws Exception{
		
		Instances instancias = source.getDataSet();
		
		System.out.println("=> Testando...");
		
		double TP = 0; //quantidade de positivos corretamente classificados
		double TN = 0; //quantidade de negativos corretamente classificados
		
		double FP = 0; //quantidade de negativos que foram classificados como positivos
		double FN = 0; //quantidade de positivos que foram classificados como negativos
		
		for (int k = 0; k < source.getDataSet().size(); k++ ) {
			//Se a instância for de teste...
			if(isParaTeste(k, indicesTeste)){
				
				//System.out.println("Número da instância: " + k);
				Instance instancia = instancias.get(k);
				//Seta as entradas
				//Considerando que o último atributo é sempre o atributo que corresponde a classe da instância, por isso usa-se o -1
				for (int i = 0; i < (instancias.numAttributes() - 1); i++) {
					
					if(instancias.attribute(i).isNumeric()){

						String nomeAtributo = instancias.attribute(i).name();
						double valor = instancia.value(i);
						//System.out.println(nomeAtributo + ": " + valor);
						
						fis.setVariable(nomeAtributo, valor);
					}	
				}

				// Evaluate
				fis.evaluate();
				
				/************************** FUZZY GERAL *****************************/
				int contPositive = 0;
				double sumPositive = 0;
				int contNegative = 0;
				double sumNegative = 0;
				
				int contRegrasAtivadas = 0;
				
			    for(net.sourceforge.jFuzzyLogic.rule.Rule r : fis.getFunctionBlock(null).getFuzzyRuleBlock("No1").getRules()){
			    	double grau = r.getDegreeOfSupport();
			    	String classe = r.getConsequents().getFirst().getTermName();
			    	if(grau > 0){ //Se a regra foi ativada
			    		contRegrasAtivadas++;
				    	//System.out.println("indice: "+ k +" | grau: " + grau + " | classe: " + classe);
				    	if(classe.equals("positive")){
				    		contPositive++;
				    		sumPositive += grau;
				    	}
				    	else if(classe.equals("negative")){
				    		contNegative++;
				    		sumNegative += grau;
				    	}
			    	}
			    }
			    
			    double mediaPositive = sumPositive/contPositive;
			    double mediaNegative = sumNegative/contNegative;
			    
			    //System.out.println("sum negative: " + sumNegative + " | contNegative: " + contNegative);
			    //System.out.println("Quantidade de regras ativadas: " + contRegrasAtivadas);
			    //System.out.println("Media positive: " + mediaPositive);
			    //System.out.println("Media negative: " + mediaNegative);
			    
			    String polarity = null;
			    if(mediaPositive >= mediaNegative){
			    	polarity = "positive";
			    }
			    else{
			    	polarity = "negative";
			    }
			    //System.out.println("Output: " + polarity);
			    String realPolarity = instancia.stringValue(instancias.numAttributes() - 1);
			    //System.out.println("Desired Output: " + realPolarity);
				if(polarity.equals("positive")){
					if(realPolarity.equals("positive")){
						TP++;
					}
					else if(realPolarity.equals("negative")){
						FP++;
					}
				}
				else if(polarity.equals("negative")){
					if(realPolarity.equals("negative")){
						TN++;
					}
					else if(realPolarity.equals("positive")){
						FN++;
					}
				}
				/**********************************************************************/
				
				//Variable tip = fis.getFunctionBlock(null).getVariable("polarity");
				//JFuzzyChart.get().chart(tip, tip.getDefuzzifier(), true);
				/************************** FUZZY CLÁSSICO *****************************/
				/*String realPolarity = instancia.stringValue(instancias.numAttributes() - 1);
				if(fis.getVariable("polarity").getValue() >= 0){
					//System.out.println("Polarity: " + fis.getVariable("polarity").getValue() + "(POSITIVE)");
					if(realPolarity.equals("positive")){
						TP++;
					}
					else if(realPolarity.equals("negative")){
						FP++;
					}
				}
				else if(fis.getVariable("polarity").getValue() < 0){
					//System.out.println("Polarity: " + fis.getVariable("polarity").getValue() + "(NEGATIVE)");
					if(realPolarity.equals("negative")){
						TN++;
					}
					else if(realPolarity.equals("positive")){
						FN++;
					}
				}*/
				/**********************************************************************/

				//System.out.println("Real Polarity: " + realPolarity);
				//JFuzzyChart.get().chart(fis.getFunctionBlock(null));
				//System.out.println(fis);
				//break;
					
			}
		}
		
		//System.out.println("Quantidade de instâncias testadas: " + indicesTeste.length);
		System.out.println("Acertos: " + (TP + TN));
		System.out.println("Erros: " + (FP + FN));
		
		double acuracia = (TP + TN)/indicesTeste.length;
		System.out.println("Acurácia: " + acuracia);
		
		double TPR = TP/(TP + FN); // taxa de verdadeiros positivos
		double TNR = TN/(TN + FP); // taxa de verdadeiros negativos
		System.out.println("Taxa de verdadeiros positivos: " + TPR);
		System.out.println("Taxa de verdadeiros negativos: " + TNR);
		
	}
	
	private static boolean isParaTeste(int k, int[] indicesTeste){
		
		boolean resultado = false;
		for(int i = 0; i < indicesTeste.length; i++){
			if(indicesTeste[i] == k){
				resultado =  true;
				break;
			}
		}
		return resultado;
	}
	
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
		
		return indicesTeste;
		
	}

}
