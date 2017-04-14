package sistemaFuzzy;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import net.sourceforge.jFuzzyLogic.FunctionBlock;
import net.sourceforge.jFuzzyLogic.membership.MembershipFunction;
import net.sourceforge.jFuzzyLogic.rule.LinguisticTerm;
import net.sourceforge.jFuzzyLogic.rule.Rule;
import net.sourceforge.jFuzzyLogic.rule.RuleBlock;
import net.sourceforge.jFuzzyLogic.rule.RuleExpression;
import net.sourceforge.jFuzzyLogic.rule.RuleTerm;
import net.sourceforge.jFuzzyLogic.rule.Variable;
import net.sourceforge.jFuzzyLogic.ruleAccumulationMethod.RuleAccumulationMethodMax;
import net.sourceforge.jFuzzyLogic.ruleActivationMethod.RuleActivationMethodMin;
import net.sourceforge.jFuzzyLogic.ruleConnectionMethod.RuleConnectionMethodAndMin;
import weka.core.Instance;
import weka.core.Instances;

public class WangMendel {
	
	private Instances instancias;
	private String classAttribute;
	
	public WangMendel(Instances instancias, String classAttribute) throws Exception{
		
		this.instancias = instancias;
		this.classAttribute = classAttribute;

	}
	
	public RuleBlock generateRuleBlock(String name, FunctionBlock functionBlock, HashMap<String, Variable> variaveis) throws Exception{
		
		RuleBlock ruleBlock = new RuleBlock(functionBlock);
		ruleBlock.setName(name);
		ruleBlock.setRuleActivationMethod(new RuleActivationMethodMin());
		ruleBlock.setRuleAccumulationMethod(new RuleAccumulationMethodMax());
		
		//Armazena o indice do atributo que corresponde a classe. Aqui considero que é sempre o último atributo.
		int indiceClasse = this.instancias.numAttributes() - 1; 
		
		int contInstancias = 0; 
		
		for (int k = 0; k < instancias.size(); k++ ) {
			
			contInstancias++;
			
			Instance instancia = instancias.get(k);

			Rule regra = new Rule("Rule " + contInstancias, ruleBlock);
			
			String classeInstancia = instancia.stringValue(indiceClasse);
			Variable output_variable = variaveis.get(classAttribute);
			
			regra.addConsequent(output_variable, classeInstancia, false);
			
			//Começa com 1 por ser um fator neutro na multiplicação
			double grauRegra = 1; 
			
			RuleExpression antecedents = null;
			RuleTerm fuzzyRuleTerm1 = null;
			ArrayList<LinguisticTerm> antecedentesConjuntosFuzzy = new ArrayList<LinguisticTerm>();
			
			//Pega cada atributo da instância
			for (int i = 0; i < indiceClasse; i++) {
				
				//Para cada atributo da instância, verifico o conjunto fuzzy de maior grau
				double maiorGrau = Double.NEGATIVE_INFINITY;
				LinguisticTerm conjuntoMaiorGrau = null;	
							
				String atributeName = instancia.attribute(i).name();					
				
				double valor = instancia.value(i);
				Variable variavel = variaveis.get(atributeName);
				
				List<LinguisticTerm> conjuntosFuzzy = variavel.linguisticTermsSorted();
				for (LinguisticTerm conjunto : conjuntosFuzzy) {

					MembershipFunction pertinencia = conjunto.getMembershipFunction();
					double grau = pertinencia.membership(valor);
					
					if(grau > maiorGrau){
						maiorGrau = grau;
						conjuntoMaiorGrau = conjunto;
					}
					
				}
				
				antecedentesConjuntosFuzzy.add(conjuntoMaiorGrau);
				
				grauRegra *= maiorGrau;
				
				if(i == 0){
					
					fuzzyRuleTerm1 = new RuleTerm(variavel, conjuntoMaiorGrau.getTermName(), false);
					
				}
				else if(i == 1){
						
					RuleTerm fuzzyRuleTerm2 = new RuleTerm(variavel, conjuntoMaiorGrau.getTermName(), false);
					antecedents = new RuleExpression(fuzzyRuleTerm1, fuzzyRuleTerm2, RuleConnectionMethodAndMin.get());
					
				}
				else{
					//System.out.println(antecedents);
					
					RuleTerm nextFuzzyRuleTerm = new RuleTerm(variavel, conjuntoMaiorGrau.getTermName(), false);
					antecedents = new RuleExpression(antecedents, nextFuzzyRuleTerm, RuleConnectionMethodAndMin.get());
					
				}
				
			}
				
			regra.setDegreeActivationWangMendel(grauRegra);
			regra.setAntecedentesConjuntosFuzzy(antecedentesConjuntosFuzzy);
			regra.setAntecedents(antecedents);		
			
			boolean adiciona_regra = true;
			
			List<Rule> regrasExistentes = ruleBlock.getRules();
			
			//Verificar se existe uma regra já adicionada com os mesmos antecedentes
			for (Rule r : regrasExistentes) {
				
				ArrayList<LinguisticTerm> antecedentes = r.getAntecedentesConjuntosFuzzy();
				
				int cont = 0;
				
				for (LinguisticTerm conjuntoRegraJaExistente : antecedentes) {
					
					for (LinguisticTerm conjuntoRegraAtual : antecedentesConjuntosFuzzy) {
						
						if(conjuntoRegraJaExistente.getTermName().equals(conjuntoRegraAtual.getTermName())){
							cont++;
						}
						
					}
					
				}
				
				if(cont == antecedentes.size()){ //regra em duplicidade. verificar qual pussui maior grau
					
					if(r.getDegreeActivationWangMendel() < regra.getDegreeActivationWangMendel()){
						
						//Remover a regra já existente e adicionar a nova regra
						ruleBlock.remove(r);
						break;
						
					}
					else{
						
						adiciona_regra = false;
						
					}
					
				}
		
			}
			
			if(adiciona_regra){
				ruleBlock.add(regra);
			}
						
		}
					
		return ruleBlock;
		
	}

}
