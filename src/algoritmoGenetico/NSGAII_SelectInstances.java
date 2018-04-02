package algoritmoGenetico;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.encodings.variable.Binary;
import jmetal.qualityIndicator.QualityIndicator;
import jmetal.util.Distance;
import jmetal.util.JMException;
import jmetal.util.Ranking;
import jmetal.util.comparators.CrowdingComparator;

public class NSGAII_SelectInstances extends Algorithm {
	
	/**
	 * Constructor
	 * @param problem Problem to solve
	 */
	public NSGAII_SelectInstances(Problem problem) {
	    super (problem) ;
	} 

	/**    
	 * Runs the NSGA-II algorithm.
	 * @return a <code>SolutionSet</code> that is a set of non dominated solutions
	 * as a result of the algorithm execution
	 * @throws JMException 
	 * @throws ClassNotFoundException 
	 */
	@Override
	public SolutionSet execute() throws JMException, ClassNotFoundException {
	  
	    int populationSize;
	    int maxEvaluations;
	    int evaluations;   
	    int requiredEvaluations; // Use in the example of use of the indicators object (see below)
	    //int generations;
	    QualityIndicator indicators; // QualityIndicator object

	    SolutionSet population;
	    SolutionSet offspringPopulation;
	    SolutionSet union;
	    
	    Operator mutationOperator;
	    Operator crossoverOperator;
	    Operator selectionOperator;

	    Distance distance = new Distance();

	    //Read the parameters
	    populationSize = ((Integer) getInputParameter("populationSize"));
	    maxEvaluations = ((Integer) getInputParameter("maxEvaluations"));
	    indicators = (QualityIndicator) getInputParameter("indicators");

	    //Initialize the variables
	    population = new SolutionSet(populationSize);
	    evaluations = 0;
	    //generations = 0;
	    requiredEvaluations = 0;

	    //Read the operators
	    mutationOperator = operators_.get("mutation");
	    crossoverOperator = operators_.get("crossover");
	    selectionOperator = operators_.get("selection");

	    // Create the initial solutionSet
	    Solution newSolution;
	    
	    /* The first chromossome (solution) enable all instances, i.e., all genes
	    are true! The problem this solution is that always it will be the best solution!
	    */    
	    /*newSolution = new Solution(problem_);
	    initializeWithOne(newSolution);
	    problem_.evaluate(newSolution);
	    population.add(newSolution);*/
	    
	    
	    for (int i = 0; i < populationSize; i++) {      
	        newSolution = new Solution(problem_);
	        problem_.evaluate(newSolution);      
	        //System.out.println("Objectives.: " + newSolution.getObjective(0) + " " + newSolution.getObjective(1));
	        //problem_.evaluateConstraints(newSolution);
	        evaluations++;
	        population.add(newSolution);              
	    } 
	    //generations++;
	    //System.out.print(evaluations + " " + generations);
	    
	    //System.out.println("");
	    //System.out.println("Printing the population...");  
	    //printPopulation(population);    
	    
	    //System.out.println("Length of parents...............: " + parents.length);
	    //System.out.println("Number of decision variables....: " + parents[0].getDecisionVariables().length);
	    //System.out.println("Number of bits p1 and p2........: " + parents[0].getNumberOfBits() + " " + parents[1].getNumberOfBits());
	    
	    //Solution[] offSpringCrossover;
	    //offSpringCrossover = (Solution[]) crossoverOperator.execute(parents);    
	    //printOffSpring(offSpringCrossover);
	    //areEquals(parents, offSpringCrossover);
	    
	    //System.out.println("");
	    //System.out.println("Printing the population...");  
	    //printPopulation(population);
	    
	    //Solution[] offSpringMutation = new Solution[2];
	    //offSpringMutation[0] = new Solution(parents[0]);
	    //offSpringMutation[1] = new Solution(parents[1]);     
	    //mutationOperator.execute(offSpringMutation[0]);
	    //mutationOperator.execute(offSpringMutation[1]);
	    //areEquals(parents, offSpringMutation);    

	    System.out.println("Executando seleção de instâncias...");
	    // Generations 
	    while (evaluations < maxEvaluations) {       
	        // Create the offSpring solutionSet      
	        offspringPopulation = new SolutionSet(populationSize);
	        Solution[] parents = new Solution[2];
	        
	        for (int i = 0; i < (populationSize / 2); i++) {
	            if (evaluations < maxEvaluations) {
	            	
	                //obtain parents
	                parents[0] = (Solution) selectionOperator.execute(population);
	                parents[1] = (Solution) selectionOperator.execute(population);
	                
	                Solution[] offSpring = (Solution[]) crossoverOperator.execute(parents);
	                mutationOperator.execute(offSpring[0]);
	                mutationOperator.execute(offSpring[1]);
	                
	                problem_.evaluate(offSpring[0]);                
	                problem_.evaluate(offSpring[1]);   
	                
	                offspringPopulation.add(offSpring[0]);
	                offspringPopulation.add(offSpring[1]);
	                
	                evaluations += 2;
	                
	            } // if                            
	        } // for
	        
	        //generations++;

	        // Create the solutionSet union of solutionSet and offSpring
	        union = ((SolutionSet) population).union(offspringPopulation);

	        // Ranking the union
	        Ranking ranking = new Ranking(union);

	        int remain = populationSize;
	        int index = 0;
	        SolutionSet front = null;
	        population.clear();

	        // Obtain the next front
	        front = ranking.getSubfront(index);

	        while ((remain > 0) && (remain >= front.size())) {
	            //Assign crowding distance to individuals
	            distance.crowdingDistanceAssignment(front, problem_.getNumberOfObjectives());
	            //Add the individuals of this front
	            for (int k = 0; k < front.size(); k++) {
	                population.add(front.get(k));
	            } // for

	            //Decrement remain
	            remain = remain - front.size();

	            //Obtain the next front
	            index++;
	            if (remain > 0) {
	                front = ranking.getSubfront(index);
	            } // if        
	        } // while

	        // Remain is less than front(index).size, insert only the best one
	        if (remain > 0) {  // front contains individuals to insert                        
	            distance.crowdingDistanceAssignment(front, problem_.getNumberOfObjectives());
	            front.sort(new CrowdingComparator());
	            for (int k = 0; k < remain; k++) {
	                population.add(front.get(k));
	            } // for

	            remain = 0;
	        } // if                               

	        // This piece of code shows how to use the indicator object into the code
	        // of NSGA-II. In particular, it finds the number of evaluations required
	        // by the algorithm to obtain a Pareto front with a hypervolume higher
	        // than the hypervolume of the true Pareto front.
	        if ((indicators != null) && (requiredEvaluations == 0)) {
	            double HV = indicators.getHypervolume(population);
	            if (HV >= (0.98 * indicators.getTrueParetoFrontHypervolume())) {
	                requiredEvaluations = evaluations;
	            } // if
	        } // if
	        
	        //System.out.print(evaluations + " " + generations);
	    } // while

	    // Return as output parameter the required evaluations
	    setOutputParameter("evaluations", requiredEvaluations);

	    // Return the first non-dominated front
	    Ranking ranking = new Ranking(population);
	    //ranking.getSubfront(0).printFeasibleFUN("FUN_NSGAII") ;

	    return ranking.getSubfront(0);
	    
	} // execute  
	    
	/**
	 * printOffSpring print the offspring
	 * @param offSpring is one vector Solution with 2 offspring
	*/ 
	private void printOffSpring(Solution[] offSpring) {    
	    
	    if (problem_.getSolutionType().getClass() == ArrayBinarySolutionType.class) {        
	    
	        Binary off1 = (Binary)offSpring[0].getDecisionVariables()[0];
	        Binary off2 = (Binary)offSpring[1].getDecisionVariables()[0];        

	        int size1 = offSpring[0].getNumberOfBits();
	        int size2 = offSpring[1].getNumberOfBits();

	        if (size1 != size2)
	            System.err.println("The number of bits of offspring are not equals!!!");        
	        else {
	            System.out.println("Printing offspring 1:");
	            for (int k = 0; k < size1; k++)
	                System.out.println(off1.getIth(k));

	            System.out.println(" ");

	            System.out.println("Printing offspring 2:");
	            for (int k = 0; k < size1; k++)
	                System.out.println(off2.getIth(k));
	        }
	    }
	    else {
	        System.out.println("NSGAII_SelectInstances class > printOffSpring method error: solution type " + 
	            problem_.getSolutionType().getClass() + " invalid");
	        System.exit(-1);        
	    }
	}    
	/**
	* Prints the population, which is one object from SolutionSet
	* @param pop
	* @throws jmetal.util.JMException 
	* @author Matheus Giovanni Pires              
	* @email  mgpires@ecomp.uefs.br
	* @data   2014/09/17    
	*/
	private void printPopulation(SolutionSet pop) throws JMException {       

	    if (problem_.getSolutionType().getClass() == ArrayBinarySolutionType.class) {            
	        int sizeOfPopulation = pop.getMaxSize(); 
	        int size;
	        for (int i = 0; i < sizeOfPopulation; i++) {   
	            System.out.println("Chromosome " + i);            
	            Binary bits = (Binary)pop.get(i).getDecisionVariables()[0];
	            size = bits.getNumberOfBits();                
	            for (int k = 0; k < size; k++)
	                System.out.println(bits.getIth(k));
	        }
	    }
	    else {
	        System.out.println("NSGAII_SelectInstances class > printPopulation method error: solution type " + 
	            problem_.getSolutionType().getClass() + " invalid");
	        System.exit(-1);
	    }
	} // end printPopulation

	/**
	 * This method compares all bits between parents and offspring, if one is not
	 * equal, one message will be printed 
	 * @param parents vector Solution with two parents
	 * @param offSpring vector Solution with two offsprings
	 */
	private void areEquals(Solution[] parents, Solution[] offSpring) {

	    Binary off1 = (Binary)offSpring[0].getDecisionVariables()[0];
	    Binary off2 = (Binary)offSpring[1].getDecisionVariables()[0];

	    Binary p1 = (Binary)parents[0].getDecisionVariables()[0];
	    Binary p2 = (Binary)parents[1].getDecisionVariables()[0];        

	    boolean flag = false;

	    if (offSpring[0].getNumberOfBits() != parents[0].getNumberOfBits() ||
	        offSpring[1].getNumberOfBits() != parents[1].getNumberOfBits()) {
	        System.err.println("Number of bits is incompatible");
	    }        
	    else {
	        int size = offSpring[0].getNumberOfBits();            

	        for (int k = 0; k < size; k++) {           

	            if (p1.getIth(k) != off1.getIth(k)) { 
	                System.out.println("Offspring 1 and father 1: bit " + k + " is diferent");
	                flag = true;
	            }

	            if (p2.getIth(k) != off2.getIth(k)) {
	                System.out.println("Offspring 2 and father 2: bit " + k + " is diferent");
	                flag = true;
	            }

	        } 
	    }
	    if (flag == false)
	        System.out.println("\nParents and offsprings are equals!");
	} // end areEquals method

	/**
	 * This method initialize the <solution> with true. It means that all instances
	 * are selected
	 * @param solution 
	 */
	private void initializeWithOne(Solution solution) {
	    Binary sol = (Binary)solution.getDecisionVariables()[0];
	    int bits = sol.getNumberOfBits();  

	    for (int i = 0; i < bits; i++)
	        sol.setIth(i, true);        
	}

}
