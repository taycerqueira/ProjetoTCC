/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package algoritmoGenetico;

/**
 * This class was created to implement the compareTo method. So, I could make
 * order an list of vector with the objectives of this class.
 * @author Matheus Giovanni Pires
 * @email mgpires@ecomp.uefs.br
 * @data 2015/08/13
 */
public class ObjectivesAccuracyComplexity implements Comparable<ObjectivesAccuracyComplexity> {
    private double accuracy;
    private double complexity;
    private int index; // is the index of solution in the populatoin (SolutionSet)

    public ObjectivesAccuracyComplexity(double accuracy, double complexity, int index) {
        this.accuracy = accuracy;
        this.complexity = complexity;
        this.index = index;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(double accuracy) {
        this.accuracy = accuracy;
    }

    public double getComplexity() {
        return complexity;
    }

    public void setComplexity(double complexity) {
        this.complexity = complexity;
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }
    

    @Override
    public int compareTo(ObjectivesAccuracyComplexity arg) {
                
        if (this.accuracy < arg.getAccuracy())
            return -1;        
        else if (this.accuracy > arg.getAccuracy()) 
            return 1;
        else
            // ser forem iguais, retorna zero
            return 0;
    }
       
} // end ObjectivesAccuracyComplexity class
