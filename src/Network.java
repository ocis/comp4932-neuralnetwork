import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import cern.colt.function.DoubleDoubleFunction;
import cern.colt.function.DoubleFunction;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Mult;


public class Network {
    private int[] neurons;
    private int layers;
    private ArrayList<DoubleMatrix2D> biases;
    private ArrayList<DoubleMatrix2D> weights;
    private static Random r = new Random();
    private static Algebra alg = new Algebra();
    
    public Network(int[] neurons){
        this.neurons = Arrays.copyOf(neurons, neurons.length);
        layers = neurons.length;
        biases = new ArrayList<DoubleMatrix2D>();
        weights = new ArrayList<DoubleMatrix2D>();
        
        for(int i = 1; i < neurons.length; i++){
            DoubleMatrix2D b = new DenseDoubleMatrix2D(neurons[i],1);
            b.assign(randomGausian);
            biases.add(b);
        }
      
        for(int i = 0; i < neurons.length - 1; i++){
            DoubleMatrix2D w = new DenseDoubleMatrix2D(neurons[i], neurons[i + 1]);
            w.assign(randomGausian);
            weights.add(w);
        }
        
    }
    
    public static DoubleFunction randomGausian = new DoubleFunction(){
        public double apply(double a) {
            return r.nextGaussian();
        }
    };
    
    public static DoubleFunction sigmoid = new DoubleFunction(){
        public double apply(double a){
            return 1.0/(1.0 + Math.exp(-a));
        }
    };
    
    public static DoubleFunction sigmoidPrime = new DoubleFunction(){
        public double apply(double a){
            return 1.0/(1.0 + Math.exp(-a)) * (1 - 1.0/(1.0 + Math.exp(-a)));
        }
    };
    
    public static DoubleDoubleFunction add = new DoubleDoubleFunction(){
        public double apply(double a, double b){
            return a + b;
        }
    };
    
    public static DoubleDoubleFunction minus = new DoubleDoubleFunction(){
        public double apply(double a, double b){
            return a - b;
        }
    };
    
    public static DoubleDoubleFunction mult = new DoubleDoubleFunction(){
        public double apply(double a, double b){
            return a * b;
        }
    };
    
    
    public DoubleMatrix2D feedforward(DoubleMatrix2D a){
        for(int i = 0; i < biases.size(); i++){
            a = alg.mult(weights.get(i), a).assign(biases.get(i), add).assign(sigmoid);
        }
        return a;
    }
    
    public void SGD(ArrayList<TestData> trainingData, int epochs, int miniBatchSize, double eta, ArrayList<TestData> inputData){
        int numTests = 0;
        int numTrainingData;
        
        if(inputData != null){
            numTests = inputData.size();
        }
        numTrainingData = trainingData.size();
        
        for(int j = 0; j < epochs; j++){
            Collections.shuffle(trainingData);
            for(int k = 0; k < numTrainingData; k += miniBatchSize){
                ArrayList<TestData> miniBatch = new ArrayList<TestData>(trainingData.subList(k, k + miniBatchSize));
                updateMiniBatch(miniBatch, eta);
            }
            if(inputData != null){
                System.out.println("Epoch " + j + ": " + evaluate(inputData) + " / " + numTests);
            } else{
                System.out.println("Epoch " + j + " complete.");
            }
        }
    }
    
    public void updateMiniBatch(ArrayList<TestData> miniBatch, double eta){
        ArrayList<DoubleMatrix2D> nabla_b = new ArrayList<DoubleMatrix2D>();
        ArrayList<DoubleMatrix2D> nabla_w = new ArrayList<DoubleMatrix2D>();
        
        for(int i = 0; i < biases.size(); i++){
            nabla_b.add(biases.get(i).like());
        }
        for(int i = 0; i < weights.size(); i++){
            nabla_w.add(weights.get(i).like());
        }
        
        for(TestData data : miniBatch){
            MatrixPair dNablas = backprop(data.getData(), data.getResult());
            
            for(int i = 0; i < nabla_b.size(); i++){
                nabla_b.get(i).assign(dNablas.getNabla_B().get(i), add);
            }
            for(int i = 0; i < nabla_w.size(); i++){
                nabla_w.get(i).assign(dNablas.getNabla_W().get(i), add);
            }
            for(int i = 0; i < weights.size(); i++){
                weights.get(i).assign(nabla_w.get(i).assign(Mult.mult(eta/miniBatch.size())), minus);
            }
            for(int i = 0; i < biases.size(); i++){
                biases.get(i).assign(nabla_b.get(i).assign(Mult.mult(eta/miniBatch.size())), minus);
            }
        }
    }
    
    private MatrixPair backprop(DoubleMatrix2D data, DoubleMatrix2D result) {
        ArrayList<DoubleMatrix2D> nabla_b = new ArrayList<DoubleMatrix2D>();
        ArrayList<DoubleMatrix2D> nabla_w = new ArrayList<DoubleMatrix2D>();
        ArrayList<DoubleMatrix2D> activations = new ArrayList<DoubleMatrix2D>();
        ArrayList<DoubleMatrix2D> zs = new ArrayList<DoubleMatrix2D>();
        
        for(int i = 0; i < biases.size(); i++){
            nabla_b.add(biases.get(i).like());
        }
        for(int i = 0; i < weights.size(); i++){
            nabla_w.add(weights.get(i).like());
        }
        
        //feed forward
        DoubleMatrix2D activation = data.copy();
        activations.add(activation.copy());
        
        for(int i = 0; i < biases.size(); i++){
            System.out.println("size of biases: " + biases.size());
            System.out.println("bias size: " + biases.get(i).size() + ", weights size: " + weights.get(i).size());
            DoubleMatrix2D z = alg.mult(biases.get(i), activation).assign(weights.get(i), add);
            zs.add(z.copy());
            z.assign(sigmoid);
            activation = z.copy();
            activations.add(activation.copy());
        }
        
        //backward pass
        DoubleMatrix2D lastActivation = activations.get(activations.size() - 1).copy();
        DoubleMatrix2D lastZ = zs.get(zs.size() - 1).copy();
        DoubleMatrix2D resultCopy = result.copy();
        
        DoubleMatrix2D delta = lastActivation.assign(resultCopy, minus).assign(lastZ.assign(sigmoidPrime), mult);
        nabla_b.set(nabla_b.size() - 1, delta.copy());
        nabla_w.set(nabla_w.size() - 1, alg.mult(delta, alg.transpose(activations.get(activations.size() - 2))));
        
        for(int i = 2; i < layers; i++){
            DoubleMatrix2D sp = zs.get(zs.size() - 1).copy();
            sp.assign(sigmoidPrime);
            delta = alg.mult(alg.transpose(weights.get(weights.size() - i + 1)), delta).assign(sp, mult);
            nabla_b.set(nabla_b.size() - i, delta.copy());
            nabla_w.set(nabla_w.size() - i, alg.mult(delta, alg.transpose(activations.get(activations.size() - i - 1))));
        }
        return new MatrixPair(nabla_b, nabla_w);
    }

    public int evaluate(ArrayList<TestData> inputData){
        int count = 0;
        for(int i = 0; i < inputData.size(); i++){
            int x = findMaxIndex(feedforward(inputData.get(i).getData()).toArray());
            int y = findMaxIndex(inputData.get(i).getResult().toArray());
            if(x == y){
                count++;
            }
        }       
        return count;
    }
    
    public int findMaxIndex(double[][] matrix){
        int index = 0;
        double max = 0;
        for(int i = 0; i < matrix.length; i++){
            if(matrix[i][0] > max){
                index = i;
                max = matrix[i][0];
            }
        }
        return index;
    }
}
