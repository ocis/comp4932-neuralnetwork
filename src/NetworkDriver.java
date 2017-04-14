import java.util.ArrayList;
import java.util.List;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;


public class NetworkDriver {
    
    public static DenseDoubleMatrix2D imageConvert(int[][] image){
        double[][] normalizedValues = new double[image.length * image[0].length][1];
        for(int a = 0; a < image.length; a++){
            for(int b = 0; b < image[a].length; b++){
                normalizedValues[a * image.length + b][0] = image[a][b] / 256.0;
            }
        }
        return new DenseDoubleMatrix2D(normalizedValues);
    }
    
    public static DenseDoubleMatrix2D labelConvert(int size, int value){
        double[][] label = new double[size][1];
        label[value][0] = 1;
        return new DenseDoubleMatrix2D(label);
    }
    
    public static void main(String[] args) {
        List<int[][]> trainImages = MnistReader.getImages("train-images.idx3-ubyte");
        int[] trainLabels = MnistReader.getLabels("train-labels.idx1-ubyte");
 
        List<int[][]> testImages = MnistReader.getImages("t10k-images.idx3-ubyte");
        int[] testLabels = MnistReader.getLabels("t10k-labels.idx1-ubyte");
        
        ArrayList<TestData> trainingData = new ArrayList<TestData>();
        ArrayList<TestData> testData = new ArrayList<TestData>();
        
        System.out.println("Converting training data");
        for(int i = 0; i < trainImages.size(); i++){
            trainingData.add(new TestData(imageConvert(trainImages.get(i)), labelConvert(10, trainLabels[i])));
        }
        
        System.out.println("Converting test data");
        for(int i = 0; i < testImages.size(); i++){
            testData.add(new TestData(imageConvert(testImages.get(i)), labelConvert(10, testLabels[i])));
        }
               
        int[] a = {784, 30, 10};
        Network n = new Network(a);
        System.out.println("Starting training");
        n.SGD(trainingData, 30, 10, 3.0, testData);

    }

}
