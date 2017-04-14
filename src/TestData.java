import java.util.ArrayList;

import cern.colt.matrix.DoubleMatrix2D;

public class TestData {
    private DoubleMatrix2D data;
    private DoubleMatrix2D result;
    
    public TestData(DoubleMatrix2D data, DoubleMatrix2D result){
        this.data = data;
        this.result = result;
    }
    
    public DoubleMatrix2D getData(){
        return data;
    }
    public void setData(DoubleMatrix2D data){
        this.data = data;
    }
    public DoubleMatrix2D getResult(){
        return result;
    }
    public void setResult(DoubleMatrix2D result){
        this.result = result;
    }
    
}
