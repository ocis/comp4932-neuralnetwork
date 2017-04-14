import java.util.ArrayList;

import cern.colt.matrix.DoubleMatrix2D;

public class MatrixPair {
    private ArrayList<DoubleMatrix2D> dNabla_B;
    private ArrayList<DoubleMatrix2D> dNabla_W;
    
    public MatrixPair(ArrayList<DoubleMatrix2D> dNabla_B, ArrayList<DoubleMatrix2D> dNabla_W){
        this.dNabla_B = dNabla_B;
        this.dNabla_W = dNabla_W;
    }
    
    public ArrayList<DoubleMatrix2D> getNabla_B(){
        return dNabla_B;
    }
    public void setNabla_B(ArrayList<DoubleMatrix2D> dNabla_B){
        this.dNabla_B = dNabla_B;
    }
    public ArrayList<DoubleMatrix2D> getNabla_W(){
        return dNabla_W;
    }
    public void setNabla_W(ArrayList<DoubleMatrix2D> dNabla_W){
        this.dNabla_W = dNabla_W;
    }
}
