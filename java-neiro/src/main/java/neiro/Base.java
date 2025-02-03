package neiro;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.*;

public class Base {

    public static void main(String[] args) throws URISyntaxException, IOException {
        List<float[]> datasTrain = new ArrayList<>();
        List<float[]> targetsTrain = new ArrayList<>();
        //prepareData(targetsTrain, datasTrain, "mnist_train_100.csv");
        prepareData(targetsTrain, datasTrain, "mnist_train.csv");
        List<float[]> datasTest = new ArrayList<>();
        List<float[]> targetsTest = new ArrayList<>();
        //prepareData(targetsTest, datasTest, "mnist_test_10.csv");
        prepareData(targetsTest, datasTest, "mnist_test.csv");


        Layer inputLayer = Layer.createLayerInput(784, FunEnum.NONE);
        Neiro biasInput = inputLayer.addBias();
        Layer hiddenLayer = Layer.createLayerHidden(100, FunEnum.SIGMOID, inputLayer);
        Neiro biasHidden = hiddenLayer.addBias();
        Layer outputLayer = Layer.createLayerOutput(10, FunEnum.SIGMOID, hiddenLayer);
        Net net = new Net(List.of(inputLayer, hiddenLayer, outputLayer), List.of(biasInput, biasHidden));
        //Net net = new Net(List.of(inputLayer, hiddenLayer, outputLayer), null);
        net.trains(datasTrain, targetsTrain,0.7F, 0.1F, 3 , 0.01F);


    /*    Net net= Net.load("file_net_addBias_h1x101_e6_0.04705009625900136_0.8639");
        //пробежимся по всем связям и "разобьём те, которые имеют слишком маленький вес" и посмотри как упрощение нейросети повлияет на точность итогового результата
        int contDeletedLink=0;
        for(Layer layer: net.layers)
            for(Neiro neiro : layer.neiros)
                for(Link link : Set.copyOf(neiro.outputs))
                    if (Math.abs(link.weight)<0.01){
                        contDeletedLink++;
                        link.terminate();
                    }
        System.out.println("Link deleted: "+contDeletedLink);
        net.trains(datasTrain, targetsTrain,0.07F, 0.01F,3 , 0.02F);
        //epoch=0, mse=0.041800059847739664, duration(ms)=25014
        //eff=0.8774
     */

        //Тестирование
        Integer countError=0;
        //прогоним через тестовый набор и посчитаем количество ошибок
        for(int i=0;i<targetsTest.size();i++){
            float[] output=net.calculate(datasTest.get(i));
            Integer targetResult=findMax(targetsTest.get(i));
            Integer outputResult=findMax(output);
            if (targetResult!=outputResult){
                countError++;
                //System.out.println("targetResult="+targetResult+", outputResult="+outputResult);
            }
        }
        Integer len = targetsTest.size();
        float eff=(float)(len - countError)/len;
        System.out.println("eff="+eff);

        Net.save("file_net",net, eff);
    }


    public static Integer findMax(float[] arr){
        float max=-999F;
        Integer imax=-1;
        for(int i=0;i<arr.length;i++)
            if (arr[i]>max){
                max=arr[i];
                imax=i;
            }
        return imax;
    }

    public static void prepareData(List<float[]> targets, List<float[]> datas, String fileName) throws IOException, URISyntaxException {
        List<String>  list = Files.readAllLines(Paths.get(Base.class.getClassLoader().getResource(fileName).toURI()));
        for(String s : list){
            String[] sarr=s.split(",");

            float[] target = new float[10];
            Arrays.fill(target, 0.01F);
            target[Integer.valueOf(sarr[0])]=0.99F;
            targets.add(target);

            float[] data = new float[sarr.length-1];
            for(int j=1; j<sarr.length; j++)
                data[j-1]=(Float.valueOf(sarr[j]) / 255.0F) * 0.99F + 0.01F;

            datas.add(data);
        }
    }
}
