package neiro4;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
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


        Layer inputLayer = Layer.createLayerInput(784, FunEnum.NONE, true);
        Layer hiddenLayer = Layer.createLayerHidden(100, FunEnum.SIGMOID, inputLayer, true);
        Layer outputLayer = Layer.createLayerOutput(10, FunEnum.SIGMOID, hiddenLayer);
        List<Neiro> list= new LinkedList<>();
        list.addAll(inputLayer.neiros);
        list.addAll(hiddenLayer.neiros);
        list.addAll(outputLayer.neiros);
        Net net = new Net(list);
        net.trains(datasTrain, targetsTrain,0.7F, 0.1F,6, 0.01F);

        /*
        Net net = Net.load("file_net_v4_e1_0.018027014310724426_0.9402");
        //пробежимся по всем связям и "разобьём те, которые имеют слишком маленький вес" и посмотри как упрощение нейросети повлияет на точность итогового результата
        int contDeletedLink = 0;
        for (Neiro neiro : net.neiros)
            for (Link link : Set.copyOf(neiro.outputs))
                if (Math.abs(link.weight) < 0.01) {
                    contDeletedLink++;
                    link.terminate();
                }
        System.out.println("Link deleted: " + contDeletedLink);
        net.trains(datasTrain, targetsTrain, 0.07F, 0.01F, 1, 0.01F);
        //Link deleted: 1483
        //epoch=0, mse=0.009201660757999096, duration(ms)=140271
        //eff=0.9617
        */

        //Тестирование
        Integer countError = 0;
        //прогоним через тестовый набор и посчитаем количество ошибок
        for (int i = 0; i < targetsTest.size(); i++) {
            float[] output = net.calculate(datasTest.get(i), 10);
            Integer targetResult = findMax(targetsTest.get(i));
            Integer outputResult = findMax(output);
            if (targetResult != outputResult) {
                countError++;
                //System.out.println("targetResult="+targetResult+", outputResult="+outputResult);
            }
        }
        Integer len = targetsTest.size();
        float eff = (float) (len - countError) / len;
        System.out.println("eff=" + eff);

        Net.save("file_net_v4", net, eff);
    }


    public static Integer findMax(float[] arr) {
        float max = -999F;
        Integer imax = -1;
        for (int i = 0; i < arr.length; i++)
            if (arr[i] > max) {
                max = arr[i];
                imax = i;
            }
        return imax;
    }

    public static void prepareData(List<float[]> targets, List<float[]> datas, String fileName) throws IOException, URISyntaxException {
        List<String> list = Files.readAllLines(Paths.get(Base.class.getClassLoader().getResource(fileName).toURI()));
        for (String s : list) {
            String[] sarr = s.split(",");

            float[] target = new float[10];
            Arrays.fill(target, 0.01F);
            target[Integer.valueOf(sarr[0])] = 0.99F;
            targets.add(target);

            float[] data = new float[sarr.length - 1];
            for (int j = 1; j < sarr.length; j++)
                data[j - 1] = (Float.valueOf(sarr[j]) / 255.0F) * 0.99F + 0.01F;

            datas.add(data);
        }
    }
}
