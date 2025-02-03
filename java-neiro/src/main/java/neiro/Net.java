package neiro;

import java.io.*;
import java.time.Duration;
import java.time.Instant;
import java.util.*;

/**Описывает нейросеть как последовательность слоёв нейронов где каждый нейрон слоя N связан с каждым нейроном слоя N+1*/
public class Net implements Serializable {
    Layer inputLayer; //входной слой
    Layer outputLayer;//выходной слой
    List<Neiro> biasNeiro=new ArrayList<>(); //множество нейронов смещения. Может быть пусто
    List<Layer> layers; //последоватлеьность слоёв

    //параметры модели и результаты после цикла обучения
    float lr;
    float moment;
    Double mse;
    Integer epoch;

    /**Сохранить обученную нейросеть в файл по его пути
     * @param filePath - путь к файлу в который сохраняем
     * @param net - сохраняемая нейросеть
     * @param eff - эффективность по результатам тестирования*/
    public static void save(String filePath, Net net, float eff) {
        filePath+=(net.biasNeiro.size()>0 ? "_addBias" : "_noBias")+"_"+"h"+(net.layers.size()-2)+"x"+net.layers.get(1).neiros.size()+"_"+"e"+net.epoch+"_"+net.mse+"_"+eff;
        try {
            var fileOutput = new FileOutputStream(filePath);
            var objectOutput = new ObjectOutputStream(fileOutput);
            objectOutput.writeObject(net);
            fileOutput.flush();
            objectOutput.flush();
        }catch (Exception e){
            throw new RuntimeException(e);
        }
        System.out.println("saving net in "+filePath);
    }
    /**Загрузить обученную нейросеть из файла
     * @param filePath - путь к файлу из которого загружаем нейронку
     * @return - нейросеть*/
    public static Net load(String filePath) {
        try {
            var fileInput = new FileInputStream(filePath);
            var objectInput = new ObjectInputStream(fileInput);
            Net net = (Net) objectInput.readObject();
            fileInput.close();
            objectInput.close();
            System.out.println("loading net from "+filePath);
            return net;
        }catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    public Net(List<Layer> layers, List<Neiro> biasNeiro) {
        this.inputLayer = layers.getFirst();
        this.outputLayer = layers.getLast();

        this.layers=layers;
        if (biasNeiro!=null) this.biasNeiro=biasNeiro;
    }


    /**
     * Поместить входные данные в нейроны входного слоя
     *
     * @param data - данные
     */
    public void setData(float[] data) {
        int i=0;
        for (Neiro neiro : inputLayer.neiros) {
            neiro.setValue(data[i++]);
            if (i>=data.length) break;
        }
    }

    /**
     * Получить данные с нейронов выходного слоя
     *
     * @return - массив значений
     */
    public float[] getData() {
        float[] result = new float[this.outputLayer.neiros.size()];
        int i = 0;
        for (Neiro neiro : outputLayer.neiros)
            result[i++] = neiro.getValue();
        return result;
    }


    /**Прямое распространение сигнала от входного слоя к выходному
     * @param data - входные данные для нейронов входного слоя
     * @return - массив данных полученных на нейронах выходного слоя*/
    public float[] calculate(float[] data){
        setData(data);

        for(Layer layer : layers)
            calculateNeiros(layer.neiros);

        return getData();
    }

    /**Распространить сигнал по входному множеству нейронов
     * @param currentNeiros -множесто нейронов по которому распространяем сигнал*/
    private void calculateNeiros(List<Neiro> currentNeiros) {
        //Бежим по всем нейронам текущего множества
        for (Neiro neiro : currentNeiros) {
            neiro.forward();//по входному значению с помощью фун активации вычисляем выходное значение
            for (Link link : neiro.outputs) {//передаём это выходное значение по всем связям в связанные нейроны следующего слоя
                Neiro outputNeiro = link.output;
                outputNeiro.recive(neiro.getValue() * link.weight);
            }
        }
    }

    /**Обратное респространение ошибки от нейронов выходного слоя к нейронам входного слоя с коррекцией весов связей участвующих в процессе
     * @param target - правильный выход нейронов выходного слоя. Обучение с учителем
     * @param lr - коэффициент обучения
     * @param moment -момент*/
    private void backward(float[] target, float lr, float moment) {
        //считаем дельту на нейронах выходного слоя
        int i = 0;
        for (Neiro neiro : outputLayer.neiros) {
            float err = target[i++] - neiro.getValue();
            neiro.delta = err * neiro.derivative(neiro.valueOutput);
        }

        //считаем дельту для нейронов скрытых слоёв сзади наперёд пока скрытые слои не кончатся
        for(int j=layers.size()-2; j>=1; j--){
            Layer layer = layers.get(j);
            for (Neiro neiro : layer.neiros) {
                if (!neiro.typeEnum.equals(TypeEnum.HIDDEN))
                    continue;

                neiro.delta = 0F;
                for (Link link : neiro.outputs) {
                    Neiro neiroOutput = link.output;
                    neiro.delta += neiroOutput.delta * link.weight;

                    //Также, для каждой исходящией связи данного нейрона сразу скорректируем её вес так как все данные у нас для этого уже есть
                    link.weightCorrect(lr, moment);
                }
                neiro.delta = neiro.derivative(neiro.valueOutput) * neiro.delta;
            }
        }

        //для нейронов входного слоя корректируем веса
        for (Neiro neiro : inputLayer.neiros)
            for (Link link : neiro.outputs)
                link.weightCorrect(lr, moment);
    }

    /**Цикл тренировки по одному вектору входных и выходных данных. Сначала прямой проход, потом обратное распространение ошибки
     * @param data - входные данные для нейронов входного слоя
     * @param target - ожидаемый правильный выход нейронов выходного слоя. Обучение с учителем
     * @param lr - коэффициент обучения
     * @param moment -момент
     * @return - квадрат ошибки суммированной по всем нейронам выходного слоя*/
    public float train(float[] data, float[] target, float lr, float moment){
        if (!biasNeiro.isEmpty()) calculateNeiros(biasNeiro);
        float[] output=calculate(data);

        backward(target, lr, moment);

        //вычислим ошибку на данном наборе данных
        float e=0F;
        for (int i=0;i<target.length;i++)
            e = Math.abs(target[i] - output[i]);
        return e*e;
    }


    /**Циклы тренировки по набору входных и выходных данных.
     * @param datas - набор входных данные для нейронов входного слоя
     * @param targets - набор правильных выходных данных на нейронах выходного слоя. Обучение с учителем
     * @param lr - коэффициент обучения
     * @param moment -момент
     * @param epoch - количество эпох. То есть полных повторов всех циклов обучения на переданных наборов входных и правильных выходных данных
     * @param minMse - минимальная ошибка. При достижении обучение прекращается
     * @return - квадрат ошибки суммированной по всем нейронам выходного слоя*/
    public void trains(List<float[]> datas, List<float[]> targets, float lr, float moment, Integer epoch, float minMse){
        if (datas.size()!=targets.size())
            throw new RuntimeException("Размерность множества данных для обучения не совпадает с размерностью множества ответов для обучения");

        this.lr=lr;
        this.moment=moment;
        this.epoch=epoch;
        this.mse=999.0;

        while (epoch-->0) {
            Instant start = Instant.now();

            Double mse = 0.0;
            for (int i = 0; i < targets.size(); i++) {
                mse += train(datas.get(i), targets.get(i), this.lr, this.moment);
                if (i>0 && i%5000==0)
                    System.out.println("processed "+i+" from "+targets.size());
            }
            mse=mse / targets.size();
            System.out.println("epoch="+epoch+", mse="+mse+", duration(ms)="+Duration.between(start, Instant.now()).toMillis());

            if (this.mse<mse){ //мы не смогли снизить ошибку на этой эпохе обучения. Подкоректируем гиперпараметры сети
                this.lr=this.lr/2;
                this.moment=this.moment/2;
                System.out.println("New set lr="+this.lr+", moment="+this.moment);
            }
            this.mse=mse;

            if (mse<=minMse) break;
        }
    }
}
